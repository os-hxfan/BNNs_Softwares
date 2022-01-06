import os
import numpy as np
import torch
import shutil
import random
import pickle
import torch.nn.functional as F
import sys
import time
import logging
from torch.utils.tensorboard import SummaryWriter
import torch.backends.cudnn as cudnn
import re
import shutil

class Flatten(torch.nn.Module):
  def __init__(self):
    super(Flatten, self).__init__()
    
  def forward(self, x):
    if len(x.shape)==1:
      return x.unsqueeze(dim=0)
    return x.reshape(x.size(0), -1)

class Add(torch.nn.Module):
  def __init__(self):
    super(Add, self).__init__()
    self.add = torch.nn.quantized.FloatFunctional()

  def forward(self, x, y):
    return self.add.add(x,y)

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [atoi(c) for c in re.split(r'(-?\d+)', text)]

def size_of_model(model):
    torch.save(model.state_dict(), "temp.p")
    size = os.path.getsize("temp.p")/1e6
    os.remove('temp.p')
    return size

class AverageMeter(object):
  def __init__(self):
      self.reset()

  def reset(self):
      self.avg = 0.0
      self.sum = 0.0
      self.cnt = 0.0

  def update(self, val, n=1):
      self.sum += val * n
      self.cnt += n
      self.avg = self.sum / self.cnt
    

def save_model(model, args, special_info=""):
  torch.save(model.state_dict(), os.path.join(args.save, 'weights'+special_info+'.pt'))

  with open(os.path.join(args.save, 'args.pt'), 'wb') as handle:
    pickle.dump(args, handle, protocol=pickle.HIGHEST_PROTOCOL)

def entropy(output):
  batch_size = output.shape[0]
  entropy = -torch.sum(torch.log(output+1e-8)*output)/batch_size
  return entropy.item()

def ece(output, target):
    _ece = 0.0
    confidences, predictions = torch.max(output, 1)
    accuracies = predictions.eq(target)
        
    bin_boundaries = torch.linspace(0, 1, 10 + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
      in_bin = confidences.gt(bin_lower.item()) * \
          confidences.le(bin_upper.item())
      prop_in_bin = in_bin.float().mean()
      if prop_in_bin.item() > 0:
        accuracy_in_bin = accuracies[in_bin].float().mean()
        avg_confidence_in_bin = confidences[in_bin].mean()
        _ece += torch.abs(avg_confidence_in_bin -
                            accuracy_in_bin) * prop_in_bin
    _ece = _ece if isinstance(_ece, float) else _ece.item()
    return _ece

def evaluate(output, input, target, model, args):
  with torch.no_grad():
    if model is not None and args.samples is not None and args.samples>1 and model.training is False:
      y = [output]
      for i in range(1, args.samples):
        out = model(input)
        y.append(out)
        if i>=3 and args.debug:
            break
      output = torch.stack(y, dim=1).mean(dim=1)
    _loss = F.nll_loss(torch.log(output), target).item()
    _, pred = output.topk(1, 1, True, True)
    pred = pred.t()
    _ece = ece(output, target)
    _entropy = entropy(output)
    _error = error(pred, target)
    return _error, _ece, _entropy, _loss

def error(output, target):
    batch_size = output.shape[1]
    correct = output.eq(target.view(1, -1).expand_as(output))
    correct_k = correct[:1].view(-1).float().sum(0)
    res = 100-correct_k.mul_(100.0/batch_size)
    return res.float().item()

def save_pickle(data, path, overwrite=False):
  path = check_path(path) if not overwrite else path
  with open(path, 'wb') as fp:
      pickle.dump(data, fp, protocol=pickle.HIGHEST_PROTOCOL)

def load_pickle(path):
    file = open(path, 'rb')
    return pickle.load(file)


def load_model(model, model_path, replace=True):
  state_dict = torch.load(model_path, map_location=torch.device('cpu'))
  model_dict = model.state_dict()
  pretrained_dict = {}
  for k,v in state_dict.items():
    _k = k.replace('module.', '')
    pretrained_dict[_k] = v
  pretrained_dict = {k: v for k,v in pretrained_dict.items() if k in model_dict.keys()}
  model_dict.update(pretrained_dict)
  model.load_state_dict(model_dict)
    
def create_exp_dir(path):
  path = check_path(path)
  os.mkdir(path)

def check_path(path):
  if os.path.exists(path):
    filename, file_extension = os.path.splitext(path)
    counter = 0
    while os.path.exists(filename+"_"+str(counter)+file_extension):
      counter+=1
    return filename+"_"+str(counter)+file_extension
  return path


def model_to_gpus(model, args):
  if args.gpu!= -1:
    device = torch.device("cuda:"+str(args.gpu))
    model = model.to(device)
  return model

def parse_args(args, label=""):
  loading_path = args.save
  dataset = args.dataset if hasattr(args, 'dataset') else ""
  model = args.model if hasattr(args, 'model') else ""
  if label == "":
    q = "quant" if args.q else "not_quant"
  else:
    q = label
  new_path = '{}-{}-{}-{}'.format(q,dataset, model, time.strftime("%Y%m%d-%H%M%S"))
  
  create_exp_dir(
    new_path)
  args.save = new_path
  if loading_path!="EXP":
    for root, dirs, files in os.walk(loading_path):
        for filename in files:
          if ".pt" in filename:
            shutil.copy(os.path.join(loading_path, filename), os.path.join(new_path, filename))
  
  log_format = '%(asctime)s %(message)s'
  logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                      format=log_format, datefmt='%m/%d %I:%M:%S %p')
  log_path = os.path.join(args.save, 'log.log')
  log_path = check_path(log_path)

  fh = logging.FileHandler(log_path)
  fh.setFormatter(logging.Formatter(log_format))
  logging.getLogger().addHandler(fh)
  
  print('Experiment dir : {}'.format(args.save))

  writer = SummaryWriter(
      log_dir=args.save+"/",max_queue=5)
  
  if torch.cuda.is_available() and args.gpu!=-1:
    logging.info('## GPUs available = {} ##'.format(args.gpu))
    torch.cuda.set_device(args.gpu)
    cudnn.benchmark = True
    cudnn.enabled = True
    torch.cuda.manual_seed(args.seed)
  else:
    logging.info('## No GPUs detected ##')
  
  seed = args.seed if hasattr(args,'seed') else 1
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  logging.info("## Args = %s ##", args)

  path = os.path.join(args.save, 'results.pickle')
  path= check_path(path)
  results = {}
  results["dataset"] = args.dataset if hasattr(args, 'dataset') else ""
  results["model"] = args.model if hasattr(args, 'model') else ""
  results["error"] = {}
  results["nll"] = {}
  results["latency"] = {}
  results["ece"] = {}
  results["entropy"] = {}
  save_pickle(results, path, True)
    
  return args, writer

      

          
  
