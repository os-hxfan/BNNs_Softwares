import sys
import argparse
import logging
import torch
import torch.autograd.profiler as profiler
from tqdm import tqdm
import os
import platform
import subprocess
import re
import gc
import numpy as np
import torch.backends.cudnn as cudnn
import random

sys.path.append("../")
sys.path.append("../../")
sys.path.append("../../../")
sys.path.append("../../../../")
sys.path.append("../../../../../")

import software.utils as utils
import software.quant as quant
from software.models import ModelFactory

parser = argparse.ArgumentParser("profile_latency")

parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--q', action='store_true',
                    help='whether the model is quantised')
parser.add_argument('--p', type=float, default=0.25,
                    help='dropout probability')
parser.add_argument('--gpu', type=int, default=1, help='gpu device ids')
parser.add_argument('--samples', type=int, default=-1, help = 'number of samples')
parser.add_argument('--model_option', type=str,
                    default='EXP', help='experiment name')

INPUT_SIZES = {"lenet": (1, 1, 28, 28), "resnet": (1, 3, 32, 32), "vgg": (1, 3, 32, 32)}
def get_processor_name():
    if platform.system() == "Windows":
        return platform.processor()
    elif platform.system() == "Darwin":
        os.environ['PATH'] = os.environ['PATH'] + os.pathsep + '/usr/sbin'
        command = "sysctl -n machdep.cpu.brand_string"
        return subprocess.check_output(command).strip()
    elif platform.system() == "Linux":
        command = "cat /proc/cpuinfo"
        all_info = str(subprocess.check_output(command, shell=True)).strip()
        for line in all_info.split("\\n"):
            if "model name" in line:
                return re.sub(".*model name.*:", "", line, 1)
        raise Exception()
    return ""

def process_line(line, cpu=True):
  if cpu:
    line = line.split()[5]
  else:
    line = line.split()[8]
  unit = line[-2:]
  value = float(line[:-2])
  if unit =="us":
    value/=1000
  return value

def profile(model, input_size, samples, cpu=True, caching=True):
  model.eval()
  size = utils.size_of_model(model)
  x = torch.randn(input_size, requires_grad=False)
  N_ITER = 100

  if not cpu:
      x = x.cuda()   
  with profiler.profile(record_shapes=False, use_cuda=not cpu) as prof:
    for i in tqdm(range(N_ITER)):
      with profiler.record_function("model_evaluation"):
        with torch.no_grad():
          y= model(x, samples, caching)
      del y
  if cpu:
      prefix="cpu"
  else:
      prefix="cuda"
      torch.cuda.empty_cache()
  s = prof.key_averages().table(sort_by=prefix+"_time_total")
  logging.info(s) 
  s = s.split('\n')
  eval_time = static_time = dynamic_time = 0
  for line in s:
    if "model_evaluation" in line:
      eval_time = process_line(line, cpu)
      if not caching:
        eval_time *= samples
    elif "static_part" in line:
      static_time = process_line(line, cpu)
    elif "dynamic_part" in line:
      dynamic_time = process_line(line, cpu) 

  gc.collect()
  return size, eval_time, static_time, dynamic_time


def set_dict_value(dictionary, value, path=[]):
    if len(path) == 1:
        dictionary[path[0]] = value
    else:
        if not path[0] in dictionary:
            dictionary[path[0]] = {}
        set_dict_value(dictionary[path[0]], value, path[1:])


def main():
  args = parser.parse_args()
  if not os.path.exists(args.save):
    os.mkdir(args.save)

  log_format = '%(asctime)s %(message)s'
  logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                      format=log_format, datefmt='%m/%d %I:%M:%S %p')
  log_path = os.path.join(args.save, 'log.log')
  fh = logging.FileHandler(log_path)
  fh.setFormatter(logging.Formatter(log_format))
  logging.getLogger().addHandler(fh)

  if torch.cuda.is_available() and args.gpu != -1:
    logging.info('## GPUs available = {} ##'.format(args.gpu))
    torch.cuda.set_device(args.gpu)
    cudnn.benchmark = True
    cudnn.enabled = True
    torch.cuda.manual_seed(1)
  else:
    logging.info('## No GPUs detected ##')

  seed = 1
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  logging.info("## Args = %s ##", args)

  results = {}
  if not os.path.exists(args.save+ "/results.pickle"):
    results["cpu"] = get_processor_name()
    results["gpu"] = torch.cuda.get_device_name(
      int(args.gpu)) if args.gpu != -1 else ""
    utils.save_pickle(results, args.save+"/results.pickle", True)
    
  else:
    results = utils.load_pickle(args.save+'/results.pickle')
  logging.info('# Beginning analysis #')

  model_option = args.model_option
  sample = args.samples
  if "lenet" in model_option or "mnist" in model_option:
      input_size = INPUT_SIZES["lenet"]
  elif "resnet" in model_option or "cifar" in model_option:
      input_size = INPUT_SIZES["resnet"]
  elif "vgg" in model_option or "svhn" in model_option:
      input_size = INPUT_SIZES["vgg"]
  
  for q in [False, True]:
    args.q = q
    model = ModelFactory.get_model(model_option, input_size,
                10, args)
    if q:
        quant.prepare_model(model)
        model.eval()
        torch.quantization.convert(model, inplace=True)
    logging.info('## Model created: ##')
    logging.info(model.__repr__())

    size, eval_time, static_time, dynamic_time = profile(model, input_size, sample, True)
    quantised="quant" if q else "not_quant"
    set_dict_value(results, size, [model_option, quantised, "size"])
    set_dict_value(results, eval_time, [model_option, quantised, "latency", "cpu", str(sample), "total_eval_time"])
    set_dict_value(results, static_time, [
                    model_option, quantised, "latency", "cpu", str(sample), "static_time"])
    set_dict_value(results, dynamic_time, [
                    model_option, quantised, "latency", "cpu", str(sample), "dynamic_time"])

    if not q and args.gpu!=-1:
        device = torch.device("cuda:"+str(args.gpu))
        model = model.to(device)
        _, eval_time, static_time, dynamic_time = profile(
            model, input_size, sample, False)
        set_dict_value(results, eval_time, [
                        model_option, quantised, "latency", "gpu", str(sample), "total_eval_time"])
        set_dict_value(results, static_time, [
            model_option, quantised, "latency", "gpu", str(sample), "static_time"])
        set_dict_value(results, dynamic_time, [
            model_option, quantised, "latency", "gpu", str(sample), "dynamic_time"])
    del model

  logging.info('## Results: {} ##'.format(results))
  utils.save_pickle(results, args.save+"/results.pickle", True)
  logging.info('# Finished #')

if __name__ == '__main__':
  main()
