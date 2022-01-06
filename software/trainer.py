
import torch
from torch.autograd import Variable
import software.utils as utils
import time 
import logging

class Trainer():
  def __init__(self, model, criterion, optimizer, scheduler, args):
    super().__init__()
    self.model = model
    self.criterion = criterion
    self.optimizer = optimizer
    self.scheduler = scheduler
    self.args = args
    
    self.train_step = 0
    self.train_time = 0.0
    self.val_step = 0
    self.val_time = 0.0

  def _scalar_logging(self, obj, main_obj, nll, error_metric, ece, entropy, info, iteration, writer):
    writer.add_scalar(info+'error', error_metric, iteration)
    writer.add_scalar(info+'loss', obj, iteration)
    writer.add_scalar(info+'smoothed_ce', main_obj, iteration)
    writer.add_scalar(info+'ece', ece, iteration)
    writer.add_scalar(info+'entropy', entropy, iteration)
    writer.add_scalar(info+'nll', nll, iteration)
    
  def _get_average_meters(self):
    error_metric = utils.AverageMeter()
    obj = utils.AverageMeter()
    main_obj = utils.AverageMeter()
    nll = utils.AverageMeter()
    ece = utils.AverageMeter()
    entropy = utils.AverageMeter()
    return error_metric, obj, main_obj, nll, ece, entropy
    
  def train_loop(self, train_loader, valid_loader, writer=None, special_infor=""):
    best_error = float('inf')
    train_error_metric = train_obj = train_main_obj = train_nll = train_ece = train_entropy  = None
    val_error_metric = val_obj = val_main_obj = val_nll = val_ece = val_entropy = None

    for epoch in range(self.args.epochs):
      if epoch >= 1 and self.scheduler is not None:
        self.scheduler.step()
      
      if self.scheduler is not None:
        lr = self.scheduler.get_last_lr()[0]
      else:
        lr = self.args.learning_rate

      if writer is not None:
        writer.add_scalar('Train/learning_rate', lr, epoch)

      logging.info(
          '### Epoch: [%d/%d], Learning rate: %e ###', self.args.epochs,
          epoch, lr)
   
      train_obj, train_main_obj, train_nll, train_error_metric, train_ece, train_entropy = self.train(epoch, train_loader, self.optimizer, writer)
      
      logging.info('#### Train | Error: %f, Train loss: %f, Train main objective: %f, Train NLL: %f, Train ECE: %f, Train entropy: %f ####',
                     train_error_metric, train_obj, train_main_obj, train_nll, train_ece, train_entropy)

      
      if writer is not None:
        self._scalar_logging(train_obj, train_main_obj, train_nll, train_error_metric, train_ece, train_entropy, "Train/", epoch, writer)
    
      # validation
      if valid_loader is not None:
        val_obj, val_main_obj, val_nll, val_error_metric, val_ece, val_entropy= self.infer(epoch,
                                                          valid_loader, writer, "Valid")
        logging.info('#### Valid | Error: %f, Valid loss: %f, Valid main objective: %f, Valid NLL: %f, Valid ECE: %f, Valid entropy: %f ####',
                      val_error_metric, val_obj, val_main_obj, val_nll, val_ece, val_entropy)
        
        if writer is not None:
          self._scalar_logging(val_obj, val_main_obj, val_nll, val_error_metric, val_ece, val_entropy, "Valid/", epoch, writer)
      
      if self.args.save_last or val_error_metric <= best_error:
        utils.save_model(self.model, self.args, special_infor)
        best_error = val_error_metric
        logging.info(
            '### Epoch: [%d/%d], Saving model! Current best error: %f ###', self.args.epochs,
            epoch, best_error)
      
    return best_error, self.train_time, self.val_time
  
  def _step(self, input, target, optimizer, epoch, n_batches, n_points, train_timer):
    start = time.time()

    input = Variable(input, requires_grad=False)
    target = Variable(target, requires_grad=False)
    if next(self.model.parameters()).is_cuda:
      input = input.cuda()
      target = target.cuda()
    if optimizer is not None:
      optimizer.zero_grad()
    output = self.model(input)

    obj, main_obj = self.criterion(output, target, self.model, n_batches, n_points)

    if optimizer is not None and obj == obj:
      obj.backward()
      if self.args.clip>0:
        torch.nn.utils.clip_grad_value_(self.model.parameters(), self.args.clip)
      for p in self.model.parameters():
        if p.grad is not None:
          p.grad[p.grad != p.grad] = 0
      optimizer.step()
      
    error_metric, ece, entropy, nll = utils.evaluate(output, input, target, self.model, self.args)

    if train_timer:
      self.train_time += time.time() - start
    else:
      self.val_time += time.time() - start
      
    return error_metric, obj.item(), main_obj.item(), nll, ece, entropy


  def train(self, epoch, loader, optimizer, writer):
    error_metric, obj, main_obj, nll, ece, entropy = self._get_average_meters()
    self.model.train()

    if hasattr(self.model, 'qat_hook'):
      self.model.qat_hook(epoch)
    
    for step, (input, target) in enumerate(loader):
      n = input.shape[0]
      _error_metric, _obj, _main_obj, _nll, _ece, _entropy= self._step(input, target, optimizer, epoch, len(loader), len(loader.dataset), True)
      
      obj.update(_obj, n)
      main_obj.update(_main_obj, n)
      nll.update(_nll, n)
      error_metric.update(_error_metric, n)
      ece.update(_ece, n)
      entropy.update(_entropy, n)

      if step % self.args.report_freq == 0:
        logging.info('##### Train step: [%03d/%03d] | Error: %f, Loss: %f, Main objective: %f, NLL: %f, ECE: %f, Entropy: %f #####',
                       len(loader),  step, error_metric.avg, obj.avg, main_obj.avg, nll.avg, ece.avg, entropy.avg)
        if writer is not None:        
          self._scalar_logging(obj.avg, main_obj.avg, nll.avg, error_metric.avg, ece.avg, entropy.avg, 'Train/Iteration/', self.train_step, writer)
        self.train_step += 1
      
      if self.args.debug:
          break
    
    return obj.avg, main_obj.avg, nll.avg, error_metric.avg, ece.avg, entropy.avg

  def infer(self, epoch, loader, writer, dataset="Valid"):
    with torch.no_grad():
      error_metric, obj, main_obj, nll, ece, entropy= self._get_average_meters()
      self.model.eval()

      for step, (input, target) in enumerate(loader):
        n = input.shape[0]
        _error_metric, _obj, _main_obj, _nll, _ece, _entropy = self._step(
             input, target, None, epoch, len(loader), n * len(loader), False)

        obj.update(_obj, n)
        main_obj.update(_main_obj, n)
        nll.update(_nll, n)
        error_metric.update(_error_metric, n)
        ece.update(_ece, n)
        entropy.update(_entropy, n)
        
        if step % self.args.report_freq == 0:
          logging.info('##### {} step: [{}/{}] | Error: {}, Loss: {}, Main objective: {}, NLL: {}, ECE: {}, Entropy: {} #####'.format(
                       dataset, len(loader), step, error_metric.avg, obj.avg, main_obj.avg, nll.avg, ece.avg, entropy.avg))
          if writer is not None:
            self._scalar_logging(obj.avg, main_obj.avg, nll.avg, error_metric.avg, ece.avg, entropy.avg, '{}/Iteration/'.format(dataset), self.val_step, writer)
          self.val_step += 1

        if self.args.debug:
          break
          

      return obj.avg, main_obj.avg, nll.avg, error_metric.avg, ece.avg, entropy.avg
