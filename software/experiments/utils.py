import numpy as np
import torch
import sys
import logging
sys.path.append('../')
from software.data import *
import software.utils as utils
from software.plots import PLT as plt

METRICS = ['entropy','ece','error','nll']
LEVELS = 5

def classification_evaluation(model, args):
    results = utils.load_pickle(args.save+"/results.pickle")
    outputs, targets = _evaluate_and_record(model, results, args)

    f = _plot_ece(outputs, targets, plt)
    path = utils.check_path(args.save+'/ece_test.png')
    plt.savefig(path)
    f = _plot_model_certainty(outputs, plt)
    path = utils.check_path(args.save+'/certainty_test.png')
    plt.savefig(path)

    orig_dataset = args.dataset
    args.dataset= 'random_'+orig_dataset
    test_loader = get_test_loader(args)
    error, ece, entropy, nll, outputs, targets = _evaluate_with_loader(test_loader, model, args)
    logging.info("## Random Error: {} ##".format(error))
    logging.info("## Random ECE: {} ##".format(ece))
    logging.info("## Random Entropy: {} ##".format(entropy))
    logging.info("## Random NLL: {} ##".format(nll))

    results["entropy"]["random"] = entropy
    results["ece"]["random"] = ece
    results["error"]["random"] = error
    results["nll"]["random"] = nll
    f = _plot_ece(outputs, targets, plt)
    path = utils.check_path(args.save+'/ece_random.png')
    plt.savefig(path)
    f = _plot_model_certainty(outputs, plt)
    path = utils.check_path(args.save+'/certainty_random.png')
    plt.savefig(path)

    utils.save_pickle(results, args.save+"/results.pickle", True)
    logging.info("## Results: {} ##".format(results))

def _plot_ece(outputs, labels, plt, n_bins=10):
    confidences, predictions = outputs.max(1)
    accuracies = torch.eq(predictions, labels)
    f, rel_ax = plt.subplots(1, 1, figsize=(4, 2.5))

    bins = torch.linspace(0, 1, n_bins + 1)

    bins[-1] = 1.0001
    width = bins[1] - bins[0]
    bin_indices = [confidences.ge(bin_lower) * confidences.lt(bin_upper) for bin_lower, bin_upper in zip(bins[:-1], bins[1:])]
    bin_corrects = np.nan_to_num(np.array([torch.mean(accuracies[bin_index].float()) for bin_index in bin_indices]))
    bin_scores = np.nan_to_num(np.array([torch.mean(confidences[bin_index]) for bin_index in bin_indices]))
  
    confs = rel_ax.bar(bins[:-1], np.array(bin_corrects), align='edge', width=width, alpha=0.75, edgecolor='b')
    gaps = rel_ax.bar(bins[:-1], bin_scores -bin_corrects, align='edge',
                      bottom=bin_corrects, color=[1, 0.7, 0.7], alpha=0.5, width=width, hatch='//', edgecolor='r')
    rel_ax.plot([0, 1], [0, 1], '--', color='gray')
    rel_ax.legend([confs, gaps], ['Outputs', 'Gap'], loc='best', fontsize='small')
    rel_ax.set_ylabel('Accuracy')
    rel_ax.set_xlabel('Confidence')
    plt.tight_layout()
    return f

def _plot_model_certainty(outputs, plt, n_bins=10):
    confidences, _ = outputs.max(1)
    confidences = np.nan_to_num(confidences)
    f, rel_ax = plt.subplots(1, 1, figsize=(4, 2.5))
    bin_height,bin_boundary = np.histogram(confidences,bins=n_bins)
    width = bin_boundary[1]-bin_boundary[0]
    bin_height = bin_height/float(max(bin_height))
    rel_ax.bar(bin_boundary[:-1],bin_height,width = width, align='center', color='b', label="Normalized counts")
    rel_ax.legend()
    rel_ax.set_xlabel('Confidence')
    f.tight_layout()

    return f

def _evaluate_with_loader(loader, model, args):
    outputs = []
    targets = []
    for i, (input, target) in enumerate(loader):
      input = torch.autograd.Variable(input, requires_grad=False)
      target = torch.autograd.Variable(target, requires_grad=False)
      if not args.q and next(model.parameters()).is_cuda:
        input = input.cuda()
        target = target.cuda()
      samples = []
      for j in range(args.samples):
        out = model(input).detach()
        samples.append(out)
        if j >= 2 and args.debug:
            break
      outputs.append(torch.stack(samples,dim=1).mean(dim=1))
      targets.append(target)
      if args.debug:
        break
    outputs = torch.cat(outputs, dim=0).cpu()
    targets = torch.cat(targets, dim=0).cpu()
    error, ece, entropy, loss = utils.evaluate(outputs, None, targets, None, args)
    return error, ece, entropy, loss, outputs, targets

def _evaluate_and_record(model, results, args, train=False, valid=False, test=True):
    train_loader, val_loader = get_train_loaders(args)
    test_loader = get_test_loader(args)

    if train:
      error, ece, entropy, nll, _, _ = _evaluate_with_loader(train_loader, model, args)
      logging.info("## Train Error: {} ##".format(error))
      logging.info("## Train ECE: {} ##".format(ece))
      logging.info("## Train Entropy: {} ##".format(entropy))
      logging.info("## Train NLL: {} ##".format(nll))

      results["entropy"]["train"] = entropy
      results["ece"]["train"] = ece
      results["error"]["train"] = error
      results["nll"]["train"] = nll

    if valid:
      error, ece, entropy, nll, _, _= _evaluate_with_loader(val_loader, model, args)
      logging.info("## Valid Error: {} ##".format(error))
      logging.info("## Valid ECE: {} ##".format(ece))
      logging.info("## Valid Entropy: {} ##".format(entropy))
      logging.info("## Valid NLL: {} ##".format(nll))

      results["entropy"]["valid"] = entropy
      results["ece"]["valid"] = ece
      results["error"]["valid"] = error
      results["nll"]["valid"] = nll

    if test:
      error, ece, entropy, nll, outputs, targets = _evaluate_with_loader(test_loader, model, args)
      logging.info("## Test Error: {} ##".format(error))
      logging.info("## Test ECE: {} ##".format(ece))
      logging.info("## Test Entropy: {} ##".format(entropy))
      logging.info("## Test NLL: {} ##".format(nll))

      results["entropy"]["test"] = entropy
      results["ece"]["test"] = ece
      results["error"]["test"] = error
      results["nll"]["test"] = nll
      return outputs, targets
