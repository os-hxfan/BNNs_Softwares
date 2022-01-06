import sys
import argparse
import logging
import torch
import numpy as np

sys.path.append("../")
sys.path.append("../../")
sys.path.append("../../../")
sys.path.append("../../../../")
sys.path.append("../../../../../")

from software.experiments.utils import _evaluate_with_loader
from software.data import get_test_loader
from software.models import ModelFactory
from software.plots import PLT as plt
import software.utils as utils



parser = argparse.ArgumentParser("compare_confidence")

parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--data', type=str,
                    default='./../experiments/data', help='experiment name')

parser.add_argument('--label', type=str, default='compare_confidence',
                    help='default experiment category ')

parser.add_argument('--bayesian_model', type=str, default='',
                    help='default experiment category ')
parser.add_argument('--pointwise_model', type=str, default='',
                    help='default experiment category ')
parser.add_argument('--dataset', type=str, default='',
                    help='default experiment category ')


parser.add_argument('--batch_size', type=int, default=128,
                    help='default batch size')
parser.add_argument('--p', type=float, default=0.25,
                    help='dropout probability')
parser.add_argument('--seed', type=int, default=1,
                    help='dropout probability')
parser.add_argument('--num_workers', type=int,
                    default=0, help='default batch size')

parser.add_argument('--gpu', type=int, default=-1, help='gpu device ids')
parser.add_argument('--q', action='store_true',
                    help='whether to do post training quantisation')
parser.add_argument('--debug', action='store_true',
                    help='whether we are currently debugging')


def _plot_model_certainty(outputs, rel_ax, n_bins=20, color='b', label='Pointwise model'):
    confidences, _ = outputs.max(1)
    confidences = np.nan_to_num(confidences)
    bin_height, bin_boundary = np.histogram(confidences, bins=n_bins)
    width = bin_boundary[1]-bin_boundary[0]
    bin_height = bin_height/float(sum(bin_height))
    rel_ax.bar(bin_boundary[:-1], bin_height, width=width,
               align='center', color=color, label=label, alpha=0.5)


def main():
  args = parser.parse_args()
  logging.info('## Testing distortions ##')
  args, _ = utils.parse_args(args, args.label)

  model_temp = ModelFactory.get_model
  logging.info('## Downloading and preparing data ##')

  with torch.no_grad():
    model = ""
    input_size = ()
    if args.dataset == "cifar":
        model = "resnet"
        args.input_size = (1, 3, 32, 32)
    elif args.dataset == "svhn":
        model = "vgg"
        args.input_size = (1, 3, 32, 32)
    elif args.dataset == "mnist":
        model = "lenet"
        args.input_size = (1, 1, 28, 28)

    bayesian_model = model_temp(model+"_all", args.input_size,
                                10, args)
    bayesian_model = utils.model_to_gpus(bayesian_model, args)
    utils.load_model(bayesian_model, args.bayesian_model+"/weights.pt")
    bayesian_model.eval()

    pointwise_model = model_temp(model+"_p", args.input_size,
                                 10, args)
    pointwise_model = utils.model_to_gpus(pointwise_model, args)
    utils.load_model(pointwise_model, args.pointwise_model+"/weights.pt")
    pointwise_model.eval()

    logging.info('## Models re-created: ##')
    logging.info(bayesian_model.__repr__())
    logging.info(pointwise_model.__repr__())

    f, rel_ax = plt.subplots(1, 1, figsize=(6, 2.1))
    args.dataset = 'random_'+args.dataset
    test_loader = get_test_loader(args)

    args.samples = 100
    _, _, _, _, outputs, _ = _evaluate_with_loader(
        test_loader, bayesian_model, args)
    _plot_model_certainty(outputs, rel_ax, color='blue',
                          label="Bayesian neural network")

    args.samples = 1
    _, _, _, _, outputs, _ = _evaluate_with_loader(
        test_loader, pointwise_model, args)
    _plot_model_certainty(outputs, rel_ax, color='black',
                          label="Standard neural network")

    rel_ax.set_xlabel('Confidence', fontweight='bold')
    rel_ax.set_ylabel('Normalized frequency', fontweight='bold')

    plt.legend()
    plt.savefig(args.save+"/confidence_comparison.pdf",  bbox_inches='tight')

    logging.info('# Finished #')


if __name__ == '__main__':
  main()
