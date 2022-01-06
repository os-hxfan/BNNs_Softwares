import sys
import argparse
import numpy as np
import logging
import copy 

sys.path.append("../")
sys.path.append("../../")
sys.path.append("../../../")
sys.path.append("../../../../")
sys.path.append("../../../../../")
sys.path.append("../../../../../../")

import software.utils as utils
from software.plots import PLT as plt

parser = argparse.ArgumentParser("compare_ood_results")

parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--all_paths', nargs='+',
                    default=[], help='experiment name')
parser.add_argument('--two_third_paths', nargs='+',
                    default=[], help='experiment name')
parser.add_argument('--half_paths', nargs='+',
                    default=[], help='experiment name')
parser.add_argument('--one_third_paths', nargs='+',
                    default=[], help='experiment name')
parser.add_argument('--ll_paths', nargs='+',
                    default=[], help='experiment name')
parser.add_argument('--profile_path', type=str,
                    default='', help='experiment name')
parser.add_argument('--profile_option', type=str,
                    default='cpu_quant', help='type of the profiling data')
parser.add_argument('--label', type=str, default='',
                    help='default experiment category ')
parser.add_argument('--dataset', type=str, default='',
                    help='default experiment category ')
parser.add_argument('--seed', type=int, default=1, help='random seed')
parser.add_argument('--debug', action='store_true',
                    help='whether we are currently debugging')

parser.add_argument('--gpu', type=int, default=0, help='gpu device ids')

def get_dict_value(dictionary, path=[], delete=False):
    if len(path) == 1:
        val = dictionary[path[0]]
        if delete:
            dictionary.pop(path[0])
        return val
    else:
        return get_dict_value(dictionary[path[0]], path[1:])


def load_data_from_path(path, label):
  if path != "" or path != "none":
      result = utils.load_pickle(path+"/results.pickle")
      return get_dict_value(result, label)
  else:
      return (0.0, 0.0)

def main():
  args = parser.parse_args()
  args, _ = utils.parse_args(args, args.label)
  logging.info('## Loading of result pickles for the experiment ##')

  label_combinations = [['error', 'test'], ['entropy', 'random'],
                        ['ece', 'test']]
  samples = [3, 4, 5, 6, 7, 8, 9, 10, 20, 50, 100]
  model_options = ["_ll", "_one_third", '_half', '_two_third', '_all']
  dataset_model = {'mnist': "lenet", 'svhn': "vgg", 'cifar': 'resnet'}
  profie_options = {'cpu_quant': ['quant', 'latency', 'cpu'],
                    'cpu_not_quant': ['not_quant', 'latency', 'cpu'],
                    'gpu_not_quant': ['not_quant', 'latency', 'gpu']}
  units = {'error': 'Accuracy [%]', 'ece': 'ECE [%]',
                   'nll': '$aNLL_{cls}$', 'entropy': 'aPE [nats]'}
  colors = ['r', 'g', 'b', 'c', 'm']
  for e,labels in enumerate(label_combinations):
    fig = plt.figure(figsize=(6, 3))
    pts = []
    for i, paths in enumerate([args.ll_paths, args.one_third_paths, args.half_paths, args.two_third_paths, args.all_paths]):
        assert len(paths) == 11 or len(paths) == 0
        time_l, main_l = [], []
        for j, path in enumerate(paths):
            main, main_std = load_data_from_path(path, labels)
            if labels[0] == 'ece':
                main *= 100
                main_std *= 100
            if main!=0.0:
                model = dataset_model[args.dataset]
                sub_model = model_options[i]
                latency_label = [model + model_options[i]] +  profie_options[args.profile_option]
                static_time = load_data_from_path(
                    args.profile_path, latency_label + [str(samples[j])] + ['static_time'])
                dynamic_time = load_data_from_path(
                    args.profile_path, latency_label + [str(samples[j])] + ['dynamic_time'])
                time = static_time + dynamic_time
                if e == 0:
                    plt.scatter([time], [100.-main], color=colors[i], alpha=0.7, s=150)
                    main_l.append(100.-main)
                else:
                    plt.scatter([time], [main], color=colors[i],
                                alpha=0.7, s=150)
                    main_l.append(main)
                time_l.append(time)
        plt.plot(time_l, main_l, color=colors[i],
                            alpha=1.)  
    plt.xlabel('Latency [ms]', fontweight ='bold') 
    plt.ylabel(units[labels[0]], fontweight ='bold')
    plt.tight_layout()

    fig.savefig(args.save+"/{}_{}.pdf".format(args.dataset,
                                              labels))



if __name__ == '__main__':
  main()
