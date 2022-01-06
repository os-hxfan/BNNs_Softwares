import sys
import argparse
import numpy as np
import logging
from math import log10, floor

sys.path.append("../")
sys.path.append("../../")
sys.path.append("../../../")
sys.path.append("../../../../")
sys.path.append("../../../../../")
sys.path.append("../../../../../../")

import software.utils as utils

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


def round_sig(x, sig=2):
    if x!=0.0:
        rounded =np.around(x, sig)
        if rounded == 0.0:
            return round(x, sig-int(floor(log10(abs(x))))-1)
        else:
            return rounded
    else:
        return x

def get_dict_value(dictionary, path=[], delete=False):
    if len(path) == 1:
        val = dictionary[path[0]]
        if delete:
            dictionary.pop(path[0])
        return val
    else:
        return get_dict_value(dictionary[path[0]], path[1:])

def latex_output(l):
    return "& \\{{{}, {}\\}} & {} & {} & {} & $ {} \pm {}$ & $ {} \pm {}$ & $ {} \pm {}$  \\\\".format(
        l[1][0], l[1][1], l[1][2], round_sig(l[1][3]), round_sig(l[1][4]), round_sig(l[1][5][0]),
        round_sig(l[1][5][1]), round_sig(l[1][6][0]), round_sig(
            l[1][6][1]), round_sig(100.-l[1][7][0]), round_sig(l[1][7][1]))


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

  samples = [3, 4, 5, 6, 7, 8, 9, 10, 20, 50, 100]
  model_options = ["_ll", "_one_third", '_half', '_two_third', '_all']
  dataset_model = {'mnist': "lenet", 'svhn': "vgg", 'cifar': 'resnet'}
  model_L = {'lenet_ll': 1, 'lenet_two_third': "$2/3 \\times N$", 'lenet_all': "$N$",
             'vgg_ll': 1, 'vgg_one_third': "$1/3 \\times N$", 'vgg_half': "$1/2 \\times N$", 'vgg_two_third': "$2/3 \\times N$", 'vgg_all': "$N$",
             'resnet_ll': 1, 'resnet_one_third': "$1/3 \\times N$", 'resnet_half': "$1/2 \\times N$",  'resnet_two_third': "$2/3 \\times N$", 'resnet_all': "$N$"}
  profie_options = {'cpu_quant': ['quant', 'latency', 'cpu'],
                    'gpu_not_quant': ['not_quant', 'latency', 'gpu']}
  optimal_latency = None 
  optimal_entropy = None 
  optimal_error = None
  optimal_ece = None
  for i, paths in enumerate([args.ll_paths, args.one_third_paths, args.half_paths, args.two_third_paths, args.all_paths]):
        assert len(paths) == 11 or len(paths) == 0
        for j, path in enumerate(paths):
            error, error_std = load_data_from_path(path, ["error", 'test'])
            if error!=0.0:
                print(path)
                entropy, entropy_std = load_data_from_path(path, ['entropy', 'random'])
                ece, ece_std = load_data_from_path(path, ['ece', 'test'])
                ece *= 100
                ece_std *= 100
                model = dataset_model[args.dataset]
                sub_model = model_options[i]

                cpu_latency_label = [model + sub_model] +  profie_options['cpu_quant']
                cpu_static_time = load_data_from_path(args.profile_path, cpu_latency_label + [str(samples[j])] + ['static_time'])
                cpu_dynamic_time = load_data_from_path(
                    args.profile_path, cpu_latency_label + [str(samples[j])] + ['dynamic_time'])
                cpu_time = cpu_static_time + cpu_dynamic_time

                gpu_latency_label = [model + model_options[i]] +  profie_options['gpu_not_quant']
                gpu_static_time = load_data_from_path(
                    args.profile_path, gpu_latency_label + [str(samples[j])] + ['static_time'])
                gpu_dynamic_time = load_data_from_path(
                    args.profile_path, gpu_latency_label + [str(samples[j])] + ['dynamic_time'])
                gpu_time = (gpu_static_time + gpu_dynamic_time)/4
                S = samples[j]
                L = model_L[model + sub_model]

                if optimal_latency is None or cpu_time < optimal_latency[0]:
                    optimal_latency = [
                        cpu_time, [L, S, "", cpu_time, gpu_time, (entropy, entropy_std), (ece, ece_std), (error, error_std), path, model]]

                if optimal_error is None or error < optimal_error[0]:
                    optimal_error = [
                        error, [L, S, "", cpu_time, gpu_time, (entropy, entropy_std), (ece, ece_std), (error, error_std), path, model]]

                if optimal_entropy is None or entropy > optimal_entropy[0]:
                    optimal_entropy = [
                        entropy, [L, S, "", cpu_time, gpu_time, (entropy, entropy_std), (ece, ece_std), (error, error_std), path, model]]

                if optimal_ece is None or ece < optimal_ece[0]:
                    optimal_ece = [
                        ece, [L, S, "", cpu_time, gpu_time, (entropy, entropy_std), (ece, ece_std), (error, error_std), path, model]]
 
  logging.info("Optimal latency configuration: {}".format(optimal_latency))
  logging.info(latex_output(optimal_latency))
  print()
  print(latex_output(optimal_latency))
  print()
  logging.info("Optimal accuracy configuration: {}".format(optimal_error))
  logging.info(latex_output(optimal_error))
  print()
  print(latex_output(optimal_error))
  print()
  logging.info("Optimal uncertainty configuration: {}".format(optimal_entropy))
  logging.info(latex_output(optimal_entropy))
  print()
  print(latex_output(optimal_entropy))
  print()
  logging.info("Optimal ECE configuration: {}".format(optimal_ece))
  logging.info(latex_output(optimal_ece))
  print()
  print(latex_output(optimal_ece))
  print()


if __name__ == '__main__':
  main()
