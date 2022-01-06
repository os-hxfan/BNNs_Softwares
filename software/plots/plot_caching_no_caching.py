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
parser.add_argument('--profile_path', type=str,
                    default='', help='experiment name')
parser.add_argument('--label', type=str, default='',
                    help='default experiment category ')
parser.add_argument('--no_caching', action='store_true',
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
    return "& \\{{{}, {}\\}} & {} & {} & {}  \\\\".format(
        l[1][0], l[1][1], l[1][2], round_sig(l[1][3]), round_sig(l[1][4]))


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

  dataset_models = {'mnist': 'lenet', 
                    'svhn': 'vgg',
                    'cifar': 'resnet'}
  model_L = {'lenet_ll': 1, 'lenet_two_third': "$2/3 \\times N$", 'lenet_all': "$N$",
             'vgg_ll': 1, 'vgg_one_third': "$1/3 \\times N$", 'vgg_half': "$1/2 \\times N$", 'vgg_two_third': "$2/3 \\times N$", 'vgg_all': "$N$",
             'resnet_ll': 1, 'resnet_one_third': "$1/3 \\times N$", 'resnet_half': "$1/2 \\times N$",  'resnet_two_third': "$2/3 \\times N$", 'resnet_all': "$N$"}
  profie_options = {'cpu_quant': ['quant', 'latency', 'cpu'],
                    'gpu_not_quant': ['not_quant', 'latency', 'gpu']}
  
  model = dataset_models[args.dataset]
  
  configs = [["_ll", 100], ["_two_third", 50]]
  for config in configs:
        sub_model = config[0]
        L = model_L[model+sub_model]
        S = config[1]

        cpu_latency_label = [model+sub_model] + profie_options['cpu_quant']
        gpu_latency_label = [model+sub_model] + profie_options['gpu_not_quant']

        if not args.no_caching:
            cpu_static_time = load_data_from_path(
                args.profile_path, cpu_latency_label + [str(S)] + ['static_time'])
            cpu_dynamic_time = load_data_from_path(
                args.profile_path, cpu_latency_label + [str(S)] + ['dynamic_time'])
            cpu_time = cpu_static_time + cpu_dynamic_time

            gpu_static_time = load_data_from_path(
                args.profile_path, gpu_latency_label + [str(S)] + ['static_time'])
            gpu_dynamic_time = load_data_from_path(
                args.profile_path, gpu_latency_label + [str(S)] + ['dynamic_time'])
            gpu_time = (gpu_static_time + gpu_dynamic_time)/4
        else:
            cpu_time = load_data_from_path(
                args.profile_path, cpu_latency_label + [str(S)] + ['total_eval_time'])
            gpu_time = load_data_from_path(
                args.profile_path, gpu_latency_label + [str(S)] + ['total_eval_time'])
            gpu_time/=4

        logging.info("Configuration: {}, {}".format(model+sub_model, S))
        logging.info(latex_output(
            [None,[L, S, "", cpu_time, gpu_time, model+sub_model]]))

if __name__ == '__main__':
  main()
