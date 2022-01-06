import sys
import argparse
import numpy as np
import logging
import copy 
from mpl_toolkits.mplot3d import Axes3D


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

  samples = [3, 4, 5, 6, 7, 8, 9, 10, 20, 50, 100]
  model_options = ["ll", "one_third", 'half', 'two_third', 'all']

  fig = plt.figure(figsize=(8,4))
  ax = Axes3D(fig)
  ax.set_box_aspect((2, 3, 1))
  error_l, entropy_l, ece_l, time_l = [], [], [], []
  for i, paths in enumerate([args.ll_paths, args.one_third_paths, args.half_paths, args.two_third_paths, args.all_paths]):
    assert len(paths) == 11 or len(paths) == 0
    for j, path in enumerate(paths):
        error, error_std = load_data_from_path(path, ['error', 'test'])
        if error!=0.0:
            print(path)
            entropy, entropy_std = load_data_from_path(path, ['entropy', 'random'])
            ece, ece_std = load_data_from_path(path, ['ece', 'test'])
            ece*=100
            ece_std*=100
            sub_model = model_options[i]

            latency_label = [model_options[i]] + [samples[j]]
            time = load_data_from_path(args.profile_path, latency_label)
            
            error_l.append(error)
            entropy_l.append(entropy)
            ece_l.append(ece)
            time_l.append(time)

  img = ax.scatter(100.-np.array(error_l), time_l, entropy_l, c=ece_l,
                    alpha=0.7, s=100)

  logging.info(error_l)
  logging.info(time_l)
  logging.info(entropy_l)
  logging.info(ece_l)
  cb = fig.colorbar(img, ax=[ax], location='top', fraction=0.046, pad=-0.04)
  cb.ax.set_xlabel('ECE [%]', fontweight='bold')
  cb.ax.xaxis.set_label_coords(1.1, 0.2)
  ax.view_init(5)
  ax.set_xlabel('Accuracy [%]', fontweight ='bold', labelpad=10) 
  ax.set_ylabel('Latency [ms]', fontweight ='bold', labelpad=10)
  ax.set_zlabel('aPE [nats]', fontweight ='bold')
  fig.savefig(args.save+"/{}.pdf".format(args.dataset), bbox_inches='tight')


if __name__ == '__main__':
  main()
