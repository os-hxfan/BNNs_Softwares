
import sys
import argparse
import numpy as np
import logging

sys.path.append("../")
sys.path.append("../../")
sys.path.append("../../../")
sys.path.append("../../../../")
sys.path.append("../../../../../")
sys.path.append("../../../../../../")


from software.plots import PLT as plt
import software.utils as utils

parser = argparse.ArgumentParser("compare_ood_results")

parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--pointwise_paths_a', nargs='+',
                    default=[], help='experiment name')
parser.add_argument('--pointwise_paths_b', nargs='+',
                    default=[], help='experiment name')
parser.add_argument('--all_paths_a', nargs='+',
                    default=[], help='experiment name')
parser.add_argument('--all_paths_b', nargs='+',
                    default=[], help='experiment name')
parser.add_argument('--two_third_paths_a', nargs='+',
                    default=[], help='experiment name')
parser.add_argument('--two_third_paths_b', nargs='+',
                    default=[], help='experiment name')
parser.add_argument('--half_paths_a', nargs='+',
                    default=[], help='experiment name')
parser.add_argument('--half_paths_b', nargs='+',
                    default=[], help='experiment name')
parser.add_argument('--one_third_paths_a', nargs='+',
                    default=[], help='experiment name')
parser.add_argument('--one_third_paths_b', nargs='+',
                    default=[], help='experiment name')
parser.add_argument('--ll_paths_a', nargs='+',
                    default=[], help='experiment name')
parser.add_argument('--ll_paths_b', nargs='+',
                    default=[], help='experiment name')

parser.add_argument('--label', type=str, default='',
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


def load_data_from_paths(paths, label):
  data = []
  for i in range(11):
    if i < len(paths) and paths[i] != "none":
      result = utils.load_pickle(paths[i]+"/results.pickle")
      data.append(get_dict_value(result, label))
    else:
      data.append((0.0, 0.0))
  return data


def plot_bar_chart(all_data, two_third_data, half_data, one_third_data, ll_data, pointwise_data, yaxis, args, dashed=False):
    colors = ['k', 'r', 'g', 'b', 'c', 'm']

    def _plot_bar(data, indeces, color, label, dash=False):
      std = ([0 for i in range(len(data))])
      mean = ([0 for i in range(len(data))])

      for i, d in enumerate(data):
        if isinstance(d, tuple):
          d = list(d)
          mean[i] = d[0]
          if yaxis == "error" or yaxis == "Accuracy [%]":
             mean[i] = 100 - mean[i]
          elif yaxis == "ece" or yaxis == "ECE [%]":
             mean[i] *= 100
             d[1] *= 100
             d[0]*=100
          std[i] = (d[1], d[1])
          if yaxis == "error" or yaxis == "Error [%]" or yaxis == "ECE [%]" or yaxis == "ece" or yaxis == "entropy" or yaxis == "aPE" or yaxis == "nll" or yaxis == "$aNLL_{cls}$":
            std[i] = (min(d[1], d[0]), d[1])
            pass
        else:
          mean[i] = d[0]

      std = np.array(std).T
      print(indeces, mean, std)

      for i, index in enumerate(indeces):
        yerr = np.array([std[0][i], std[1][i]]).reshape(2, 1)
        if dash is False:
          plt.bar([index], [mean[i]], color=color, yerr=yerr, label=label,
                  ecolor='k', capsize=5, alpha=0.25+i*(0.75/10))
        else:
          plt.bar([index], [mean[i]], color=color, yerr=yerr, label=label,
                  ecolor='k', capsize=5, alpha=0.25+i*(0.75/10), hatch='//')
    offset = 1 if dashed else 0
    _plot_bar(pointwise_data, np.array(
        [0])+offset, colors[0], "Pointwise", dashed)
    _plot_bar(ll_data, np.array(
        [3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23])+offset, colors[1], "LL", dashed)

    if 'mnist' not in args.label:
      _plot_bar(one_third_data, np.array(
          [26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46])+offset, colors[2], "1/3", dashed)
      _plot_bar(half_data, np.array(
          [49, 51, 53, 55, 57, 59, 61, 63, 65, 67, 69])+offset, colors[3], "1/2", dashed)
      _plot_bar(two_third_data, np.array(
          [72, 74, 76, 78, 80, 82, 84, 86, 88, 90, 92])+offset, colors[4], "2/3", dashed)
      _plot_bar(all_data, np.array(
          [95, 97, 99, 101, 103, 105, 107, 109, 111, 113, 115])+offset, colors[5], "ALL", dashed)
    else:
      _plot_bar(two_third_data, np.array(
          [26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46])+offset, colors[-2], "2/3", dashed)
      _plot_bar(all_data, np.array(
          [49, 51, 53, 55, 57, 59, 61, 63, 65, 67, 69])+offset, colors[-1], "ALL", dashed)


def main():
  args = parser.parse_args()
  args, _ = utils.parse_args(args, args.label)
  logging.info('## Loading of result pickles for the experiment ##')

  units = {'error': 'Accuracy [%]', 'ece': 'ECE [%]',
           'nll': '$aNLL_{cls}$', 'entropy': 'aPE [nats]'}
  labels = ['error', 'nll', 'ece', 'entropy']
  datasets = ['test', 'random']
  for label in labels:
    for dataset in datasets:
        pointwise_data_a = load_data_from_paths(
            args.pointwise_paths_a, [label, dataset])
        all_data_a = load_data_from_paths(args.all_paths_a, [label, dataset])
        two_third_data_a = load_data_from_paths(
            args.two_third_paths_a, [label, dataset])
        half_data_a = load_data_from_paths(
            args.half_paths_a, [label, dataset])
        one_third_data_a = load_data_from_paths(
            args.one_third_paths_a, [label, dataset])
        ll_data_a = load_data_from_paths(args.ll_paths_a, [label, dataset])

        pointwise_data_b = load_data_from_paths(
            args.pointwise_paths_b, [label, dataset])
        all_data_b = load_data_from_paths(args.all_paths_b, [label, dataset])
        two_third_data_b = load_data_from_paths(
            args.two_third_paths_b, [label, dataset])
        half_data_b = load_data_from_paths(
            args.half_paths_b, [label, dataset])
        one_third_data_b = load_data_from_paths(
            args.one_third_paths_b, [label, dataset])
        ll_data_b = load_data_from_paths(args.ll_paths_b, [label, dataset])

        plt.figure(figsize=(8, 3), facecolor='w')
        plot_bar_chart(all_data_a, two_third_data_a, half_data_a, one_third_data_a, ll_data_a, pointwise_data_a,
                       units[label], args, False)
        plot_bar_chart(all_data_b, two_third_data_b, half_data_b, one_third_data_b, ll_data_b, pointwise_data_b,
                       units[label], args, True)

        plt.tick_params(axis='x', which='both', length=0)
        if 'mnist' not in args.label:
          plt.xticks(ticks=[0, 13, 36, 59, 82, 103], labels=[
              'S', 'LL', '1/3', '1/2', '2/3', 'ALL'], fontsize=14)
          plt.xlim(-0.5, 116.5)
        else:
          plt.xticks(ticks=[0, 13, 36, 59], labels=[
              'S', 'LL', '2/3', 'ALL'], fontsize=14)
          plt.xlim(-0.5, 70.5)
        plt.ylabel(units[label], fontsize=14)
        plt.tight_layout()
        plt.grid()
        path = utils.check_path(
            args.save+'/{}_{}.pdf'.format(dataset, label))
        plt.savefig(path)


if __name__ == '__main__':
  main()
