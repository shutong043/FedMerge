from loaddata import *
import argparse
import ast
def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
parser = argparse.ArgumentParser(description='')
parser.add_argument('--path', default='./dump_items', type=str, help='the path for generated non-IID settings and experiment results,please put original cifar100 and tinyimagenet dataset in this path.')
parser.add_argument('--mark', default='FeMAM', type=str, help='Rename this variable to distinguish between different experiment outcomes')
parser.add_argument('--data-name', default='cifar100', type=str, help='choose the data between tinyimagenet and cifar100')
parser.add_argument('--num-client', default=50, type=int, help='choose the number of clients')
parser.add_argument('--use-class-partition', default=True, type=str2bool, help='whether to partition data by class')
parser.add_argument('--num-class-per-cluster', default='[30, 30, 30, 30, 30]', type=str, help='choose the number of class per cluster in dataset,for example,[3,3,3,3,3] means five clusters, three class per cluster')
parser.add_argument('--alpha', default=0.1, type=float, help='choose the paramter of dirichlet distribution for data')
parser.add_argument('--device', default=0, type=int, help='gpu device number')
if __name__=='__main__':
    args = parser.parse_args()
    args.num_class_per_cluster=ast.literal_eval(args.num_class_per_cluster)
    produce_data(args)
