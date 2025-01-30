from loaddata import *
from server_merging import Server
import time
from metric import save_meters
from clients_merging import client
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
parser.add_argument('--mark', default='FedMerge', type=str, help='Rename this variable to distinguish between different experiment outcomes')
parser.add_argument('--data-name', default='cifar100', type=str, help='choose the data between tinyimagenet and cifar100')
parser.add_argument('--local-epoch', default=2, type=int, help='number of local epoch, default is 2')
parser.add_argument('--model-name', default='resnet9', type=str, help='use resnet as the model')
parser.add_argument('--num-client', default=50, type=int, help='choose the number of clients')
parser.add_argument('--use-class-partition', default=True, type=str2bool, help='whether to partition data by class')
parser.add_argument('--num-class-per-cluster', default='[30, 30, 30, 30, 30]', type=str, help='choose the number of class per cluster in dataset,for example,[3,3,3,3,3] means five clusters, three class per cluster')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--global-epoch', default=500, type=int, help='number of global epoch')
parser.add_argument('--batch-size', default=32, type=int, help='number of batch size')
parser.add_argument('--use-diff', default=True, type=str2bool, help='whether use gradient as clustering samples')
parser.add_argument('--alpha', default=0.1, type=float, help='choose the paramter of dirichlet distribution for data')
parser.add_argument('--device', default=0, type=int, help='gpu device number')
parser.add_argument('--num-shared', default=5, type=int, help='')
parser.add_argument('--learning_rate_Theta', default=1, type=int, help='')
parser.add_argument('--learning_rate_W', default=0.01, type=int, help='')
if __name__=='__main__':
    args = parser.parse_args()
    data, client_by_class, class_by_client, gd_cluster = load_data(args)
    args.num_class_per_cluster=ast.literal_eval(args.num_class_per_cluster)
    args.client_by_class=client_by_class
    args.class_by_client=class_by_client
    args.gd_cluster=gd_cluster

    model=load_model(args,data['client' + str(0)]['train'])
    clients=[]
    for i in range(args.num_client):
        clients.append(client(i,args,data['client' + str(i)],model))
    server=Server(args)
    server.init_model(data,clients)
    server.time.append(time.time())
    for epoch in range(args.global_epoch):
        print('epoch:', epoch)

        server.model_distribute(epoch,clients)
        server.eval(clients)
        server.train(clients)
        server.aggregate()
        server.model_distribute(epoch,clients)
        server.test(clients)
    server.meters.update_clients(clients)
    save_meters(args,server.meters)
    server.time.append(time.time())



