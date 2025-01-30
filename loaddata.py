import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from scipy.stats import dirichlet
import random
from tqdm import tqdm
from scipy.ndimage import uniform_filter1d
from collections import OrderedDict
from torch.utils.data import Dataset,DataLoader
import pickle
import resnet
from sklearn.model_selection import train_test_split
import cv2
import os
import pandas as pd
from PIL import Image
import torchvision.models as models
import torch.nn as nn

random.seed(11)
np.random.seed(11)
torch.manual_seed(11)


def separate_data(data, num_clients, num_classes, niid=False, balance=False, partition=None, class_per_client=2,
                  alpha=0.1):
    X = [[] for _ in range(num_clients)]
    y = [[] for _ in range(num_clients)]
    idxs_per_client = [[] for _ in range(num_clients)]  # 新增变量，保存每个数据的原始索引
    statistic = [[] for _ in range(num_clients)]

    dataset_content, dataset_label = data

    dataidx_map = {}

    if partition == "dir":
        min_size = 0
        K = num_classes
        N = len(dataset_label)

        while min_size < num_classes:
            idx_batch = [[] for _ in range(num_clients)]
            for k in range(K):
                idx_k = np.where(dataset_label == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
                proportions = np.array([p * (len(idx_j) < N / num_clients) for p, idx_j in zip(proportions, idx_batch)])
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
                min_size = min([len(idx_j) for idx_j in idx_batch])

        for j in range(num_clients):
            dataidx_map[j] = idx_batch[j]

    # assign data
    for client in range(num_clients):
        idxs = dataidx_map[client]
        X[client] = dataset_content[idxs]
        y[client] = dataset_label[idxs]
        idxs_per_client[client] = idxs  # 保存当前客户端的原始索引

        for i in np.unique(y[client]):
            statistic[client].append((int(i), int(sum(y[client] == i))))

    del data

    for client in range(num_clients):
        print(f"Client {client}\t Size of data: {len(X[client])}\t Labels: ", np.unique(y[client]))
        print(f"\t\t Samples of labels: ", [i for i in statistic[client]])
        print(f"\t\t Original indices: {idxs_per_client[client]}")  # 打印原始索引信息
        print("-" * 50)

    return X, y, statistic



def split_data_val(X, y, val_size, test_size):
    train_data, val_data, test_data = [], [], []
    num_samples = {'train': [], 'val': [], 'test': []}

    for i in range(len(y)):
        X_train, X_test, y_train, y_test = train_test_split(
            X[i], y[i], test_size=test_size, shuffle=True)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=val_size, shuffle=True)
        train_data.append({'x': X_train, 'y': y_train})
        num_samples['train'].append(len(y_train))
        val_data.append({'x': X_val, 'y': y_val})
        num_samples['val'].append(len(y_val))
        test_data.append({'x': X_test, 'y': y_test})
        num_samples['test'].append(len(y_test))

    print("Total number of samples:", sum(num_samples['train'] + num_samples['val'] + num_samples['test']))
    print("The number of train samples:", num_samples['train'])
    print("The number of validation samples:", num_samples['val'])
    print("The number of test samples:", num_samples['test'])
    print()
    del X, y

    return train_data, val_data, test_data


def memmap_clients(clients,args,client_by_class,class_by_client,gd_cluster,statistic,num_per_class):
    dir = args.path+'/'+args.data_name+'_' + str(args.num_class_per_cluster) + '_' + str(
        args.use_class_partition)+ '_' + str(args.alpha)+'/'
    if not os.path.exists(dir):
        os.makedirs(dir)
    if client_by_class is not None:
        with open(dir + 'client_by_class.pkl', 'wb') as file:
            pickle.dump(client_by_class, file)
    if class_by_client is not None:
        with open(dir + 'class_by_client.pkl', 'wb') as file:
            pickle.dump(class_by_client, file)
    if gd_cluster is not None:
        with open(dir + 'gd_cluster.pkl', 'wb') as file:
            pickle.dump(gd_cluster, file)
    if statistic is not None:
        with open(dir + 'statistic.pkl', 'wb') as file:
            pickle.dump(statistic, file)
    if num_per_class is not None:
        with open(dir + 'num_per_class.pkl', 'wb') as file:
            pickle.dump(num_per_class, file)
    for client_id,client in clients.items():
        if not os.path.exists(dir + client_id):
            os.makedirs(dir + client_id)
        for data_id,datas in client.items():
            np.save(dir + client_id + '/' + data_id + '_data.npy', datas[0].shape)
            fp = np.memmap(dir + client_id+'/'+data_id+'_data.dat', dtype=datas[0].dtype, mode='w+', shape=datas[0].shape)
            fp[:] = datas[0][:]
            fp.flush()
            np.save(dir + client_id + '/' + data_id + '_label.npy', datas[1].shape)
            fp = np.memmap(dir + client_id + '/' + data_id + '_label.dat', dtype=datas[1].dtype, mode='w+',
                           shape=datas[1].shape)
            fp[:] = datas[1][:]
            fp.flush()
def load_memmap_clients(args):
    dir = args.path + '/' + args.data_name + '_' + str(args.num_class_per_cluster) + '_' + str(
        args.use_class_partition) + '_' + str(args.alpha) + '/'
    client_by_class = None
    if os.path.exists(dir + 'client_by_class.pkl'):
        with open(dir + 'client_by_class.pkl', 'rb') as file:
            client_by_class = pickle.load(file)
    class_by_client=None
    if os.path.exists(dir + 'class_by_client.pkl'):
        with open(dir + 'class_by_client.pkl', 'rb') as file:
            class_by_client=pickle.load(file)
    statistic = None
    if os.path.exists(dir + 'statistic.pkl'):
        with open(dir + 'statistic.pkl', 'rb') as file:
            statistic = pickle.load(file)
    num_per_class = None
    if os.path.exists(dir + 'num_per_class.pkl'):
        with open(dir + 'num_per_class.pkl', 'rb') as file:
            num_per_class = pickle.load(file)
    gd_cluster = None
    if os.path.exists(dir + 'gd_cluster.pkl'):
        with open(dir + 'gd_cluster.pkl', 'rb') as file:
            gd_cluster = pickle.load(file)
    clients=OrderedDict()
    for i in range(args.num_client):
        clients['client' + str(i)]={}
        for data_id in ['train','eval','test']:
            clients['client'+str(i)][data_id]=[]
            shapes=tuple(np.load(dir + 'client' + str(i) + '/' + data_id + '_data.npy'))
            clients['client'+str(i)][data_id].append(np.memmap(dir + 'client' + str(i) + '/' + data_id + '_data.dat', dtype=np.uint8, mode='r',
                           shape=shapes))
            shapes = tuple(np.load(dir + 'client' + str(i) + '/' + data_id + '_label.npy'))
            clients['client' + str(i)][data_id].append(
                np.memmap(dir + 'client' + str(i) + '/' + data_id + '_label.dat', dtype=np.int64, mode='r',
                          shape=shapes))
    return clients, client_by_class, class_by_client, gd_cluster, statistic, num_per_class


def load_memmap_clients_domain(args):
    if args.data_name=='pacs':
        dir=args.path + '/'+'pacs'+'/'
    client_by_class = None
    if os.path.exists(dir + 'client_by_class.pkl'):
        with open(dir + 'client_by_class.pkl', 'rb') as file:
            client_by_class = pickle.load(file)
    class_by_client=None
    if os.path.exists(dir + 'class_by_client.pkl'):
        with open(dir + 'class_by_client.pkl', 'rb') as file:
            class_by_client=pickle.load(file)
    gd_cluster = None
    if os.path.exists(dir + 'gd_cluster.pkl'):
        with open(dir + 'gd_cluster.pkl', 'rb') as file:
            gd_cluster = pickle.load(file)
    statistic = None
    if os.path.exists(dir + 'statistic.pkl'):
        with open(dir + 'statistic.pkl', 'rb') as file:
            statistic = pickle.load(file)
    num_per_class = None
    if os.path.exists(dir + 'num_per_class.pkl'):
        with open(dir + 'num_per_class.pkl', 'rb') as file:
            num_per_class = pickle.load(file)
    clients=OrderedDict()
    for i in range(args.num_client):
        clients['client' + str(i)]={}
        inds=['train', 'test']
        for data_id in inds:
            clients['client'+str(i)][data_id]=[]
            shapes=tuple(np.load(dir + 'client' + str(i) + '/' + data_id + '_data.npy'))
            # sampletype=np.float32 if args.finetune else np.uint8
            sampletype=np.uint8
            clients['client'+str(i)][data_id].append(np.memmap(dir + 'client' + str(i) + '/' + data_id + '_data.dat', dtype=sampletype, mode='r',
                           shape=shapes))
            shapes = tuple(np.load(dir + 'client' + str(i) + '/' + data_id + '_label.npy'))
            clients['client' + str(i)][data_id].append(
                np.memmap(dir + 'client' + str(i) + '/' + data_id + '_label.dat', dtype=np.int64, mode='r',
                          shape=shapes))
    if gd_cluster is None:
        gd_cluster=statistic
    return clients, client_by_class,class_by_client,gd_cluster,statistic,num_per_class

def train2trainval(trainset,num_per_class,ratio=0.1):
    new_trainset=[[],[]]
    new_valset=[[],[]]
    for i in range(len(num_per_class)):
        cur_dataset=trainset[0][trainset[1]==i]
        cur_label=trainset[1][trainset[1]==i]
        cur_size=len(cur_label)
        permute=np.random.permutation(np.arange(cur_size))
        val_ind=permute[:int(ratio*cur_size)]
        train_ind=permute[int(ratio*cur_size):]
        if i==0:
            new_trainset[0]=cur_dataset[train_ind]
            new_trainset[1]=cur_label[train_ind]
            new_valset[0]=cur_dataset[val_ind]
            new_valset[1]=cur_label[val_ind]
        else:
            new_trainset[0]=np.concatenate([new_trainset[0],cur_dataset[train_ind]],axis=0)
            new_trainset[1] = np.concatenate([new_trainset[1], cur_label[train_ind]], axis=0)
            new_valset[0] = np.concatenate([new_valset[0], cur_dataset[val_ind]], axis=0)
            new_valset[1] = np.concatenate([new_valset[1], cur_label[val_ind]], axis=0)
    permute=np.random.permutation(len(new_trainset[0]))
    new_trainset[0]=new_trainset[0][permute]
    new_trainset[1]=new_trainset[1][permute]
    permute = np.random.permutation(len(new_valset[0]))
    new_valset[0] = new_valset[0][permute]
    new_valset[1] = new_valset[1][permute]
    return new_trainset,new_valset


def load_val_images(args):
    val_path, file_to_class = args
    data=np.empty((len(file_to_class),64,64,3),dtype=np.uint8)
    labels=[]
    for i,file_name in enumerate(os.listdir(val_path)):
        img = cv2.imread(os.path.join(val_path, file_name))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if len(img.shape) == 2:
            img = np.stack([img] * 3, axis=-1)
        label = file_to_class[file_name]
        data[i]=img
        labels.append(label)
    return data, labels
def load_images(args):
    path, wnid = args
    dir_path=os.listdir(os.path.join(path, wnid, 'images'))
    data=np.empty((len(dir_path),64,64,3),dtype=np.uint8)
    labels=[]
    for i,img_path in enumerate(dir_path):
        img = cv2.imread(os.path.join(path, wnid, 'images', img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert from OpenCV's BGR to RGB
        if len(img.shape) == 2:
            img = np.stack([img] * 3, axis=-1)
        data[i] = img
        labels.append(wnid)
    return data, labels

def load_cifar100(args):
    dir=args.path
    trainset = torchvision.datasets.CIFAR100(root=dir+'/cifar100/', train=True, download=True)
    testset = torchvision.datasets.CIFAR100(root=dir+'/cifar100/', train=False, download=True)
    num_class=len(trainset.classes)
    num_per_class=[]
    for i in range(num_class):
        num_per_class.append(trainset.targets.count(i)+\
                             testset.targets.count(i))
    trainset = [trainset.data, np.array(trainset.targets)]
    testset = [testset.data, np.array(testset.targets)]
    return trainset,testset,np.array(num_per_class)

class randompicker:
    def __init__(self,arr):
        self.arr=arr
        self.available=arr.copy()
    def pick(self):
        if not self.available:
            self.available=self.arr.copy()
        chosen=random.choice(self.available)
        self.available.remove(chosen)
        return chosen



def produce_cluster_by_class(num_per_class,num_class_per_cluster):
    num_class=len(num_per_class)
    if sum(num_class_per_cluster) < num_class or any(i>num_class for i in num_class_per_cluster):
        raise Exception('The number of classes in clusters is not appropriate.')

    picker=randompicker(np.arange(num_class).tolist())
    cluster_by_class=[]
    for i in num_class_per_cluster:
        classes = []
        for j in range(i):
            classes.append(picker.pick())
        cluster_by_class.append(classes)
    return cluster_by_class

def produce_client_by_class(ran,num_per_class,num_client,cluster_by_class,num_class_per_cluster):
    if ran:
        random_numbers=uniform_filter1d(np.random.rand(len(num_class_per_cluster)),size=3)
        num_client_per_cluster=np.round(random_numbers*num_client/np.sum(random_numbers)).astype(int)
        num_client_per_cluster[num_client_per_cluster<2]=2
        diff=np.sum(num_client_per_cluster)-num_client
        if diff>0:
            while diff!=0:
                ind=np.argmax(num_client_per_cluster)
                num_client_per_cluster[ind]-=1
                diff=diff-1
        elif diff<0:
            while diff!=0:
                 ind=np.argmin(num_client_per_cluster)
                 num_client_per_cluster[ind]+=1
                 diff=diff+1
        assert np.sum(num_client_per_cluster)==num_client
    client_by_class=[elem for i,elem in enumerate(cluster_by_class) for _ in range(num_client_per_cluster[i])]
    class_by_client=[]
    for i in range(len(num_per_class)):
        arr=[]
        for j,elem in enumerate(client_by_class):
            if i in elem:
                arr.append(j)
        class_by_client.append(arr)


    return client_by_class,class_by_client

def get_sample(clients,dirichlet_dis,is_train,dataset,classid,client_by_class,class_by_client):
    ind=np.random.permutation(len(dataset[1][dataset[1]==classid]))
    cur_data=dataset[0][dataset[1]==classid][ind]
    cur_label=dataset[1][dataset[1]==classid][ind]
    prob=dirichlet([dirichlet_dis]*len(class_by_client[classid]),seed=classid).rvs()
    class_num_by_client=np.round(len(ind)*prob)[0]
    diff=np.sum(class_num_by_client)-len(ind)
    count=0
    if diff>0:
        while diff!=0:
            class_num_by_client[np.argmax(class_num_by_client)]-=1
            count=count+1
            if count>20:
                break
            diff-=1
    elif diff<0:
        while diff!=0:
            class_num_by_client[np.argmin(class_num_by_client)]+=1
            count = count + 1
            if count > 20:
               break
            diff+=1
    assert np.sum(class_num_by_client)==len(ind)
    count=0
    while np.any(class_num_by_client<1):
        count+=1
        if count > 20:
            break
        class_num_by_client[np.argmax(class_num_by_client)] -= 1
        class_num_by_client[np.argmin(class_num_by_client)] += 1
    cur_cumsum=np.concatenate(([0],np.cumsum(class_num_by_client))).astype(int)
    for j in range(len(class_by_client[classid])):
        data_dict=[cur_data[cur_cumsum[j]:cur_cumsum[j+1]],cur_label[cur_cumsum[j]:cur_cumsum[j+1]]]
        if 'client'+str(class_by_client[classid][j]) not in clients:
            clients['client' + str(class_by_client[classid][j])] = {}
            clients['client' + str(class_by_client[classid][j])]['train']={}
            clients['client' + str(class_by_client[classid][j])]['eval'] = {}
            clients['client' + str(class_by_client[classid][j])]['test'] = {}
        if is_train==0:
            if clients['client' + str(class_by_client[classid][j])]['train']=={}:
                clients['client' + str(class_by_client[classid][j])]['train'] = data_dict
            else:
                clients['client'+str(class_by_client[classid][j])]['train'][0]=np.concatenate([clients['client'+str(class_by_client[classid][j])]['train'][0],data_dict[0]],axis=0)
                clients['client'+str(class_by_client[classid][j])]['train'][1]=np.concatenate([clients['client'+str(class_by_client[classid][j])]['train'][1],data_dict[1]])
        elif is_train==2:
            if clients['client' + str(class_by_client[classid][j])]['test'] == {}:
                clients['client' + str(class_by_client[classid][j])]['test'] = data_dict
            else:
                clients['client' + str(class_by_client[classid][j])]['test'][0] = np.concatenate(
                    [clients['client' + str(class_by_client[classid][j])]['test'][0], data_dict[0]], axis=0)
                clients['client' + str(class_by_client[classid][j])]['test'][1] = np.concatenate(
                    [clients['client' + str(class_by_client[classid][j])]['test'][1], data_dict[1]])
        elif is_train==1:
            if clients['client' + str(class_by_client[classid][j])]['eval'] == {}:
                clients['client' + str(class_by_client[classid][j])]['eval'] = data_dict
            else:
                clients['client' + str(class_by_client[classid][j])]['eval'][0] = np.concatenate(
                    [clients['client' + str(class_by_client[classid][j])]['eval'][0], data_dict[0]], axis=0)
                clients['client' + str(class_by_client[classid][j])]['eval'][1] = np.concatenate(
                    [clients['client' + str(class_by_client[classid][j])]['eval'][1], data_dict[1]])

    return clients

def produce_client_by_data(trainset,valset,testset,num_per_class,client_by_class,class_by_client,dirichlet_dis):
    num_class=len(num_per_class)
    clients={}
    for i in range(num_class):
        clients=get_sample(clients,dirichlet_dis,0,trainset,i,client_by_class,class_by_client)
        clients = get_sample(clients, dirichlet_dis, 1, valset, i, client_by_class, class_by_client)
        clients=get_sample(clients,dirichlet_dis,2,testset,i,client_by_class,class_by_client)
    clients=OrderedDict(clients)
    for i in range(len(clients)):
        value=clients.pop('client'+str(i))
        clients['client'+str(i)]=value
    return clients

def generate_dataloader(name,data,batch_size,transform=None):

    class Mydataset(Dataset):
        def __int__(self, data, label, transform=None):
            self.data = data
            self.label = label
            self.transform = transform

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            sample_data = self.data[idx]
            if self.transform:
                sample_data = self.transform(sample_data)

            return sample_data, self.label[idx]


    if transform is None:
        if name=='cifar100':
            transform = transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.5071,0.4867,0.4408), (0.2675,0.2565,0.2761))])
        if name == 'pacs':
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])
    dataset=Mydataset()
    dataset.data=data[0]
    dataset.label=data[1]
    dataset.transform=transform
    dataloader=DataLoader(dataset,batch_size=batch_size,shuffle=True,drop_last=False)
    return dataloader


def produce_data(args):
    trainset, testset, num_per_class = load_cifar100(args)
    if not args.use_class_partition:
        X, y, statistic = separate_data(
            (np.concatenate([trainset[0], testset[0]], axis=0), np.concatenate([trainset[1], testset[1]], axis=0)),
            args.num_client, len(num_per_class),
            True, False, 'dir',None, args.alpha)
        train_data, eval_data, test_data = split_data_val(X, y, val_size=0.1, test_size=1 / 6)
        clients = {}
        for i in range(args.num_client):
            clients['client' + str(i)] = {}
            clients['client' + str(i)]['train'] = []
            clients['client' + str(i)]['eval'] = []
            clients['client' + str(i)]['test'] = []
            clients['client' + str(i)]['train'].append(train_data[i]['x'])
            clients['client' + str(i)]['train'].append(train_data[i]['y'])
            clients['client' + str(i)]['eval'].append(eval_data[i]['x'])
            clients['client' + str(i)]['eval'].append(eval_data[i]['y'])
            clients['client' + str(i)]['test'].append(test_data[i]['x'])
            clients['client' + str(i)]['test'].append(test_data[i]['y'])
        del X, y, train_data, eval_data, test_data
        memmap_clients(clients, args, None, None, None, statistic, num_per_class)
        clients, _, _, _, statistic, num_per_class = load_memmap_clients(args)
        return clients

    trainset, evalset = train2trainval(trainset, num_per_class)
    gd_cluster = None
    cluster_by_class = produce_cluster_by_class(num_per_class, args.num_class_per_cluster)
    client_by_class, class_by_client = produce_client_by_class(True, num_per_class, args.num_client, cluster_by_class,
                                                               args.num_class_per_cluster)
    clients = produce_client_by_data(trainset, evalset, testset, num_per_class, client_by_class, class_by_client,
                                     10)
    memmap_clients(clients, args, client_by_class, class_by_client, gd_cluster, None, None)
    clients, client_by_class, class_by_client, gd_cluster, _, _ = load_memmap_clients(args)
    return clients, client_by_class, class_by_client, gd_cluster

def load_data(args):
    if args.use_class_partition:
        clients, client_by_class, class_by_client, gd_cluster, _, _ = load_memmap_clients(args)
        return clients, client_by_class, class_by_client, gd_cluster
    else:
        clients, _, _, gd_cluster, statistic, num_per_class = load_memmap_clients(args)
        return clients, statistic, num_per_class, None

def resize_image(image_array):
    img = Image.fromarray(image_array)  # 将 NumPy 数组转换为 PIL 图像
    img_resized = img.resize((64, 64), Image.Resampling.LANCZOS)  # 调整大小为 64x64
    return np.array(img_resized, dtype='uint8')  # 转换回 NumPy 数组

def load_data_domain(args):
    clients, client_by_class, class_by_client, gd_cluster, _, _ = load_memmap_clients_domain(args)
    for client, datasets in clients.items():
        for dataset_type in ['train', 'test']:
            memmap_data=clients[client][dataset_type][0]
            resized_data = np.empty((memmap_data.shape[0], 64, 64, 3), dtype='uint8')
            for i in range(memmap_data.shape[0]):
                resized_data[i] = resize_image(memmap_data[i])
            clients[client][dataset_type][0]=resized_data
    return clients, client_by_class, class_by_client, gd_cluster


def load_model_domain(args):
    if 'resnet' in args.model_name:
        model = resnet.ResNet9(num_class=len(args.class_by_client))
    if "9" in args.model_name:
        model = resnet.ResNet9(num_class=len(args.class_by_client))
    if "18" in args.model_name:
        model = resnet.ResNet18(num_class=len(args.class_by_client))
    if "34" in args.model_name:
        model = resnet.ResNet34(num_class=len(args.class_by_client))
    if "50" in args.model_name:
        model = resnet.ResNet50(num_class=len(args.class_by_client))
    if "101" in args.model_name:
        model = resnet.ResNet101(num_class=len(args.class_by_client))
    if "152" in args.model_name:
        model = resnet.ResNet152(num_class=len(args.class_by_client))
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, len(args.class_by_client))
    return model

def load_model(args,sample_data):
    if 'resnet' in args.model_name:
        model = resnet.ResNet9(num_class=len(args.class_by_client))
    if "9" in args.model_name:
        model = resnet.ResNet9(num_class=len(args.class_by_client))
    if "18" in args.model_name:
        model=resnet.ResNet18(num_class=len(args.class_by_client))
    if "34" in args.model_name:
        model=resnet.ResNet34(num_class=len(args.class_by_client))
    if "50" in args.model_name:
        model=resnet.ResNet50(num_class=len(args.class_by_client))
    if "101" in args.model_name:
        model=resnet.ResNet101(num_class=len(args.class_by_client))
    if "152" in args.model_name:
        model=resnet.ResNet152(num_class=len(args.class_by_client))
    train_loader = generate_dataloader(args.data_name, sample_data, args.batch_size)
    model.eval()
    for (input, target) in train_loader:
        output = model(input)
        break
    return model
