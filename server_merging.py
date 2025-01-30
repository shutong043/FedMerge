import copy
import numpy as np
from loaddata import generate_dataloader
from metric import server_meters
from sklearn.cluster import KMeans
from merging_utils import *
from loaddata import load_model
class Server():
    def __init__(self,args):
        self.time=[]
        self.args=args
        self.learning_rate_Theta=self.args.learning_rate_Theta
        self.learning_rate_W=self.args.learning_rate_W
        self.num_client=self.args.num_client
        self.num_shared=args.num_shared
        self.structure= {0:{}}
        self.eval_acc=[]
        self.test_acc = []
        self.mean_test_acc=0
        self.meters=server_meters(args)
        self.process_mask=[0,True]*self.args.num_client

    def init_model(self, data,clients):
        b=[client.num_data for client in clients]
        self.num_per_client=[a/sum(b) for a in b]
        cur_model=load_model(self.args, data['client' + str(0)]['train'])
        self.Theta = [copy.deepcopy(load_model(self.args, data['client' + str(0)]['train']).state_dict()) for i in range(self.num_shared)]
        self.W = np.zeros([self.args.num_client, self.num_shared])
        for i in range(self.args.num_client):
            merged_weght = weighted_average_state_dict(self.Theta, row_softmax(self.W[i]))
            cur_model.load_state_dict(copy.deepcopy(merged_weght))
            self.structure[0][i] = [np.array(i), copy.deepcopy(cur_model)]
        self.keys = self.structure[0][0][1].state_dict().keys()
    def model_distribute(self,epoch,clients):
        for i in range(self.args.num_client):
            merged_weght = weighted_average_state_dict(self.Theta, row_softmax(self.W[i]))
            self.structure[0][i][1].load_state_dict(copy.deepcopy(merged_weght))
        self.epoch=epoch
        for client in clients:
            client.load_model(self.structure)
    def train(self,clients):
        self.grads=[]
        self.before_update=[]
        for client in clients:
            client.train()
            self.grads.append(client.diff_structure[0].cpu().state_dict())
            self.before_update.append(client.before_update[0].cpu().state_dict())

    def eval(self, clients):
        acc = []
        loss=[]
        for client in clients:
            eval_loader = generate_dataloader(client.args.data_name, client.data['eval'], batch_size=32)
            meter = client.eval(eval_loader,'eval')
            acc.append(meter.accuracy_score)
            loss.append(meter.new_loss)
        self.eval_acc.append(np.array(acc))
        print('eval_accuracy:', acc)
        print('overall_eval_accuracy:', np.sum(np.array(acc)) / self.args.num_client)
        print('eval_loss:', loss)
        print('overall_eval_loss:', np.sum(np.array(loss)) / self.args.num_client)
        self.meters.update_optimal(copy.deepcopy(self.structure), np.sum(np.array(loss)) / self.args.num_client,
                                   copy.deepcopy(self.epoch), copy.deepcopy(self.mean_test_acc),
                                   copy.deepcopy(self.test_acc))
        self.meters.update_structure_list(copy.deepcopy(self.structure))
        self.meters.update(self.epoch, acc, np.sum(np.array(acc)) / self.args.num_client, loss,
                           np.sum(np.array(loss)) / self.args.num_client)
        return False
    def test(self,clients):
        self.test_acc = []
        for client in clients:
            test_loader = generate_dataloader(client.args.data_name, client.data['test'], batch_size=32)
            meter = client.eval(test_loader,'test')
            self.test_acc.append(meter.accuracy_score)
        print('test_accuracy:',  self.test_acc)
        self.mean_test_acc=np.sum(np.array( self.test_acc)) / self.args.num_client
        print('overall_test_accuracy:',  self.mean_test_acc)

    def update_Theta(self, grad):
        weighted_W = row_softmax(copy.deepcopy(self.W))
        weighted_W = (weighted_W.T * self.num_per_client).T
        weighted_W = weighted_W / np.sum(weighted_W, axis=0, keepdims=True)
        for j, global_model in enumerate(self.Theta):
            Theta_grad = weighted_average_state_dict(grad, weighted_W[:, j])
            # norm_of_grad,Theta_grad=get_norm_state_dict(Theta_grad)
            # print(norm_of_grad)
            self.Theta[j] = calculate_state_dict_minus(self.Theta[j],
                                                       weighted_average_state_dict([Theta_grad], [1]))
        return self.Theta

    def update_W(self, grad, avg_Theta_list):
        weighted_W = row_softmax(copy.deepcopy(self.W))
        W_grad = np.zeros([self.num_client, self.num_shared])
        for i in range(len(self.W)):
            for j in range(len(self.W[0])):
                W_grad[i, j] = weighted_W[i, j] * calculate_state_dict_inner_product(
                    get_top_layers_from_key(calculate_state_dict_minus(self.Theta[j], avg_Theta_list[i]), 'linear'),
                    get_top_layers_from_key(grad[i], 'linear'))
        W_grad=W_grad/np.sum(np.abs(W_grad))
        W_init = copy.deepcopy(self.W)
        n = 0
        while True:
            self.W = self.W - 0.01*np.array(self.num_per_client).reshape(-1, 1) * (W_grad)
            n = n + 1
            # if np.any(np.abs(self.W - W_init) > 0.1):
            if np.any(np.abs(self.W - W_init) > self.learning_rate_W) or n > 10000:
                # if np.any(np.abs(self.W - W_init) > 0.01) or np.any(self.W < 0.01) or n * self.learning_rate_W > 50:
                print("W learning rate:", n)
                break
        return self.W

    def aggregate(self):
        for i in range(1):
            self.update_W(self.grads, self.before_update)
            with open(self.args.path+"/" + str(self.args.data_name) + "_"+ str(self.args.use_class_partition) + "_" + str(
                    self.args.num_shared) + ".txt", 'a') as file:  # 'a' 表示追加模式
                file.write(f"Epoch: {self.epoch}\n")
                np.savetxt(file, row_softmax(self.W), fmt="%.3f")  # 将矩阵以固定精度写入文件
                file.write("\n")  # 添加一个换行分隔下一次的写入
            print(f"W = {row_softmax(self.W).round(3)}")
            self.update_Theta(self.grads)

