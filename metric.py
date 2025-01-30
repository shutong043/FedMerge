import torch
from sklearn.metrics import classification_report,precision_score,recall_score,f1_score,accuracy_score
import numpy as np
import os
import pickle
class server_meters():
    def __init__(self,args):
        self.args=args
        self.overall_eval_loss=[]
        self.min_overall_eval_loss=999
        self.optimal_structure=None
        self.optimal_epoch=999
        self.structure_list=[]
        self.stop_growing_epoch=0
        self.stop_layer_only_epoch=0
        self.eval_accuracy=[]
        self.overall_eval_accuracy=[]
        self.eval_loss=[]
        self.overall_eval_loss=[]
        self.process_mask=None
        self.client_meter = []
    def update_optimal(self,structure,overall_eval_loss,epoch,overall_test_accuracy,test_accuracy_by_client):
        if self.min_overall_eval_loss>overall_eval_loss:
            self.optimal_structure=structure
            self.min_overall_eval_loss=overall_eval_loss
            self.optimal_epoch=epoch
            self.optimal_test_accuracy=overall_test_accuracy
            self.optimal_test_accuracy_by_client=test_accuracy_by_client
    def update_final(self, structure, epoch, overall_test_accuracy, test_accuracy_by_client):
            self.final_structure = structure
            self.final_epoch = epoch
            self.final_test_accuracy = overall_test_accuracy
            self.final_test_accuracy_by_client = test_accuracy_by_client
    def update_mask(self,process_mask):
        self.process_mask=process_mask
    def update_structure_list(self,structure):
        if len(self.structure_list)==0 or len(structure)>len(self.structure_list[-1]):
            for layer in list(structure.keys()):
                for group_id, group in structure[layer].items():
                    structure[layer][group_id][1]=None
            self.structure_list.append(structure)
    def update(self,epoch,eval_accuracy,overall_eval_accuracy,eval_loss,overall_eval_loss,):
        self.eval_accuracy.append((epoch,eval_accuracy))
        self.overall_eval_accuracy.append((epoch,overall_eval_accuracy))
        self.eval_loss.append((epoch,eval_loss))
        self.overall_eval_loss.append((epoch,overall_eval_loss))
    def update_clients(self,clients):
        for client in clients:
            self.client_meter.append(client.meters)


class meters():

    def __init__(self):
        self.ground_truth = None
        self.predict = None
        self.num_sample =['total number of data points in this client',]
        self.labels_id=['labels id in this client',]
        self.loss=['local training loss for every epoch',]
        self.batch_size=['evey local training batch size',]
        self.num_per_class=['number of samples for every class in this client',]
    def update_performance(self, ground_truth,predict,loss):
        ground_truth=ground_truth.cpu().numpy()
        m=torch.nn.Softmax(dim=1)
        predict=torch.argmax(m(predict),dim=1)
        predict=predict.data.cpu().numpy()
        loss=loss.detach().cpu().numpy().item()
        if self.ground_truth is None or self.predict is None:
            self.ground_truth=ground_truth
            self.predict=predict
        else:
            self.ground_truth=np.concatenate([self.ground_truth,ground_truth],axis=0,dtype=np.int32)
            self.predict=np.concatenate([self.predict,predict],axis=0,dtype=np.int32)
        self.loss.append(loss)
        self.batch_size.append(len(ground_truth))
    def produce_metric(self):
        self.new_loss=['local training loss for every epoch',]
        self.new_loss=[los*batch for los,batch in zip(self.loss[1:],self.batch_size[1:])]
        self.new_loss=np.sum(np.array(self.new_loss))/np.sum(np.array(self.batch_size[1:]))
        self.new_loss=np.float32(self.new_loss)
        del self.loss
        self.batch_size=self.batch_size[:2]
        self.labels_id.append(sorted(list(set(self.ground_truth))))
        self.num_sample.append(len(self.ground_truth))
        self.num_per_class.append([len(self.ground_truth[self.ground_truth==label]) for label in self.labels_id[1]])
        self.meter=classification_report(self.ground_truth,self.predict,zero_division=0)
        self.precision_score=np.float32(precision_score(self.ground_truth,self.predict,average='macro',zero_division=0))
        self.recall_score = np.float32(recall_score(self.ground_truth, self.predict,average='macro',zero_division=0))
        self.f1_score = np.float32(f1_score(self.ground_truth, self.predict,average=None,zero_division=0))
        self.accuracy_score = np.float32(accuracy_score(self.ground_truth, self.predict))
        del self.ground_truth
        del self.predict
def get_nonexistant_path(fpath):
    if not os.path.exists(fpath):
        return fpath
    root,extension=os.path.splitext(fpath)
    i=1
    new_fpath='{}_{}{}'.format(root,i,extension)
    while os.path.exists(new_fpath):
        i+=1
        new_fpath='{}_{}{}'.format(root,i,extension)
    return new_fpath


def save_meters(args,meter):
    dir=args.path
    if not os.path.exists(dir):
        os.makedirs(dir)
    dir = dir + '/' + 'her3' + '_' + str(args.mark)+'_' + str(args.model_name)+'_' + str(args.num_class_per_cluster) + '_' + str(
        args.use_class_partition) + '_' + str(args.alpha) + '_' + str(
        args.data_name)+ '_' + str(
        args.num_shared)+ '_' + str(
        args.learning_rate_W)+ '_' + str(
        args.learning_rate_W)+ '_' + str(
        args.learning_rate_Theta)
    fpath = get_nonexistant_path(dir+'.pkl')
    with open(fpath,'wb') as f:
        pickle.dump(meter,f)