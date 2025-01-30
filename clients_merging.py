import copy
from loaddata import generate_dataloader
import torch
import torch.nn as nn
from collections import OrderedDict
from metric import meters
class client():
    def __init__(self,i,args,data,model):
        self.args=args
        self.id=i
        self.data=data
        self.num_data=len(data['train'][0])
        self.keys = model.state_dict().keys()
        self.structure = {}
        self.meters= {'train':[],'test':[],'eval':[]}
        self.eval_acc=[999]
        self.eval_loss=[999]
        self.train_acc=[999]
        self.process_mask=[0,True]
    def load_model(self,global_structure):
        layers = [list(global_structure.keys())[-1]]
        for layer in layers:
            for group_id,group in global_structure[layer].items():
                if self.id in group[0]:
                    self.structure[layer]=copy.deepcopy(group[1])
    def train(self):
        train_loader = generate_dataloader(self.args.data_name, self.data['train'], batch_size=self.args.batch_size)

        self.before_update = copy.deepcopy(self.structure)

        meter_train = self.train_model(train_loader)
        self.train_acc.append(meter_train.accuracy_score)
        self.compute_diff_structure()
        return meter_train

    def eval(self,loader,mode):
        self.meters[mode].append(meters())
        for layer, model in self.structure.items():
            model.cuda(self.args.device).eval()
        for i, (input, target) in enumerate(loader):
            target = target.cuda(self.args.device)
            input = input.cuda(self.args.device)
            output = 0
            for layer, model in self.structure.items():
                output += model(input)
            loss = nn.CrossEntropyLoss().cuda(self.args.device)(output/len(self.structure.items()), target)
            self.meters[mode][-1].update_performance(target, output, loss)
            del input, target
        self.meters[mode][-1].produce_metric()
        if mode=='eval':
            self.eval_acc.append(self.meters[mode][-1].accuracy_score)
            self.eval_loss.append(self.meters[mode][-1].new_loss)
        for layer, model in self.structure.items():
            model.cpu()
        return self.meters[mode][-1]

    def compute_diff_structure(self):
        layers = [list(self.structure.keys())[-1]]
        self.diff_structure=copy.deepcopy(self.structure)
        for layer in layers:
            diff_param = OrderedDict()
            param_new = self.structure[layer].state_dict()
            param_old = self.before_update[layer].state_dict()
            for key in param_new.keys():
                diff_param[key] = param_old[key]-param_new[key]
            self.diff_structure[layer].load_state_dict(diff_param)

    def train_model(self, train_loader):
        self.meters['train'].append(meters())
        layers = [list(self.structure.keys())[-1]]
        for layer in layers:
            for cur_layer, model in self.structure.items():
                if cur_layer == layer:
                    model.cuda(self.args.device).train()
                    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
                else:
                    model.cuda(self.args.device).eval()
        for epoch in range(self.args.local_epoch):
            for i, (input, target) in enumerate(train_loader):
                target = target.cuda(self.args.device)
                input = input.cuda(self.args.device)
                output = 0
                for cur_layer, model in self.structure.items():
                    output += model(input)
                loss = nn.CrossEntropyLoss().cuda(self.args.device)(output/len(self.structure.items()), target)
                lambda_reg = 0.01
                l2_diff = 0.0
                for (gp_id1, model1), (gp_id2, model2) in zip(self.structure.items(),
                                                              self.before_update.items()):

                    model2.cuda(self.args.device)
                    for (name1, p1), (name2, p2) in zip(model1.named_parameters(), model2.named_parameters()):
                        if 'running_mean' not in name1 and 'running_var' not in name1:
                            l2_diff += torch.norm(p1 - p2.detach(), p=2) ** 2
                loss += lambda_reg * l2_diff
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if epoch == self.args.local_epoch - 1:
                    self.meters['train'][-1].update_performance(target, output, loss)
                del input, target
            if epoch == self.args.local_epoch - 1:
                self.meters['train'][-1].produce_metric()
        for (gp_id1, model1), (gp_id2, model2) in zip(self.structure.items(), self.before_update.items()):
            model1.cpu()
            model2.cpu()
        return self.meters['train'][-1]

