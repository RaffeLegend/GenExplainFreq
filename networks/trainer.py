import functools
import torch
import torch.nn as nn
from networks.base_model import BaseModel, init_weights
import sys
from models import get_model

class Trainer(BaseModel):
    def name(self):
        return 'Trainer'

    def __init__(self, opt):
        super(Trainer, self).__init__(opt)
        self.opt = opt  
        self.model = get_model(opt.arch)
        torch.nn.init.normal_(self.model.classification_branch.fc.weight.data, 0.0, opt.init_gain)

        if opt.fix_backbone:
            params = []
            for name, p in self.model.classification_branch.named_parameters():
                if  name=="fc.weight" or name=="fc.bias": 
                    params.append(p) 
                else:
                    p.requires_grad = False
            for name, p in self.model.reconstruction_branch.named_parameters():
                params.append(p)
        else:
            print("Your backbone is not fixed. Are you sure you want to proceed? If this is a mistake, enable the --fix_backbone command during training and rerun")
            import time 
            time.sleep(3)
            params = self.model.parameters()

        if opt.optim == 'adam':
            self.optimizer = torch.optim.AdamW(params, lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=opt.weight_decay)
        elif opt.optim == 'sgd':
            self.optimizer = torch.optim.SGD(params, lr=opt.lr, momentum=0.0, weight_decay=opt.weight_decay)
        else:
            raise ValueError("optim should be [adam, sgd]")

        # self.loss_fn = nn.BCEWithLogitsLoss()
        # self.loss_fn = self.combined_loss()

        self.model.to(opt.gpu_ids[0])


    def adjust_learning_rate(self, min_lr=1e-6):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] /= 10.
            if param_group['lr'] < min_lr:
                return False
        return True


    def set_input(self, input):
        self.input = input[0].to(self.device)
        self.label = input[1].to(self.device).float()


    def forward(self):
        # self.output = self.model(self.input)
        # self.output = self.output.view(-1).unsqueeze(1)
        self.classification, self.reconstruction = self.model(self.input)
        self.classification = self.classification.view(-1).unsqueeze(1)


    def get_loss(self):
        return self.combined_loss()
        return self.loss_fn(self.output.squeeze(1), self.label)
    
    def combined_loss(self):
    # Classification loss (CrossEntropyLoss)
        classification_loss = nn.BCEWithLogitsLoss()(self.classification.squeeze(1), self.label)
    
        # Reconstruction loss (MSELoss)
        reconstruction_loss = nn.MSELoss()(self.reconstruction, self.input)
    
        # Total loss is the sum of both losses
        total_loss = classification_loss + reconstruction_loss
        return total_loss

    def optimize_parameters(self):
        self.forward()
        self.loss = self.combined_loss()
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()
