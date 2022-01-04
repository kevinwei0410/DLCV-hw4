import torch
from byol_pytorch import BYOL
from torchvision import models
import random
import numpy as np
import os
from torchvision import transforms
import torch.utils.data as Data
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.cuda
import torch.nn as nn
from torch.autograd import Function
from imageio import imread
from csv import reader
from PIL import Image
from matplotlib import pyplot as plt
from skimage.transform import resize

import warnings; warnings.simplefilter('ignore')



learning_rate = 1e-4
weight_decay = 0
lamda = 0.012

#MINIBATCH_SIZE = 16
MINIBATCH_SIZE = 2
EPOCH = 1000


gpu_paralell = False
gpu_id = 0

training_path = "./mini/train/"
val_path = "./mini/val/"
trainging_csv = "./mini/train.csv"
val_csv = "./mini/val.csv"

#reproducible
class Reproducible():
    def __init__(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


class MiniImageNet():
    def __init__(self, img_path, csv_path):
        lines = [x.strip() for x in open(csv_path, 'r').readlines()][1:]

        data = []
        label = []
        lb = -1

        self.wnids = []

        for l in lines:
            idx, name, wnid = l.split(',')
            path = os.path.join(img_path, name)
            if wnid not in self.wnids:
                self.wnids.append(wnid)
                lb += 1
            data.append(path)
            label.append(lb)

        self.data = data
        self.label = label

        self.transform = transforms.Compose([
        transforms.Resize(128),   
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.485, 0.456, 0.406],
                             std = [0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        path, label = self.data[i], self.label[i]
        image = self.transform(Image.open(path).convert('RGB'))
        #print(float(label))
        return image, label

class Training():
    def __init__(self):
        
        #data
        trainset = MiniImageNet(training_path, trainging_csv)
        self.train_data_loader = DataLoader(dataset=trainset, batch_size=MINIBATCH_SIZE, num_workers=0,  shuffle=True)
    
        valset = MiniImageNet(val_path, val_csv)
        self.val_data_loader = DataLoader(dataset=valset, batch_size=MINIBATCH_SIZE, num_workers=0,  shuffle=False)
        
        
        #model setup
        self.resnet = models.resnet50(pretrained=False).cuda()
        self.resnet.load_state_dict(torch.load("./pretrain_model_SL.pt"))
        


        # gpu device checking
        self.use_gpu = torch.cuda.is_available()
        print("Ready to use GPU : {}".format(self.use_gpu))
        
                
        #learner setting
        self.learner = BYOL(
            self.resnet,
            image_size = 128,
            hidden_layer = 'avgpool',
            projection_size = 256, 
            projection_hidden_size = 4096,
            moving_average_decay = 0.99
        )
        
        
        self.Training()
    
    def Training(self):
        
        
        #loss_function = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.learner.parameters(), lr=3e-4)
        
        self.training_loss_save_array = []
        self.val_loss_save_array = []
        self.training_acc_save_array = []
        self.val_acc_save_array = []
        
        print("start to train...")
        for epoch in range(1, EPOCH + 1):
            print("epoch {}".format(epoch))
            training_running_acc = 0.0
            validation_running_acc = 0.0
            training_running_loss = 0.0
            validation_running_loss = 0.0
            
            self.learner.train()
            
            for step, (batch_x, batch_y) in enumerate(self.train_data_loader):
                
                

                if self.use_gpu:
                    batch_x, batch_y = torch.tensor(batch_x.cuda()), torch.tensor(batch_y.cuda())
                
                optimizer.zero_grad()
                loss = self.learner(batch_x)
                pred = self.resnet(batch_x)
                
                #print(pred.shape)
        
                pred_class = np.argmax(pred.cpu().detach().numpy(), axis = 1)
                training_running_acc += np.sum(np.array(pred_class) == np.array(batch_y.cpu()))
                #print(np.array(pred_class))
                #print(np.array(batch_y.cpu()))
        
                loss.backward() 
                optimizer.step() 
                

                training_running_loss += loss.item()
                
                self.learner.update_moving_average()
                

            self.learner.eval()
                    
            '''if epoch % 1 == 0:
                    
                for step, (val_x, val_y) in enumerate(self.val_data_loader):
                    if self.use_gpu:
                        val_x, val_y = torch.tensor(val_x.cuda()),  torch.tensor(val_y.cuda())
                    
                    loss = self.learner(val_x)
    
                    pred = self.resnet(val_x)
    
                    validation_running_loss += loss.item()
                    
    
                    pred_class = np.argmax(pred.cpu().detach().numpy(), axis = 1)
                    validation_running_acc += np.sum(np.array(pred_class) == np.array(val_y.cpu()))'''
                    
            self.training_loss_save_array.append(training_running_loss / len(self.train_data_loader))
            self.training_acc_save_array.append(training_running_acc / (len(self.train_data_loader) * MINIBATCH_SIZE))
            
            self.val_loss_save_array.append(validation_running_loss / len(self.val_data_loader))
            self.val_acc_save_array.append(validation_running_acc  / (len(self.val_data_loader) * MINIBATCH_SIZE))
            
            print('training loss: {}, training acc: {}'.format(self.training_loss_save_array[epoch - 1], self.training_acc_save_array[epoch - 1]))
            print('validation loss: {}, val acc: {}'.format(self.val_loss_save_array[epoch - 1], self.val_acc_save_array[epoch - 1]))
            
            torch.save(self.resnet.state_dict(), './checkpoint/weight_{}_{}.pt'.format(epoch, str(self.training_loss_save_array[epoch - 1])))  
            
            
        
            #trainset = MiniImageNet(training_path, trainging_csv)
            

        
        


if __name__ == '__main__':
    #r = Reproducible(123)
    resnet = Training()
    