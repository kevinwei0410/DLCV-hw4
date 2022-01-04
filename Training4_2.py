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
MINIBATCH_SIZE = 64
EPOCH = 200


gpu_paralell = False
gpu_id = 0

training_path = "./office/train/"
val_path = "./office/val/"
trainging_csv = "./office/train.csv"
val_csv = "./office/val.csv"



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

#resnet = models.resnet50(pretrained=False)
#resnet.fc = nn.Linear(2048, 65)

#print(resnet)

'''#print(resnet)        
        #model setup
        self.resnet = models.resnet50(pretrained=False)
        #self.resnet.load_state_dict(torch.load("./pretrain_model_SL.pt"))
        self.resnet.load_state_dict(torch.load("./checkpoint/weight_106_0.291213141468664.pt"))
        # ./checkpoint/weight_106_0.291213141468664.pt
        
        self.resnet.fc = nn.Linear(2048, 65)
        
        self.resnet = self.resnet.cuda()
        
        #print(self.resnet)'''

class Training():
    def __init__(self):
        
        
        #data
        p = self.preprocessing()
        self.resnet = models.resnet50(pretrained=False)
        
        
       
       
        
        #model setup
        
        #self.resnet.load_state_dict(torch.load("./pretrain_model_SL.pt"))
        self.resnet.load_state_dict(torch.load("./checkpoint/weight_106_0.291213141468664.pt"))
        # ./checkpoint/weight_106_0.291213141468664.pt
        
        self.resnet.fc = nn.Linear(2048, 65)
        #print(self.resnet)

        self.resnet = self.resnet.cuda()
        # gpu device checking
        self.use_gpu = torch.cuda.is_available()
        print("Ready to use GPU : {}".format(self.use_gpu))
        
                
        #learner setting
        '''self.learner = BYOL(
            self.resnet,
            image_size = 128,
            hidden_layer = 'avgpool',
            projection_size = 256, 
            projection_hidden_size = 4096,
            moving_average_decay = 0.99
        )'''
        
        
        self.Training()
    
    def Training(self):
        
        
        loss_function = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(self.resnet.parameters(), lr=learning_rate, eps=1e-08, weight_decay=0.001)

        for param in self.resnet.parameters():
            param.requires_grad = False
        self.resnet.fc.weight.requires_grad = True
        self.resnet.fc.bias.requires_grad = True
        
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
            
            self.resnet.train()
            
            for step, (batch_x, batch_y) in enumerate(self.train_data_loader):
                
                

                if self.use_gpu:
                    loss_function = loss_function.cuda()
                    batch_x, batch_y = torch.tensor(batch_x.cuda()), torch.tensor(batch_y.cuda())
                
                optimizer.zero_grad()
               # loss = self.learner(batch_x)
                pred = self.resnet(batch_x)
                loss = loss_function(pred, batch_y)
                
                pred_class = np.argmax(pred.cpu().detach().numpy(), axis = 1)
                training_running_acc += np.sum(np.array(pred_class) == np.array(batch_y.cpu()))
        
                loss.backward() 
                optimizer.step() 
                

                training_running_loss += loss.item()
                
                #self.learner.update_moving_average()
                

            self.resnet.eval()
                    
            if epoch % 1 == 0:
                    
                for step, (val_x, val_y) in enumerate(self.val_data_loader):
                    if self.use_gpu:
                        val_x, val_y = torch.tensor(val_x.cuda()),  torch.tensor(val_y.cuda())
                    
                    #loss = self.learner(val_x)
                    pred = self.resnet(val_x)
                    loss = loss_function(pred, val_y)
                    
    
                    validation_running_loss += loss.item()
                    
    
                    pred_class = np.argmax(pred.cpu().detach().numpy(), axis = 1)
                    validation_running_acc += np.sum(np.array(pred_class) == np.array(val_y.cpu()))
                    
            self.training_loss_save_array.append(training_running_loss / len(self.train_data_loader))
            self.training_acc_save_array.append(training_running_acc / (len(self.train_data_loader) * MINIBATCH_SIZE))
            
            self.val_loss_save_array.append(validation_running_loss / len(self.val_data_loader))
            self.val_acc_save_array.append(validation_running_acc  / (len(self.val_data_loader) * MINIBATCH_SIZE))
            
            print('training loss: {}, training acc: {}'.format(self.training_loss_save_array[epoch - 1], self.training_acc_save_array[epoch - 1]))
            print('validation loss: {}, val acc: {}'.format(self.val_loss_save_array[epoch - 1], self.val_acc_save_array[epoch - 1]))
            
            torch.save(self.resnet.state_dict(), './checkpoint_5/weight_{}_{}.pt'.format(epoch, str(self.val_acc_save_array[epoch - 1])))  
            
            

    def preprocessing(self):
                             
        '''transform = transforms.Compose([
        transforms.Resize(128),
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.485, 0.456, 0.406],
                             std = [0.229, 0.224, 0.225])
        ])'''
        
        training_img, Training_data_label_array = self.ReadFile(training_path)
        val_img, Val_data_lablel_array = self.ReadFile(val_path)
        
       
        #training data preprocessing
        Training_data_image_array = torch.tensor(training_img).type(torch.FloatTensor)
        Val_data_image_array = torch.tensor(val_img).type(torch.FloatTensor)
        
        
        
        #label data preprocessing
        Training_data_label_array = torch.tensor(Training_data_label_array).type(torch.LongTensor)
        Val_data_lablel_array = torch.tensor(Val_data_lablel_array).type(torch.LongTensor)
        

        


        #pytorch dataloader to load data
        torch_training_dataset = Data.TensorDataset(Training_data_image_array, Training_data_label_array)
        torch_val_dataset = Data.TensorDataset(Val_data_image_array, Val_data_lablel_array)
        
                
        self.train_data_loader = Data.DataLoader(
        dataset=torch_training_dataset,
        batch_size=MINIBATCH_SIZE,
        shuffle=True,
        num_workers=0
        )
        
        
        
        self.val_data_loader = Data.DataLoader(
        dataset=torch_val_dataset,
        batch_size=MINIBATCH_SIZE,
        shuffle=False,
        num_workers=0
        )
        
    def ReadFile(self, path):    
        
        
         transform = transforms.Compose([
         transforms.Resize([128, 128]),
         transforms.ToTensor(),
         transforms.Normalize(mean = [0.485, 0.456, 0.406],
                             std = [0.229, 0.224, 0.225])
         ])
        
        
         files = sorted(list(set(os.listdir(path))))
        # print(files)
         label_array = []
         label_index_array = []
         Image_array = []
         for file in files:
             label_name, idx = file.split("000")
             label_array.append(label_name)
             
        
         label_array = set(label_array)
         label_array = sorted(list(label_array))
         
         #print(label_array)
         
         for file in files:
             label_name, idx = file.split("000")
             
             label_index_array.append(label_array.index(label_name))
             
             
             Image_array.append(np.array(transform(Image.open(os.path.join(path, file)).convert('RGB'))))
             #print(np.array(Image_array).shape)
         #print(label_index_array)
         
         #print(files)
             
         return Image_array, label_index_array
    


if __name__ == '__main__':
    r = Reproducible(123)
    resnet_classificer = Training()