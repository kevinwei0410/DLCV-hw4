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
import torch.nn.functional as F
import math
from torchvision.transforms.functional import resize
from skimage.transform import resize


training_path = "./hw4_data/hw4_data/mini/train/"
val_path = "./hw4_data/hw4_data/mini/val/"
trainging_csv = "./hw4_data/hw4_data/mini/train.csv"
val_csv = "./hw4_data/hw4_data/mini/val.csv"



#debug -- print all elements in array
np.set_printoptions(threshold = np.inf)


import warnings; warnings.simplefilter('ignore')




QUERY = 15
TRAIN_WAY = 5

# unchangable
TEST_WAY = 5
SHOT = 10

#training hyperparameter
gpu_paralell = False
MINIBATCH_SIZE_train = 400
MINIBATCH_SIZE_val = 100
gpu_id = 0
EPOCH = 100



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

class DataPreprocessing():
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
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        path, label = self.data[i], self.label[i]
        image = self.transform(Image.open(path).convert('RGB'))
        return image, label


class sampling():

    def __init__(self, label, n_batch, n_cls, n_per):
        self.n_batch = n_batch
        self.n_cls = n_cls
        self.n_per = n_per

        label = np.array(label)
        self.m_ind = []
        for i in range(max(label) + 1):
            ind = np.argwhere(label == i).reshape(-1)
            ind = torch.from_numpy(ind)
            self.m_ind.append(ind)
    
    def __len__(self):
        return self.n_batch
    
    def __iter__(self):
        for i_batch in range(self.n_batch):
            batch = []
            classes = torch.randperm(len(self.m_ind))[:self.n_cls]
            for c in classes:
                l = self.m_ind[c]
                pos = torch.randperm(len(l))[:self.n_per]
                batch.append(l[pos])
            #print("batch",len(batch))
            batch = torch.stack(batch).t().reshape(-1)
            #sprint("batch",np.array( batch).shape)
            yield batch

class Convnet(nn.Module):

    def __init__(self, in_channel = 3, hid_channel = 64, out_channel = 32):
        super().__init__()
        self.encoder = nn.Sequential(
            conv_block(in_channel, hid_channel),
            conv_block(hid_channel, hid_channel),
            conv_block(hid_channel, hid_channel),
            conv_block(hid_channel, out_channel),
        )
        self.out_channels = 32
        
        self.fc1 = nn.Linear(out_channel * TRAIN_WAY * TRAIN_WAY,512)
        self.fc2 = nn.Linear(512,128)
        
    def forward(self, x):
        x = self.encoder(x)  
        # data shot : train_way * (out_channel * train_way * train_way) 5*1600
        # data query : (train_way * query) * (out_channel * train_way * train_way) 75*1600
        #print(x.shape)
        x = x.view(x.size(0), -1)
        
        #print(x.shape)
        
        dout = nn.functional.relu(self.fc1(x))
        dout = nn.functional.relu(self.fc2(dout))
        return dout 

def conv_block(in_channels, out_channels):
    bn = nn.BatchNorm2d(out_channels)
    nn.init.uniform_(bn.weight)
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        bn,
        nn.ReLU(),
        nn.MaxPool2d(2)
    )




class parametric_function(nn.Module):
    def __init__(self):
        super(parametric_function, self).__init__()
        
        self.fc1 = nn.Linear(1280,5) 
        #self.fc2 = nn.Linear(512,128)
        self.fc3 = nn.Linear(512,5) 
        
    def forward(self,a, b):
        a = a.unsqueeze(1).expand(a.shape[0], b.shape[0], -1)
        b = b.unsqueeze(0).expand(a.shape[0], b.shape[0], -1)
        
        a = a.contiguous().view(75, -1) 
        b = b.contiguous().view(75, -1)
        
        #print(a.shape)
        #print(b.shape)
        
        concat = torch.cat((a, b), dim = 1)
        vec = self.fc1(concat)
        #vec = self.fc2(vec)
        
        return self.fc3(vec)


class Training():
    def __init__(self):
        
        
        
        trainset = DataPreprocessing(training_path, trainging_csv)
        train_sampler = sampling(trainset.label, MINIBATCH_SIZE_train, TRAIN_WAY, SHOT + QUERY)
        self.train_data_loader = DataLoader(dataset=trainset, batch_sampler=train_sampler, num_workers=0, pin_memory=True)
    
        valset = DataPreprocessing(val_path, val_csv)
        val_sampler = sampling(valset.label, MINIBATCH_SIZE_val, TEST_WAY, SHOT + QUERY)
        self.val_data_loader = DataLoader(dataset=valset, batch_sampler=val_sampler, num_workers=0, pin_memory=True)



        self.main_model = Convnet()
        
        # gpu device checking
        self.use_gpu = torch.cuda.is_available()
        print("Ready to use GPU : {}".format(self.use_gpu))
        if self.use_gpu:
            if gpu_paralell:
                self.main_model = torch.nn.DataParallel(self.main_model).cuda()
            else:
                torch.cuda.set_device(gpu_id)
                self.main_model.cuda()



        t = self.training()
        
        
    def euclidean_metric(self, a, b):
        n = a.shape[0]
        m = b.shape[0]
        a = a.unsqueeze(1).expand(n, m, -1)
        b = b.unsqueeze(0).expand(n, m, -1)
        logits = -((a - b)**2).sum(dim=2)
        
       #print(logits.shape)
        
        return logits
    
    
    def cos_similarity(self, a, b):
        n = a.shape[0]
        m = b.shape[0]
        a = a.unsqueeze(1).expand(n, m, -1)
        b = b.unsqueeze(0).expand(n, m, -1)
        
       # print(a.shape)
        #print(b.shape)
        
        #a_norm = np.linalg.norm(a, axis=1, keepdims=True)
        #b_norm = np.linalg.norm(b, axis=1, keepdims=True)
        #similiarity = np.dot(a, b.T)/(a_norm * b_norm) 
        #cosine_distance = 1. - similiarity
        
        cos = nn.CosineSimilarity(dim=2, eps=1e-6)
        cosine_distance = cos(a, b)
        
        
        return cosine_distance



    def training(self):

        self.training_loss_save_array = []
        self.val_loss_save_array = []
        self.training_acc_save_array = []
        self.val_acc_save_array = []
        
        

        optimizer = torch.optim.Adam(self.main_model.parameters(), lr=0.001)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
        
        
        
        if self.use_gpu:
            self.param_func_model = parametric_function().cuda()
        optimizer_param_func_model = torch.optim.Adam(self.param_func_model.parameters(), lr=0.001)
        
        
        print("start to train...")
        for epoch in range(1, EPOCH + 1):
            print("epoch {}".format(epoch))
            training_running_acc = 0.0
            validation_running_acc = 0.0
            training_running_loss = 0.0
            validation_running_loss = 0.0
            
            self.main_model.train()
            
            for step, (batch_x, batch_y) in enumerate(self.train_data_loader):
                #print("step", step)
                ##print(batch_x.shape)


                if self.use_gpu:
                    batch_x, batch_y = torch.tensor(batch_x.cuda()), torch.tensor(batch_y.cuda())
                
                p = SHOT * TRAIN_WAY
                data_shot, data_query = batch_x[:p], batch_x[p:]
                
                
                optimizer.zero_grad()
                proto = self.main_model(data_shot)
                
               # print(proto.shape)
                
                proto = proto.reshape(SHOT, TRAIN_WAY, -1).mean(dim=0)
                #print("xxx", proto.shape)
                #print("www", data_query.shape)
    
                label = torch.arange(TRAIN_WAY).repeat(QUERY)
                label = label.type(torch.cuda.LongTensor)
    
                #logits = self.euclidean_metric(self.main_model(data_query), proto)
                #logits = self.cos_similarity(self.main_model(data_query), proto)
                logits = self.param_func_model(self.main_model(data_query), proto) 
                
                
                #print(logits.shape)
                #print(label.shape)
                
                loss = F.cross_entropy(logits, label)
                
                pred_class = np.argmax(logits.cpu().detach().numpy(), axis = 1)
                
                #print(np.array(pred_class).shape)
                training_running_acc += np.sum(np.array(pred_class) == np.array(label.cpu()))
                training_running_loss += loss
                #print(pred_class)
                #print(label)
                
                
                loss.backward() 
                optimizer.step() 
                #optimizer_param_func_model.step()
                
                proto = None; logits = None; loss = None
                

            self.main_model.eval()
            with torch.no_grad():
                if epoch % 1 == 0:
                        
                    for step, (val_x, val_y) in enumerate(self.val_data_loader):
                        if self.use_gpu:
                            val_x, val_y = torch.tensor(val_x.cuda()),  torch.tensor(val_y.cuda())
                            
                            
                        p = SHOT * TEST_WAY
                        data_shot, data_query = val_x[:p], val_x[p:]
                        
                        proto = self.main_model(data_shot)
        
                        proto = proto.reshape(SHOT, TEST_WAY, -1).mean(dim=0)
        
                        label = torch.arange(TEST_WAY).repeat(QUERY)
                        label = label.type(torch.cuda.LongTensor)
            
                        #logits = self.euclidean_metric(self.main_model(data_query), proto)
                        #logits = self.cos_similarity(self.main_model(data_query), proto)
                        logits = self.param_func_model(self.main_model(data_query), proto) 
                        
                        loss = F.cross_entropy(logits, label)
                        
                        
        
                        pred_class = np.argmax(logits.cpu().detach().numpy(), axis = 1)
                        validation_running_acc += np.sum(np.array(pred_class) == np.array(label.cpu()))
                        validation_running_loss += loss
                        
                        proto = None; logits = None; loss = None
                    
            lr_scheduler.step()
                    
            self.training_loss_save_array.append(training_running_loss / len(self.train_data_loader))
            self.training_acc_save_array.append(training_running_acc / (QUERY * TRAIN_WAY * MINIBATCH_SIZE_train))
            
            print(len(self.train_data_loader) * MINIBATCH_SIZE_train)
            
            self.val_loss_save_array.append(validation_running_loss / len(self.val_data_loader))
            self.val_acc_save_array.append(validation_running_acc  / (QUERY * TEST_WAY * MINIBATCH_SIZE_val))
            
            print('training loss: {}, training acc: {}'.format(self.training_loss_save_array[epoch - 1], self.training_acc_save_array[epoch - 1]))
            print('validation loss: {}, val acc: {}'.format(self.val_loss_save_array[epoch - 1], self.val_acc_save_array[epoch - 1]))
            
            #torch.save(self.main_model.state_dict(), './checkpoint2/weight_{}.pt'.format(epoch))  


    
if __name__ == "__main__":
    R = Reproducible(123)
    Prototypical_NET = Training()
    

    