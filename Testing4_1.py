import os
import sys
import argparse

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import Sampler

import csv
import random
import numpy as np
import pandas as pd

from PIL import Image
filenameToPILImage = lambda y: Image.open(y)

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(SEED)
np.random.seed(SEED)

def worker_init_fn(worker_id):                                                          
    np.random.seed(np.random.get_state()[1][0] + worker_id)

# mini-Imagenet dataset
class MiniDataset(Dataset):
    def __init__(self, csv_path, data_dir):
        self.data_dir = data_dir
        self.data_df = pd.read_csv(csv_path).set_index("id")

        self.transform = transforms.Compose([
            filenameToPILImage,
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

    def __getitem__(self, index):
        path = self.data_df.loc[index, "filename"]
        label = self.data_df.loc[index, "label"]
        image = self.transform(os.path.join(self.data_dir, path))
        return image, label

    def __len__(self):
        return len(self.data_df)

class GeneratorSampler(Sampler):
    def __init__(self, episode_file_path):
        episode_df = pd.read_csv(episode_file_path).set_index("episode_id")
        self.sampled_sequence = episode_df.values.flatten().tolist()

    def __iter__(self):
        return iter(self.sampled_sequence) 

    def __len__(self):
        return len(self.sampled_sequence)
def euclidean_metric(a, b):
        n = a.shape[0]
        m = b.shape[0]
        a = a.unsqueeze(1).expand(n, m, -1)
        b = b.unsqueeze(0).expand(n, m, -1)
        logits = -((a - b)**2).sum(dim=2)
        
       #print(logits.shape)
        
        return logits
def predict(model, data_loader):
    prediction_results = []
    with torch.no_grad():
        # each batch represent one episode (support data + query data)
        for i, (val_x, val_y) in enumerate(data_loader):
            #print("wae")
            #val_x, val_y = torch.tensor(val_x.cuda()),  torch.tensor(val_y.cuda())
            # split data into support and query data
            support_input = val_x[:5 * 1,:,:,:] 
            query_input   = val_x[5 * 1:,:,:,:]

            # create the relative label (0 ~ N_way-1) for query data
            #label_encoder = {val_y[i * args.N_shot] : i for i in range(args.N_way)}
            #query_label = torch.cuda.LongTensor([label_encoder[class_name] for class_name in val_y[args.N_way * args.N_shot:]])

            # TODO: extract the feature of support and query data
            # TODO: calculate the prototype for each class according to its support data

            # TODO: classify the query data depending on the its distense with each prototype
            #print("ssww")
            #print(label_encoder)
            
            proto = model(support_input)

            proto = proto.reshape(1, 5, -1).mean(dim=0)

            label = torch.arange(5).repeat(15)
            label = label.type(torch.cuda.LongTensor)

            logits = euclidean_metric(model(query_input), proto)
            #logits = cos_similarity(model(query_input), proto)
            #logits = param_func_model(main_model(query_input), proto) 
            
            #loss = F.cross_entropy(logits, label)
            
            

            pred_class = np.argmax(logits.cpu().detach().numpy(), axis = 1)
            #validation_running_acc += np.sum(np.array(pred_class) == np.array(label.cpu()))
            #validation_running_loss += loss
            
            prediction_results.extend(pred_class)
            proto = None; logits = None; loss = None
    #print("ss",prediction_results)
    return prediction_results

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
        
        self.fc1 = nn.Linear(out_channel * 5 * 5,512)
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


def WriteToCSV(result, csv_path): 
    #image_id = sorted(list(set(os.listdir(testing_data_path))))
    
    first_row = ["episode_id"]
    for i in range(75):
        first_row.append("query" + str(i))
    
    episode_id = 0
    with open(csv_path, 'w', newline='') as csvfile:
          writer = csv.writer(csvfile)
          writer.writerow(first_row)
          row = [episode_id] # id 0
          for res in range(1, len(result) + 1, 1):
              row.append(result[res - 1])
              if res % 75 == 0:
                  writer.writerow(row)
                  episode_id += 1
                  row = [episode_id]
                  


def parse_args():
    parser = argparse.ArgumentParser(description="Few shot learning")
    parser.add_argument('--N-way', default=5, type=int, help='N_way (default: 5)')
    parser.add_argument('--N-shot', default=1, type=int, help='N_shot (default: 1)')
    parser.add_argument('--N-query', default=15, type=int, help='N_query (default: 15)')
    parser.add_argument('--load', type=str, help="Model checkpoint path")
    parser.add_argument('--test_csv', type=str, help="Testing images csv file")
    parser.add_argument('--test_data_dir', type=str, help="Testing images directory")
    parser.add_argument('--testcase_csv', type=str, help="Test case csv")
    parser.add_argument('--output_csv', type=str, help="Output filename")

    return parser.parse_args()


test_csv = ""
test_data_dir = ""
testcase_csv = ""
output_csv = ""


if __name__=='__main__':
    #args = parse_args()
    
    test_csv = sys.argv[1]
    test_data_dir = sys.argv[2]
    testcase_csv = sys.argv[3]
    output_csv = sys.argv[4]
    

    test_dataset = MiniDataset(test_csv, test_data_dir)

    test_loader = DataLoader(
        test_dataset, batch_size=5 * (15 + 1),
        num_workers=0, pin_memory=False, worker_init_fn=worker_init_fn,
        sampler=GeneratorSampler(testcase_csv))

    # TODO: load your model
    model = Convnet()
    model.load_state_dict(torch.load('weight_77_0.49853333333333333.pt'))
    
    print("start to test")
    prediction_results = predict(model, test_loader)
    
    #print(len(prediction_results))
    
    #print(75 * 600)
    
    # TODO: output your prediction to csv
    WriteToCSV(prediction_results, output_csv)
    
    
    
'''
python test_testcase.py --test_csv ./hw4_data/hw4_data/mini/val.csv --test_data_dir  ./hw4_data/hw4_data/mini/val/ --testcase_csv ./hw4_data/hw4_data/mini/val_testcase.csv --output_csv ./output.csv
'''
    
    