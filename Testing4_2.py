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
import csv
from PIL import Image
from matplotlib import pyplot as plt
from skimage.transform import resize
import sys
import warnings; warnings.simplefilter('ignore')



gpu_paralell = False
gpu_id = 0

MINIBATCH_SIZE = 32


val_path = ""
val_csv = ""
output_csv = ""

weight_path = "./weight_16.pt"


label_array = ['Alarm_Clock', 'Backpack', 'Batteries', 'Bed', 'Bike', 'Bottle', 'Bucket', 'Calculator', 'Calendar', 'Candles', 'Chair', 'Clipboards', 'Computer', 'Couch', 'Curtains', 'Desk_Lamp', 'Drill', 'Eraser', 'Exit_Sign', 'Fan', 'File_Cabinet', 'Flipflops', 'Flowers', 'Folder', 'Fork', 'Glasses', 'Hammer', 'Helmet', 'Kettle', 'Keyboard', 'Knives', 'Lamp_Shade', 'Laptop', 'Marker', 'Monitor', 'Mop', 'Mouse', 'Mug', 'Notebook', 'Oven', 'Pan', 'Paper_Clip', 'Pen', 'Pencil', 'Postit_Notes', 'Printer', 'Push_Pin', 'Radio', 'Refrigerator', 'Ruler', 'Scissors', 'Screwdriver', 'Shelf', 'Sink', 'Sneakers', 'Soda', 'Speaker', 'Spoon', 'TV', 'Table', 'Telephone', 'ToothBrush', 'Toys', 'Trash_Can', 'Webcam']


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
        
        
class Testing():
    def __init__(self):
        
        #data
        p = self.preprocessing()

        self.resnet = models.resnet50(pretrained=False)
        
        
       
       
        
        #model setup
        self.resnet.fc = nn.Linear(2048, 65)
        #print(self.resnet)

        self.resnet = self.resnet.cuda()
        
        
        self.resnet.load_state_dict(torch.load(weight_path))
        # gpu device checking
        self.use_gpu = torch.cuda.is_available()
        print("Ready to use GPU : {}".format(self.use_gpu))
        

        
        self.Testing()

    def Testing(self):
        


        
        self.val_loss_save_array = []
        self.val_acc_save_array = []
        


        validation_running_acc = 0.0
        validation_running_loss = 0.0
        
        self.resnet.eval()

        pred_class_ans = []
        for step, (val_x, val_y) in enumerate(self.val_data_loader):
            if self.use_gpu:
                val_x, val_y = torch.tensor(val_x.cuda()),  torch.tensor(val_y.cuda())
            
            #loss = self.learner(val_x)
            pred = self.resnet(val_x)
            #loss = loss_function(pred, val_y)
            

            #validation_running_loss += loss.item()
            

            pred_class = np.argmax(pred.cpu().detach().numpy(), axis = 1)
            validation_running_acc += np.sum(np.array(pred_class) == np.array(val_y.cpu()))
            pred_class_ans.extend(pred_class)
            
        
        pred_class_ans_string = []
        for ans in pred_class_ans:
            pred_class_ans_string.append(str(label_array[int(ans)]))
            
        self.WriteToCSV(pred_class_ans_string, output_csv)

    def preprocessing(self):

        
        Val_data_image_array = self.ReadFile(val_path)
        Val_data_lablel_array = self.LabelGenerate(val_path)
       
        #training data preprocessing
        Val_data_image_array = torch.tensor(Val_data_image_array).type(torch.FloatTensor)
        Val_data_lablel_array = torch.tensor(Val_data_lablel_array).type(torch.LongTensor)

        

        


        #pytorch dataloader to load data
        torch_val_dataset = Data.TensorDataset(Val_data_image_array, Val_data_lablel_array)
        

        
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
         
         Image_array = []
             
        
         #label_array = set(label_array)
         #label_array = sorted(list(label_array))
         

         for file in files:
             #label_name, idx = file.split("000")
             
             #label_index_array.append(label_array.index(label_name))
             
             
             Image_array.append(np.array(transform(Image.open(os.path.join(path, file)).convert('RGB'))))
             
         return Image_array
     
    def LabelGenerate(self, path):
        Label_array = []
        files = sorted(list(set(os.listdir(path))))
        for file in files:
            Label_array.append(0)
        return Label_array
    
    def WriteToCSV(self,result, csv_path): 
        file_name = sorted(list(set(os.listdir(val_path))))
        
        first_row = ["id", "file_name", "label"]
        
        idx = 0
        with open(csv_path, 'w', newline='') as csvfile:
              writer = csv.writer(csvfile)
              writer.writerow(first_row)
              row = [idx] # id 0
              for res in range(1, len(result) + 1, 1):
                  row.append(file_name[res - 1])
                  row.append(result[res - 1])
                  writer.writerow(row)
                  idx += 1
                  row = [idx]
   
                  
   
def eval_acc(path):
    #file_name = sorted(list(set(os.listdir(path))))
    with open(path, newline='') as csvfile:
      rows = csv.reader(csvfile)
      rows.__next__() 
      #print(rows)
      count = 0
      for row in rows:
          if row[1].split("000")[0] == row[2]:
              #print(row)
              count += 1
      print("acc : ", count / (406))
    

if __name__ == '__main__':
    val_csv = sys.argv[1]
    val_path = sys.argv[2]
    output_csv = sys.argv[3]
    
    
    r = Reproducible(123)
    resnet_classificer = Testing()
    
    #eval_acc("./output.csv")