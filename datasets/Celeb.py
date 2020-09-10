from torchvision import transforms,datasets 
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset
import torch
from PIL import Image
import numpy as np
import os

class CelebData(Dataset):

    def __init__(self,config,split):
        super().__init__()
        self.img_size = config['img_size']
        self.batch = config['batch_size']
        self.root = os.path.join(config["root"],split)
        self.dict = config["classes"]
        self.split = split
        self.load_data()

    def load_data(self):
        
        self.male = os.listdir(os.path.join(self.root,'male'))
        self.female = os.listdir(os.path.join(self.root,'female'))
        self.data = np.array(self.female + self.male)
        
    def load_image(self,idx):

        _class = 0 
        if idx > len(self.female):
            _class = 1

        _file = self.data[idx]

        img = TF.to_tensor(TF.resize(Image.open(os.path.join(self.root,self.dict[_class],_file)),self.img_size))
        
        index = np.random.choice(self.data.shape[0],1)[0]
        file1 = self.data[index]
        

        if index>len(self.female):
            domain = 1
            file2 = np.random.choice(np.delete(self.data,index,0)[len(self.female):],1)[0]
        else:
            domain = 0
            file2= np.random.choice(np.delete(self.data,index,0)[:len(self.female)-1],1)[0]

        ref1 = TF.to_tensor(TF.resize(Image.open(os.path.join(self.root,self.dict[domain],file1)),self.img_size))
        ref2 = TF.to_tensor(TF.resize(Image.open(os.path.join(self.root,self.dict[domain],file2)),self.img_size))

        return img,_class,ref1,ref2,domain

    def __getitem__(self, idx):

        img,og_domain,ref1,ref2,domain = self.load_image(idx)
        
        data = (img,
               torch.tensor(og_domain).long(),
               ref1,
               ref2,
               torch.tensor(domain).long())
            
        
        return data

    def __len__(self):
        return self.data.shape[0]

    
    
