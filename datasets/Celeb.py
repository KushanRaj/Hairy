from torchvision import transforms,datasets 
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset
import torch
from PIL import Image
import numpy as np
import os
from glob import glob
import random

class CelebData(Dataset):

    def __init__(self,config,split):
        super().__init__()
        self.img_size = config['img_size']
        self.batch = config['batch_size']
        self.root = os.path.join(config["root"],split)
        self.dict = config["classes"]
        self.split = split
        self.num_domains = config["num_domains"]
        self.load_data()

    def load_data(self):
        
        self.male = list(glob(os.path.join(self.root,'male','*.jpg')))
        self.female = list(glob(os.path.join(self.root,'female','*.jpg')))
        self.people = np.array(self.female + self.male)
        
        self.domain = np.concatenate((np.zeros((len(self.female,))),np.ones((len(self.male,)))))
        self.data = np.stack((self.people,self.domain)).T
        np.random.shuffle(self.data)

        
    def load_image(self,idx):

        

        _file,_class = self.data[idx]

        img = TF.to_tensor(TF.resize(Image.open(_file),self.img_size))
        
        class1 = np.random.randint(0,self.num_domains,(1,))[0]
        
        if class1 == 0:
            file1,file2 = np.random.choice(self.female,(2,))
        else:
            file1,file2 = np.random.choice(self.male,(2,))
        

        ref1 = TF.to_tensor(TF.resize(Image.open(file1),self.img_size))
        ref2 = TF.to_tensor(TF.resize(Image.open(file2),self.img_size))

        return img,_class,ref1,ref2,class1

    def __getitem__(self, idx):

        img,og_domain,ref1,ref2,domain = self.load_image(idx)
        
        data = (img,
               torch.tensor(float(og_domain)).long(),
               ref1,
               ref2,
               torch.tensor(float(domain)).long())
            
        
        return data

    def __len__(self):
        return self.data.shape[0]

    
    
