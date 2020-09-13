from datasets import Celeb
from utils import common
from torch.utils.data import DataLoader,RandomSampler
from modules import StarGAN_DSC
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np 
import torchvision

dataset_helper = {
    "Celeb": Celeb.CelebData
}

model_helper = {
    "StarGAN" : StarGAN_DSC.StarGan_v1_5
}


class Trainer:
    def __init__(self, args):
        self.config = common.read_yaml(args.config)
        self._create_dataloader()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._create_model()
        self.writer = SummaryWriter(log_dir='logdir')
        self.epochs = 0
        self.img_size = self.config['img_size'] 
        

    def _create_model(self):
        self.model = model_helper[self.config["model"]](self.config, self.device)

    def _create_dataloader(self):
        if "train" in self.config["SPLIT"]:
            train_dataset = dataset_helper[self.config["dataset"]](self.config, "train")
            
            self.train_dataloader = DataLoader(
                dataset=train_dataset, 
                batch_size=self.config["batch_size"], 
                shuffle=True, 
                drop_last=False, 
                num_workers=self.config["num_workers"]
            )

            
            
        if "val" in self.config["SPLIT"]:
            val_dataset = dataset_helper[self.config["dataset"]](self.config, "val")
            self.val_dataloader = DataLoader(
                    dataset=val_dataset, 
                    batch_size=self.config["batch_size"], 
                    shuffle=False, 
                    drop_last=False, 
                    num_workers=self.config["num_workers"]
                )
    def re__init(self,epoch_path,div_loss_path):
        self.epochs = np.load(epoch_path,allow_pickle=True)[0]
        self.model.model.diversity_wt = np.load(div_loss_path,allow_pickle=True)[0]
        self.model.load(f'{self.config["weight_save"]}/{self.config["model"]}/{i}')

    def _run(self):
        print ("start training")
        epoch = self.epochs
        sampler = iter(self.train_dataloader)
        for i in tqdm(range(epoch,self.config["epochs"])):
            train_log = self.model.train(self.train_dataloader)
            
            
            
            print(f"train metrics: "+ ' '.join([f'{x} - {y}' for x,y in train_log.items()]))
            for x,y in train_log.items():
                self.writer.add_scalar(f"train/{x}",y,i)
            

            
            if "val" in self.config["SPLIT"]:
                val_log = self.model.valid(self.val_dataloader)
                

                print(f"val metrics: "+ ' '.join([f'{x} - {y}' for x,y in val_log.items()]))
                for x,y in val_log.items():
                    self.writer.add_scalar(f"val/{x}",y,i)
                
            if i % self.config["show_every"] == 0:
                source,_,_,_,_ = next(sampler)
                ref, ref_domain,_,_,_ = next(sampler)
                grid = torchvision.utils.make_grid(source)
                ref_grid = torchvision.utils.make_grid(ref)
                self.writer.add_image("source",grid,i)
                self.writer.add_image("ref",ref_grid,i)
                torch.cuda.empty_cache()
                self.model.model.eval()
                style = self.model.model.style_enc(ref.to(self.device))[torch.arange(0,self.config["batch_size"]),ref_domain]
                gen_img = self.model.model.generator(source.to(self.device),style) 
                grid = torchvision.utils.make_grid(gen_img)   
                self.writer.add_image("generated_image",grid,i)

            self.model.model.diversity_wt = max(0,self.model.model.diversity_wt*(1-i/self.config["epochs"]))

            if i % self.config["save_every"] == 0:
                
                self.model.save(f'{self.config["weight_save"]}/{self.config["model"]}/{i}')
                np.save(self.config['parameter'],[self.epochs])
                np.save(self.config['parameter2'],[self.model.model.diversity_wt])
            
        
    
    def close_writer(self):
        self.writer.close()
    


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="trainer script")
    parser.add_argument("--config", required=True, type=str, help="path to config file")
    args = parser.parse_args()

    trainer = Trainer(args)
    trainer._run()
    
