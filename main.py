from datasets import Celeb
from utils import common
from torch.utils.data import DataLoader
from modules import StarGAN
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np 
import torchvision

dataset_helper = {
    "Celeb": Celeb.CelebData
}

model_helper = {
    "StarGAN" : StarGAN.StarGan_v1_5
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

            
            self.source,_,_,_,_ = next(iter(self.train_dataloader))
            self.ref, self.ref_domain,_,_,_ = next(iter(self.train_dataloader))
            self.grid = torchvision.utils.make_grid(self.source)
            self.ref_grid = torchvision.utils.make_grid(self.ref)
            
        if "val" in self.config["SPLIT"]:
            val_dataset = dataset_helper[self.config["dataset"]](self.config, "val")
            self.val_dataloader = DataLoader(
                    dataset=val_dataset, 
                    batch_size=self.config["batch_size"], 
                    shuffle=False, 
                    drop_last=False, 
                    num_workers=self.config["num_workers"]
                )
    def re__init(self,epoch_path):
        self.epochs = np.load(epoch_path,allow_pickle=True)[0]
        self.model.load(f'{self.config["weight_save"]}/{self.config["model"]}/{i}')

    def _run(self):
        print ("start training")
        self.writer.add_image("source",self.grid,0)
        self.writer.add_image("ref",self.ref_grid,0)
        epoch = self.epochs
        
        for i in tqdm(range(epoch,self.config["epochs"])):
            train_log = self.model.train(self.train_dataloader)
            print(f"train metrics: disc_loss - {train_log['disc_loss']}  gen_loss - {train_log['gen_loss']}")

            self.writer.add_scalar("train/disc_loss",train_log["disc_loss"],i)
            self.writer.add_scalar("train/gen_loss",train_log["gen_loss"],i)

            if i % self.config["valid_every"] == 0:
                
                self.model.save(f'{self.config["weight_save"]}/{self.config["model"]}/{i}')
                np.save(self.config['parameter'],[self.epochs])
                if "val" in self.config["SPLIT"]:
                    val_log = self.model.valid(self.val_dataloader)
                    print(f"valid metrics: disc_loss - {val_log['disc_loss']}  gen_loss - {val_log['gen_loss']}")
                    self.writer.add_scalar("val/disc_loss",val_log["disc_loss"],i)
                    self.writer.add_scalar("val/gen_loss",val_log["gen_loss"],i)
            if i % self.config["show_every"] == 0:
                torch.cuda.empty_cache()
                self.model.model.eval()
                style = self.model.model.style_enc(self.ref.to(self.device))[torch.arange(0,self.config["batch_size"]),self.ref_domain]
                gen_img = self.model.model.generator(self.source.to(self.device),style) 
                grid = torchvision.utils.make_grid(gen_img)   
                self.writer.add_image("generated_image",grid,i)
        
    
    def close_writer(self):
        self.writer.close()
    def load_model(self,path):
        self.model.load_model(path)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="trainer script")
    parser.add_argument("--config", required=True, type=str, help="path to config file")
    args = parser.parse_args()

    trainer = Trainer(args)
    trainer._run()
    
