import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
from torch.autograd import Variable
from tqdm import tqdm




class Mish(nn.Module):
    def __init__(self):
        super(Mish,self).__init__()

    def forward(self, x):
        x = x * (torch.tanh(F.softplus(x)))
        return x

class DSCConv2d(nn.Module):

    def __init__(self,filters,out_filters,kernel_size=3,padding=0,stride=1):
        super(DSCConv2d,self).__init__()

        self.conv1 = nn.Conv2d(filters,filters, kernel_size=kernel_size,padding=padding,stride=stride,groups = filters)
        self.pointwise = nn.Conv2d(filters,out_filters,1)

    def forward(self,x):

        return self.pointwise(self.conv1(x))

class DSCConvTranspose2d(nn.Module):

    def __init__(self,filters,out_filters,kernel_size=3,padding=0,stride=1):
        super(DSCConvTranspose2d,self).__init__()

        self.conv1 = nn.ConvTranspose2d(filters,filters, kernel_size=kernel_size,padding=padding,stride=stride,groups = filters)
        self.pointwise = nn.Conv2d(filters,out_filters,1)

    def forward(self,x):

        return self.pointwise(self.conv1(x))

class loop_conv(nn.Module):

    def __init__(self,l,filters,k,p,s,activation):
        super(loop_conv,self).__init__()
        model = []
        activations = {'mish':Mish(),
                       'leaky':nn.LeakyReLU()}
        for i in range(l):
            
            model += [nn.Sequential(
                                    DSCConv2d(filters[i],filters[i+1], kernel_size=k[i],padding=p[i],stride=s[i]),
                                    activations[activation],
                                    nn.InstanceNorm2d(filters[i+1], affine=True)
                                    )]
        self.module = nn.ModuleList(model)
        
    def forward(self,x,l=None):
        
        for i in self.module:
            x = i(x)
            
        return x

class loop_deconv(nn.Module):

    def __init__(self,l,filters,activation,latent_dim):
        super(loop_deconv,self).__init__()
        model = []
        activations = {'mish':Mish(),
                       'leaky':nn.LeakyReLU()}
        norm = []
        for i in range(l):
            model.append(nn.Sequential(
                                    DSCConvTranspose2d(filters[i],filters[i+1], kernel_size=4,padding=1,stride=2),
                                    activations[activation]))
            norm.append(AdaIN(latent_dim,filters[i+1]))
                                    
        self.model = nn.ModuleList(model)
        self.norm = nn.ModuleList(norm)
    def forward(self,x,l):
        
        for i,y in zip(self.model,self.norm):
            x = i(x)
            x = y(x,l)
            
        return x

class AdaIN(nn.Module):
    def __init__(self, style_dim, num_features):
        super().__init__()
        self.norm = nn.InstanceNorm2d(num_features, affine=False)
        self.fc = nn.Linear(style_dim, num_features*2)

    def forward(self, x, s):
        h = self.fc(s)
        h = h.view(h.size(0), h.size(1), 1, 1)
        gamma, beta = torch.chunk(h, chunks=2, dim=1)
        
        return (1 + gamma) * self.norm(x) + beta

class ResBlock(nn.Module):
    def __init__(self, latent_dim,in_f,filters,activation,norm='None',upsample=False,pool=False):
        super(ResBlock, self).__init__()
        
        norms = {'IN':nn.InstanceNorm2d(filters,affine=True),
                'AdaIN':AdaIN(latent_dim,filters),
                'None':None}
        activations = {'mish':Mish(),
                       'leaky':nn.LeakyReLU()}
        if upsample:
            self.conv_1 = nn.Sequential(
                                        DSCConvTranspose2d(in_f,filters, kernel_size=4,padding=1, stride=2),
                                        activations[activation])
            self.conv_2 = nn.Sequential(
                                        DSCConvTranspose2d(filters,filters, kernel_size=3,padding=1, stride=1),
                                        activations[activation])

        else:
            self.conv_1 = nn.Sequential(
                                        DSCConv2d(in_f,filters, kernel_size=3,padding=1, stride=1),
                                        activations[activation])
            self.conv_2 = nn.Sequential(
                                        DSCConv2d(filters,filters, kernel_size=3,padding=1, stride=1),
                                        activations[activation])
            
        self.norm = norms[norm]
        
        self.is_norm = norm
        self.pool = pool
        
    
    def forward(self, x,l=None):
        
        x = self.conv_1(x)
        y = x + self.conv_2(x)
        if self.pool:
            y = F.avg_pool2d(y,2)
        if self.is_norm=="IN":
            y = self.norm(y)
        elif self.is_norm=="AdaIN":
            y = self.norm(y,l)    
        
        return y

class MappingNetwork(nn.Module):
    def __init__(self, latent_dim=16, style_dim=64, num_domains=2):
        super(MappingNetwork,self).__init__()
        

        self.shared = nn.ModuleList([nn.Sequential(nn.Linear(i, 512), Mish()) for i in [latent_dim]+[512]*3 ])    

        self.unshared = nn.ModuleList([nn.Sequential(nn.Linear(512, 512),
                                            Mish(),
                                            nn.Linear(512, 512),
                                            Mish(),
                                            nn.Linear(512, 512),
                                            Mish(),
                                            nn.Linear(512, style_dim)) for i in range(num_domains)])
    def forward(self,z):
            
            for q in self.shared:
                z = q(z)
            
            
            output = torch.stack([i(z) for i in self.unshared],1).cuda() 
            
            return output

class Generator(nn.Module):
    """Generator network"""
    def __init__(self, img_size = 256, latent_dim = 16,style_dim=64,num_domains=2):
        super(Generator, self).__init__()

        self.downsample = loop_conv(1,[3,64],[7],[3],[1],'mish')

        filters= [64,128,256]+ [512]*7 +[256,128]
        filters_out = filters[1:] + [64]
        self.res = nn.ModuleList([ResBlock(style_dim,i,j,'mish',y,u,p) \
                                for i,j,y,u,p in zip(filters,filters_out,['IN']*6 + ['AdaIN']*6,[False]*8+[True]*4,[True]*4+[False]*8)
                                 ])

        #self.upsample = loop_deconv(3,[512,256,128,64],'mish',style_dim)

        self.conv = DSCConv2d(64,3, 7,padding=3,stride=1)

        

    def forward(self,x,l):
        
        x = self.downsample(x)
        for i in self.res:
            x = i(x,l)
        #x = self.upsample(x,l)
        x = self.conv(x)
        x = torch.tanh(x)
        
        return x

class StyleEncoder(nn.Module):
    def __init__(self, img_size=256, style_dim=64, num_domains=2):
    
        super(StyleEncoder, self).__init__()

        '''
        p = int(np.log2(img_size)) 

        self.shared = loop_conv(p,[3]+[2**n for n in range(6,6+p)],[4]*p,[1]*p,[2]*p,'leaky')
        '''
        filters= [64,128,256]+ [512]*3 
        filters_out = filters[1:] + [512]
        self.conv_1 = loop_conv(1,[3,64],[7],[3],[1],'leaky')

        self.res = nn.ModuleList([ResBlock(style_dim,i,j,'leaky',pool=p) \
                                for i,j,p in zip(filters,filters_out,[True]*6)
                                 ])
        self.conv_2 = DSCConv2d(512,512, 4,padding=1,stride=2)

        self.unshared = nn.ModuleList([nn.Linear(512, style_dim) for i in range(num_domains)])
        
    def forward(self, x):

        y = self.conv_1(x)
        for i in self.res:
            y = i(y)
        y = self.conv_2(y)
        y = y.view(y.size(0), -1)
        output = []
        for layer in self.unshared:
            output.append(layer(y))
        output = torch.stack(output, dim=1)  
        
        return output

class Discriminator(nn.Module):
    def __init__(self, img_size=256, style_dim=64, num_domains=2):
        super(Discriminator, self).__init__()

        
        '''
        p = int(np.log2(img_size)) 

        self.shared = loop_conv(p,[3]+[2**n for n in range(6,6+p)],[4]*p,[1]*p,[2]*p,'leaky')
        '''
        filters= [64,128,256]+ [512]*3 
        filters_out = filters[1:] + [512]
        self.conv_1 = loop_conv(1,[3,64],[7],[3],[1],'leaky')

        self.res = nn.ModuleList([ResBlock(style_dim,i,j,'leaky',pool=p) \
                                for i,j,p in zip(filters,filters_out,[True]*6)
                                 ])
        self.conv_2 = DSCConv2d(512,512, 4,padding=1,stride=2)

        self.unshared = nn.ModuleList([nn.Linear(512, 1) for i in range(num_domains)])

    def forward(self, x):
        y = self.conv_1(x)
        for i in self.res:
            y = i(y)
        y = self.conv_2(y)
        y = y.view(y.size(0), -1)
        output = []
        for layer in self.unshared:
            output.append(layer(y))
        output = torch.stack(output, dim=1)  
        
        return output  
        
        

class Model(nn.Module):

    def __init__(self,latent_dim,style_dim,img_size,num_domains,lr,map_lr,beta,device):

        super(Model,self).__init__()

        self.map = MappingNetwork(latent_dim,style_dim).to(device)
        self.generator = Generator(img_size,latent_dim,style_dim,num_domains).to(device)
        self.style_enc = StyleEncoder(img_size,style_dim,num_domains).to(device)
        self.discriminator = Discriminator(img_size,style_dim,num_domains).to(device)
        self.entropy_loss = nn.BCEWithLogitsLoss()
        self.reconstruction_loss = nn.BCEWithLogitsLoss()
        self.dsc_optim = optim.Adam(self.discriminator.parameters(), lr=lr, betas=beta)
        self.gen_optim = optim.Adam(self.generator.parameters(), lr=lr, betas=beta)
        self.style_optim = optim.Adam(self.style_enc.parameters(), lr=lr, betas=beta)
        self.map_optim = optim.Adam(self.map.parameters(), lr=map_lr, betas=beta)
        self.device = device
        self.style_dim = style_dim
        
        

    def forward(self,img,domain,og_domain,z=None,x=None):

        assert (z is None) != (x is None)
        output = torch.arange(0,domain.size(0))
        if z:
            z1,z2 = z
            l = self.map(z1)[output,domain]
            l2 = self.map(z2)[output,domain]
        else:
            x1,x2 = x
            l = self.style_enc(x1)[output,domain]
            
            l2 = self.style_enc(x2)[output,domain]
        
        
        
        fake = self.generator(img,l)

        real_cls = self.discriminator(img)[output,og_domain]
        
        fake_cls = self.discriminator(fake)[output,domain]

        style = self.style_enc(fake)[output,domain]

        real_style = self.style_enc(img)[output,og_domain]
        
        reconstruct = self.generator(fake,real_style)
        
        disc_loss = (self.entropy_loss(real_cls,torch.ones_like(real_cls).float().to(self.device)) +
                     self.entropy_loss(fake_cls,torch.zeros_like(fake_cls).float().to(self.device))).to(self.device)  

        gen_adv_loss = self.entropy_loss(fake_cls,torch.ones_like(fake_cls).float().to(self.device))        
        
        
        cycle_loss = torch.abs(img-reconstruct).mean().to(self.device)

        style_loss = torch.abs(l - style).mean().to(self.device)


        fake2 = self.generator(img,l2)
        diversity_loss = torch.abs(fake-fake2).mean().to(self.device)

        self.dsc_optim.zero_grad()
        self.map_optim.zero_grad()
        self.style_optim.zero_grad()
        self.gen_optim.zero_grad()
        disc_loss.backward(retain_graph=True)
        
        grad = []
        for i in self.discriminator.parameters():
            grad += [torch.sum(i.grad.pow(2))] 
        
        
        penalty = 0.5 * torch.tensor(grad).mean(0)

        disc_loss += penalty.to(self.device)

        gen_loss = (gen_adv_loss + style_loss - diversity_loss + cycle_loss).to(self.device)

        return disc_loss, gen_loss

        
class StarGan_v1_5():

    def __init__(self,config,device):

        self.latent_dim = config['latent_dim']
        self.num_domain = config['num_domains']
        self.style_dim = config['style_dim']
        self.img_size = config['img_size']
        self.device = device
        self.batch = config['batch_size']
        self.model = Model(self.latent_dim,self.style_dim,
                           self.img_size,self.num_domain,config['lr'],config['map_lr'],config['beta'],self.device)

    def save(self,path):
        torch.save(self.model.state_dict(), path+'model.pth')
        torch.save(self.model.dsc_optim.state_dict(), path+'dsc_optim.pth')
        torch.save(self.model.gen_optim.state_dict(), path+'gen_optim.pth')
        torch.save(self.model.style_optim.state_dict(), path+'style_optim.pth')
        torch.save(self.model.map_optim.state_dict(), path+'map_optiml.pth')

    def load(self,path):

        self.model.load_state_dict(torch.load(path+'model.pth'))
        self.model.dsc_optim.load_state_dict(torch.load(path))
        self.model.dsc_optim.load_state_dict( torch.load(path+'dsc_optim.pth'))
        self.model.gen_optim.load_state_dict( torch.load(path+'gen_optim.pth'))
        self.model.style_optim.load_state_dict( torch.load(path+'style_optim.pth'))
        self.model.map_optim.load_state_dict( torch.load(path+'map_optiml.pth'))
    
    
    
    def train(self, dataloader):
        torch.cuda.empty_cache()
        self.model.train()
        epoch_logs = {"gen_loss": [],"disc_loss": []}
        
        for indx, data in tqdm(enumerate(dataloader)):
            
            img,og_domain,x1,x2,domain = data
            img = img.to(self.device)
            x1 = x1.to(self.device)
            x2 = x2.to(self.device)
            z1 = torch.normal(torch.tensor([0.5]).repeat(self.batch,self.latent_dim),1).to(self.device)
            z2 = torch.normal(torch.tensor([0.5]).repeat(self.batch,self.latent_dim),1).to(self.device)
            
            disc_loss, gen_loss = self.model(img,domain,og_domain,z = (z1,z2))

            self.model.dsc_optim.zero_grad()
            disc_loss.backward(retain_graph=True)
            self.model.map_optim.zero_grad()
            self.model.style_optim.zero_grad()
            self.model.gen_optim.zero_grad()
            gen_loss.backward()
            self.model.dsc_optim.step()
            self.model.map_optim.step()
            self.model.gen_optim.step()
            self.model.style_optim.step()
            

            disc_loss2, gen_loss2 = self.model(img,domain,og_domain,x = (x1,x2))

            self.model.dsc_optim.zero_grad()
            disc_loss2.backward(retain_graph=True)
            self.model.gen_optim.zero_grad()
            gen_loss2.backward()
            self.model.dsc_optim.step()
            
            
            
            self.model.gen_optim.step()
            
            epoch_logs["gen_loss"].append((gen_loss+gen_loss2).mean().item())
            epoch_logs["disc_loss"].append((disc_loss+disc_loss2).mean().item())

        epoch_logs["gen_loss"] = torch.tensor(epoch_logs["gen_loss"]).mean()
        epoch_logs["disc_loss"] = torch.tensor(epoch_logs["disc_loss"]).mean()

            


        
        return epoch_logs
    
    def valid(self, dataloader):
        torch.cuda.empty_cache()
        self.model.eval()


        epoch_logs = {"gen_loss": [],"disc_loss": []}
        
        for indx, data in tqdm(enumerate(dataloader)):
            
            img,og_domain,x1,x2,domain = data
            img = img.to(self.device)
            x1 = x1.to(self.device)
            x2 = x2.to(self.device)
            z1 = torch.normal(torch.tensor([0.5]).repeat(self.batch,self.latent_dim),1).to(self.device)
            z2 = torch.normal(torch.tensor([0.5]).repeat(self.batch,self.latent_dim),1).to(self.device)
            
            disc_loss, gen_loss = self.model(img,domain,og_domain,z = (z1,z2))

            disc_loss2, gen_loss2 = self.model(img,domain,og_domain,x = (x1,x2))

            epoch_logs["gen_loss"].append((gen_loss+gen_loss2).mean().item())
            epoch_logs["disc_loss"].append((disc_loss+disc_loss2).mean().item())

        epoch_logs["gen_loss"] = torch.tensor(epoch_logs["gen_loss"]).mean()
        epoch_logs["disc_loss"] = torch.tensor(epoch_logs["disc_loss"]).mean()


        return epoch_logs




        
        












