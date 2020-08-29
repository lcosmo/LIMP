import models.SimplePointNet as SimplePointNet
import torch.nn as nn
import torch


class Encoder(nn.Module):
    def __init__(self, opt):
        nn.Module.__init__(self)
        self.ptnet = SimplePointNet.SimplePointNet(opt.LATENT_SPACE*2,
                                                   opt.DESC_SIZE,
                                                   [32, 128, 256 ],
                                                   [512, 256, 128],
                                                   [ 0 ])
        
    def forward(self,x):
        x = self.ptnet(x)
        return x
    
    
class Decoder(nn.Module):
    def __init__(self, opt):
        nn.Module.__init__(self)
        self.opt = opt;
        self.fc1 = nn.Sequential(nn.Linear(opt.LATENT_SPACE, 1024), 
                        nn.LeakyReLU(), 
                        nn.Linear(1024, 2048),
                        nn.LeakyReLU(), 
                        nn.Linear(2048, opt.NUM_POINTS*3))
    
    def forward(self, x):
        x = self.fc1(x).view(x.shape[0], self.opt.NUM_POINTS,-1)
        return x
    

class PointNetVAE(nn.Module):
    
    def __init__(self,opt):
        nn.Module.__init__(self)
        
        self.opt = opt
        self.enc = Encoder(opt)
        self.dec = Decoder(opt)

    def forward(self,x):
        
        latent = self.enc(x)
        
        if self.train:
            self.z_mu = latent[...,:self.opt.LATENT_SPACE]
            self.z_var  = latent[...,self.opt.LATENT_SPACE:]
            std = torch.exp(self.z_var / 2)
            eps = torch.randn_like(std)

            latent = eps.mul(std).add_(self.z_mu) 
        
        return self.dec(latent[...,:self.opt.LATENT_SPACE])
        
    def enable_bn(self,flag):
        for m in self.modules():
          if isinstance(m, nn.BatchNorm1d):
            if flag:
                m.train()
            else:
                m.eval()