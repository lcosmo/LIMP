#!/usr/bin/env python
# coding: utf-8

import os
#os.environ["CUDA_VISIBLE_DEVICES"]="0"

import scipy.sparse as sp
import torch 
import dill
import pickle
import torch.nn as nn
from random import shuffle

import torch_geometric
from torch_geometric.data.dataloader import DataLoader, DataListLoader

from misc_utils import *
from utils_distance import *
from datasets.faust_2500 import Faust2500Dataset

from models.model import PointNetVAE

device = 'cuda'

# import shutil
# shutil.rmtree('data/faust2500/processed')

dataset_train = Faust2500Dataset('data/faust2500')

#model options
opt = lambda x:x
opt.NUM_POINTS = dataset_train[0].pos.shape[0]
opt.BATCH_SIZE = 16
opt.DESC_SIZE = 512 #pointwise descriptro size after convlolutional layers
opt.LATENT_SPACE = 256 #dimension of the full (pose+style) latent space
opt.POSE_SIZE = 64 #number of dimension dedicated to pose encoding 

opt.LOCAL_TH = 0.1
opt.LEARNING_RATE = 1e-4


## get a randomly selected samples inside the same randomly selected  class
class IntraSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, dataset):
        self.class_ids = dataset.class_ids
    
    def __iter__(self):
        class_idx  = self.class_ids[np.random.randint(len(self.class_ids))]
        class_idxs = [i for i,id in enumerate(self.class_ids) if id==class_idx]
        shuffle(class_idxs)
        return iter(class_idxs)

    def __len__(self):
        print ('\tcalling Sampler:__len__')
        return self.num_samples
    
intra_loader = DataListLoader(dataset_train, batch_size=2, sampler = IntraSampler(dataset_train)) 
interp_loader = DataListLoader(dataset_train, batch_size=2, shuffle=True)
rec_loader = DataListLoader(dataset_train, batch_size=opt.BATCH_SIZE, shuffle=True, num_workers=0)


#DEFINE 3 LOSSES

#optimize gloabl euclidean distortions on data points
def optimize_rec(vae, batch_loader):
    vae.enable_bn(True)
    
    data = batch_loader.__iter__().next()
    
    x = torch.stack([d.pos for d in data]).to(device)
    
    rec = vae(x)    
    
    De_t = pairwise_dists(x)
    De_r = pairwise_dists(rec)

    loss1 = torch.mean((De_t-De_r)**2) #L1 loss on the absoulute euclidean distortion
#     loss2 = torch.mean( ((De_t-De_r)/(De_t+1e-3) )**2 ) #L2 loss on the relative euclidean distortion
    
    loss = loss1#+loss2
    
    kl_loss = 0.5 * torch.mean(torch.exp(vae.z_var) + vae.z_mu**2 - 1.0 - vae.z_var)
    loss += 1e-0*kl_loss
    
    return loss    


def optimize_intep(vae, batch_loader, opt, geoloss=1e1, eucloss=1e1):  
    vae.enable_bn(False)
    
    data = batch_loader.__iter__().next()
#     data = data.to(device)
    T    = data[0].faces[None,...].to(device)

    x = torch.stack([d.pos for d in data]).to(device)
    orix = torch.stack([d.oripos for d in data]).to(device)
    Dg = torch.stack([d.Dg for d in data]).to(device)

    De = calc_euclidean_dist_matrix(orix)    

    latent = vae.enc(x)[...,:vae.opt.LATENT_SPACE]

    a = torch.rand(1).to(device)
    Dg_t = Dg[0]*a+(1-a)*Dg[1]
    De_t = De[0]*a+(1-a)*De[1]

    latent_t = latent[0:1]*a + (1-a)*latent[1:2]

    rec = vae.dec(latent_t)

    loss1=torch.zeros(1).to(device)
    loss2=torch.zeros(1).to(device)

    if geoloss>0:
        Dg_r, grad, div, W, S, C = distance_GIH(rec, T)
        loss1 = geoloss*torch.mean( ((Dg_t-Dg_r.float()))**2)

    if eucloss>0:
        localmask = (Dg_t<opt.LOCAL_TH*Dg_t.max()).float()
        De_r = calc_euclidean_dist_matrix(rec)        
        loss2 = eucloss*torch.sum( localmask*((De_t-De_r)/(De_t+1e-3))**2  )/torch.sum(localmask)

    loss=loss1 + loss2

    return loss  


def optimize_disent_ext(vae, batch_loader, opt, geoloss=1e1):
    vae.enable_bn(False)
    
    data = batch_loader.__iter__().next()
#     data = data.to(device)
    T    = data[0].faces[None,...].to(device)

    x = torch.stack([d.pos for d in data]).to(device)
    orix = torch.stack([d.oripos for d in data]).to(device)
    Dg = torch.stack([d.Dg for d in data]).to(device)
    
    De = calc_euclidean_dist_matrix(orix)    
    
    latent = vae.enc(x)[...,:vae.opt.LATENT_SPACE]

    a = torch.rand(1).to(device)
    Dg_t = Dg[0]

    loss1=torch.zeros(1).to(device)
    loss2=torch.zeros(1).to(device)

    localmask = (Dg_t<opt.LOCAL_TH*Dg_t.max()).float()

    # Interpolationg only the pose latent vector of different subjects should preserve geodesic distances
    latent_t = torch.cat([latent[0:1,:opt.POSE_SIZE]*a + (1-a)*latent[1:2,:opt.POSE_SIZE], latent[0:1,opt.POSE_SIZE:]],-1)
    rec = vae.dec(latent_t)        
    
    Dg_r, grad, div, W, S, C = distance_GIH(rec, T)            
    loss = geoloss*torch.mean( ((Dg_t-Dg_r.float()))**2)
    
    return loss


def optimize_disent_int(vae, batch_loader, opt,  geoloss=1e0, eucloss=1e0):
    vae.enable_bn(False)
    
    data = batch_loader.__iter__().next()
#     data = data.to(device)
    T    = data[0].faces[None,...].to(device)

    x = torch.stack([d.pos for d in data]).to(device)
    orix = torch.stack([d.oripos for d in data]).to(device)
    Dg = torch.stack([d.Dg for d in data]).to(device)
    
    De = calc_euclidean_dist_matrix(orix)    
    
    latent = vae.enc(x)[...,:vae.opt.LATENT_SPACE]

    a = torch.rand(1).to(device)
    Dg_t = Dg[0]

    loss1=torch.zeros(1).cuda()
    loss2=torch.zeros(1).cuda()

    localmask = (Dg_t<opt.LOCAL_TH*Dg_t.max()).float()

    # Interpolationg only the style latent vector of the same subject should not result in changes on it's embedding
    latent_t = torch.cat([latent[0:1,:opt.POSE_SIZE], latent[0:1,opt.POSE_SIZE:]*a + (1-a)*latent[1:2,opt.POSE_SIZE:]],-1)
    rec = vae.dec(latent_t)        
    De_r = calc_euclidean_dist_matrix(rec)        

    loss = eucloss*torch.mean( ((De[0:1]-De_r)/(De[0:1]+1e-3) )**2 ) #L2 loss on the relative euclidean distortion
    
    return loss


# ######### Optimization procedure
# #   0-7000: we start optimizing only for global euclidean distortion on data points 
# #           to avoid local minima introduced by local error metrics
# # 7000-10000: we now start optimizing also for linear interpolation and disentanglement of latent codes. Global
# #           euclidean distortion is not the ideal loss for itnerpolating shapes,  we instead use 
# #           local euclidean metric. We will also use geodesic metric in the next optimization stage.
# # 10000-15000: now that shapes are clean also in the interpolated latent space we can safely
# #           compute geodesic distances (GIH) and use them in the loss terms for interpolation and disentanglement.
# # 15000-: during the last iterations we favor local euclidean loss to remove wrinkles
# #           introduced by geodesics preservation (due to the non exact isometries of pose changes) 


NUM_ITERATIONS = 20000
virtual_batch_size = 20 #multiplier applied to the interations number (should be adapted to the size and complexity of the dataset)

vae = PointNetVAE(opt).to(device)

i=0
optimizer = torch.optim.Adam(vae.parameters(), lr=opt.LEARNING_RATE*0.1)

total_loss=0
losses = []
t=time.time()
for i in range(i,NUM_ITERATIONS+1):
  
    for inner_it in range(virtual_batch_size):
        optimizer.zero_grad()
        loss1 = loss2 = loss3 = loss4 = torch.zeros((1,)).float().to(device)

        loss1 = optimize_rec(vae,rec_loader)

        if i>7000 and i<= 10000: #geodesic weight=0, local_euclidean=1e1 
            loss2 = optimize_intep(vae,interp_loader, opt, 0, 1e1) 
            loss3 = optimize_disent_int(vae,intra_loader, opt, 0,1e1)
        if i>10000 and i<= 15000: #geodesic weight=1e-2, local_euclidean=1e2, dis_euclidean=1e1 
            loss2 = optimize_intep(vae,interp_loader, opt, 1e-2,1e2) 
            loss3 = optimize_disent_int(vae,intra_loader, opt, 1e-2,1e1) 
            loss4 = optimize_disent_ext(vae,interp_loader, opt, 1e1)       
        if i>15000: #geodesic weight=1e-3, local_euclidean=1e1, dis_euclidean=1e0
            loss2 = optimize_intep(vae,interp_loader, opt,1e-3, 1e2) 
            loss3 = optimize_disent_int(vae,intra_loader, opt, 1e-3,1e0)
            loss4 = optimize_disent_ext(vae,interp_loader, opt, 1e0)      

        loss = loss1 + 1e-1*loss2 + 1e1*(1e0*loss3 + 1e0*loss4)
        loss.backward()
        optimizer.step()
    
        losses.append([loss.item(),loss1.item(),loss2.item(),loss3.item(),loss4.item()]) 
    
      
    if i%10==0:
        avg_loss = np.mean(losses[-50:],0)
        print('%d] loss: %.2e  (%.2e, %.2e, %.2e, %.2e),  time: %.2f s' % 
              (i, avg_loss[0],avg_loss[1],avg_loss[2],avg_loss[3],avg_loss[4], time.time()-t))
        t=time.time()
        
    
    #network weights checkpoints
    os.makedirs('trained',exist_ok=True)
    prefix = 'FAUST'
    if i%100 ==0 :
        if i<=3000:
            torch.save(vae.state_dict(), 'trained/'+prefix+'_vae.dict')
        elif i <= 4000:
            torch.save(vae.state_dict(), 'trained/'+prefix+'_vae_euc.dict')
        else:
            torch.save(vae.state_dict(), 'trained/'+prefix+'_vae_euc_gd.dict')
             
                
