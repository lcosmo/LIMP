import os
# os.environ["CUDA_VISIBLE_DEVICES"]="0"

import scipy.sparse as sp
import torch 
import dill
import pickle
import torch.nn as nn
import shutil

import torch_geometric
from torch_geometric.data.dataloader import DataLoader, DataListLoader


from misc_utils import *
from utils_distance import *
from datasets.faust_2500 import Faust2500Dataset
from models.model import PointNetVAE


device = 'cuda'

dataset_train = Faust2500Dataset('data/faust2500',train=False,test=True,transform_data=False)


#model options
opt = lambda x:x
opt.NUM_POINTS = dataset_train[0].pos.shape[0]
opt.BATCH_SIZE = 16
opt.DESC_SIZE = 512 #pointwise descriptro size after convlolutional layers
opt.LATENT_SPACE = 256 #dimension of the full (pose+style) latent space
opt.POSE_SIZE = 64 #number of dimension dedicated to pose encoding 

opt.LOCAL_TH = 0.1
opt.LEARNING_RATE = 0.1e-4

vae = PointNetVAE(opt).to(device)


#load pretrained model
net_type = '4'

loss_step = '' 
loss_step = '_ae_euc' 
loss_step = '_ae_euc_gd1' 
loss_step = '_ae_euc_gd2' 

vae.load_state_dict(torch.load('pretrained/FAUST_vae_euc_gd.dict'), strict=False)
vae.eval()

alllsp,allx = calc_allsp(vae, device, dataset_train)
allx=allx*[-1, 1, 1]

VERT = dataset_train[0].pos.data.cpu().numpy()
TRIV = dataset_train[0].faces.data.cpu().numpy()


shape1 = 7
shape2 = 12

n = 5
a = np.linspace(0,1,n)[:,None]

#interpolation sequence
print('INTERPOLATION')
int_lsp = alllsp[shape1,:][None,:]*a + (1-a)*alllsp[shape2,:][None,:]
REC_X = decode(vae,device,int_lsp)

R2,t1 = rigid_transform_3D(REC_X[0,:].T,allx[shape2,:].T)
R1,t1 = rigid_transform_3D(REC_X[-1,:].T,allx[shape1,:].T)
Rs = np.asarray(R1[None,...])*a[...,None] + (1-a[...,None])*np.asarray(R2[None,...])

f = plot_colormap([np.matmul(x,R.T) for x,R in zip(REC_X,Rs)],[TRIV]*n,[VERT[:,0]]*n)

#pose transfer
print('POSE TRANSFER')
int_lsp = alllsp[shape1,:][None,:]*a + (1-a)*alllsp[shape2,:][None,:]
int_lsp[:,opt.POSE_SIZE:] = alllsp[shape2,opt.POSE_SIZE:]
REC_X = decode(vae,device,int_lsp)

R2,t1 = rigid_transform_3D(REC_X[0,:].T,allx[shape2,:].T)
R1,t1 = rigid_transform_3D(REC_X[-1,:].T,allx[shape1,:].T)
Rs = np.asarray(R1[None,...])*a[...,None] + (1-a[...,None])*np.asarray(R2[None,...])

f = plot_colormap([np.matmul(x,R.T) for x,R in zip(REC_X,Rs)],[TRIV]*n,[VERT[:,0]]*n)