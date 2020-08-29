import torch, time, os, sys, numpy as np
from torch import nn, optim
from torch.nn import functional as F
from torch.autograd import Variable
from torch import autograd


class SimplePointNet(nn.Module):
  '''
  Simplified PointNet, without embedding transformer matrices.
  Akin to the method in Achlioptas et al, Learning Representations and
  Generative Models for 3D Point Clouds.

  E.g.
  s = SimplePointNet(100, 200, (25,50,100), (150,120))
  // Goes: 3 -> 25 -> 50 -> 100 -> 200 -> 150 -> 120 -> 100
  '''

  def __init__(self,
               latent_dimensionality : int,
               convolutional_output_dim : int,
               conv_layer_sizes,
               fc_layer_sizes,
               transformer_positions,
               end_in_batchnorm=False):

      super(SimplePointNet, self).__init__()
      self.LD = latent_dimensionality
      self.CD = convolutional_output_dim
      self.transformer_positions = transformer_positions
      self.num_transformers = len(self.transformer_positions)

      assert self.CD % 2 == 0, "Input Conv dim must be even"
    
      # Basic order #
      # B x N x 3 --Conv_layers--> B x C --Fc_layers--> B x L
      # We divide the output by two in the conv layers because we are using
      # both average and max pooling, which will be concatenated.
      self._conv_sizes = [3] + [k for k in conv_layer_sizes] + [self.CD] #//2
      self._fc_sizes = [self.CD] + [k for k in fc_layer_sizes]

      ### Convolutional Layers ###
      self.conv_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(self._conv_sizes[i], self._conv_sizes[i+1], 1),
                nn.BatchNorm1d(self._conv_sizes[i+1]),
                nn.ReLU()
            )
            for i in range(len(self._conv_sizes)-1)
      ])

      ### Transformers ###
      # These are run and applied to the input after the corresponding convolutional
      # layer is run. Note that they never change the feature size (or indeed the 
      # tensor shape in general).
      # E.g. if 0 is given in the positions, a 3x3 matrix set will be applied.
      self.transformers = nn.ModuleList([
          SimpleTransformer(self._conv_sizes[jj]) 
          for jj in self.transformer_positions
      ])

      ### Fully Connected Layers ###
      self.fc_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self._fc_sizes[i], self._fc_sizes[i+1]),
                nn.BatchNorm1d(self._fc_sizes[i+1]),
                nn.ReLU()
            )
            for i in range(len(self._fc_sizes)-1)] 
            +
            ([ nn.Linear(self._fc_sizes[-1], self.LD), nn.BatchNorm1d(self.LD) ]
             if end_in_batchnorm else
             [ nn.Linear(self._fc_sizes[-1], self.LD) ])
      )

  def move_eye(self, device):
      for t in self.transformers: t.move_eye(device)
  
  def forward(self, P):
      ''' 
      Input: B x N x 3 point clouds (non-permuted)
      Output: B x LD embedded shapes
      '''
      P = P.permute(0,2,1)
      assert P.shape[1] == 3, "Unexpected shape"
      # Now P is B x 3 x N
      for i, layer in enumerate(self.conv_layers): 
          if i in self.transformer_positions:
              T = self.transformers[ self.transformer_positions.index(i) ](P)
              P = layer( torch.bmm(T, P) )
          else:
              P = layer(P)
      # Pool over the number of points.
      # i.e. P: B x C_D x N --Pool--> B x C_D x 1
      # Then, P: B x C_D x 1 --> B x C_D (after squeeze)
      # Note: F.max_pool1d(input, kernel_size)
      P = F.max_pool1d(P, P.shape[2]).squeeze(2)
      #P = #torch.cat( (
      #      F.max_pool1d(P, P.shape[2]).squeeze(2),
#            F.avg_pool1d(P, P.shape[2]).squeeze(2) ),
      #      dim=1)
      for j, layer in enumerate(self.fc_layers):
          P = layer(P)
      return P


##############################################################################################


class SimpleTransformer(nn.Module):
    
    def __init__(self, 
                 input_dimensionality,
                 convolutional_dimensions=(64,128,512),
                 fc_dimensions=(512,256)):

        super(SimpleTransformer, self).__init__()
        
        # Setting network dimensions
        self.input_feature_len = input_dimensionality
        self.conv_dims = [self.input_feature_len] + [a for a in convolutional_dimensions]
        self.fc_dims = [f for f in fc_dimensions] #+ [self.input_feature_len**2]

        ### Convolutional Layers ###
        self.conv_layers = nn.ModuleList([
              nn.Sequential(
                  nn.Conv1d(self.conv_dims[i], self.conv_dims[i+1], 1),
                  nn.BatchNorm1d(self.conv_dims[i+1]),
                  nn.ReLU()
              )
              for i in range(len(self.conv_dims)-1)
        ])

        ### Fully Connected Layers ###
        self.fc_layers = nn.ModuleList([
              nn.Sequential(
                  nn.Linear(self.fc_dims[i], self.fc_dims[i+1]),
                  nn.ReLU()
              ) for i in range(len(self.fc_dims)-1) ]
            + [ nn.Linear(self.fc_dims[-1], self.input_feature_len**2) ]
        )
        
        ### Identity matrix added to the transformer at the end ###
        self.eye = torch.eye(self.input_feature_len)


    def forward(self, x):
        '''
        Input: B x F x N, e.g. F = 3 at the beginning
            i.e. expects a permuted point cloud batch
        Output: B x F x F set of transformation matrices
        '''
        SF = x.shape[1] # Size of the features per point
        #assert SF == self.input_feature_len, "Untenable feature len"

        # Unfortunately, I need to handle the current device here
        # because eye is local and I don't see a better way to 
        # do this. However, nowhere is a device setting kept.
        #if x.is_cuda: device = x.get_device()
        #else:         device = torch.device('cpu')

        # Convolutional layers
        for i, layer in enumerate(self.conv_layers): x = layer(x)
        # Max pooling
        x = F.max_pool1d(x, x.shape[2]).squeeze(2)
        # Fully connected layers
        for j, layer in enumerate(self.fc_layers): x = layer(x)
        # Fold into list of matrices
        #x = x.view(-1, SF, SF) + self.eye
        x = x.view(-1, SF, SF) + self.eye.to(x.device)
        #x += self.eye.to(device)
        return x

    def move_eye(self, device):
        self.eye = self.eye.to(device)


 #
