# LIMP
### Learning Latent Shape Representations with Metric Preservation Priors

This repository contains the official PyTorch implementation of:

*LIMP: Learning Latent Shape Representations with Metric Preservation Priors.*  
by Luca Cosmo, Antonio Norelli, Oshri Halimi, Ron Kimmel and Emanuele Rodolà.  
Oral at ECCV 2020.  
https://arxiv.org/abs/2003.12283

![gif link](https://media.giphy.com/media/6239ksApKG10JzxYOP/giphy.gif)

**Abstract**: In this paper, we advocate the adoption of metric preservation as a powerful prior for learning latent representations of deformable 3D shapes. Key to our construction is the introduction of a geometric distortion criterion, defined directly on the decoded shapes, translating the preservation of the metric on the decoding to the formation of linear paths in the underlying latent space. Our rationale lies in the observation that training samples alone are often insufficient to endow generative models with high fidelity, motivating the need for large training datasets. In contrast, metric preservation provides a rigorous way to control the amount of geometric distortion incurring in the construction of the latent space, leading in turn to synthetic samples of higher quality. We further demonstrate, for the first time, the adoption of differentiable intrinsic distances in the backpropagation of a geodesic loss. Our geometric priors are particularly relevant in the presence of scarce training data, where learning any meaningful latent structure can be especially challenging. The effectiveness and potential of our generative model is showcased in applications of style transfer, content generation, and shape completion.

[2 minutes trailer video](https://youtu.be/NPE_uey-dXo)  
[10 minutes oral presentation video](https://youtu.be/P4uxICQ3QXI)

## Requirements

* tqdm==4.41.1
* numpy==1.18.5
* torch==1.6.0+cu101
* matplotlib==3.2.2
* scipy==1.4.1
* plotly==4.4.1
* requests==2.23.0
* dill==0.3.2
* plyfile==0.7.2
* torch_geometric==1.6.1
* torch_scatter==2.0.5

## Cite
If you make use of this code in your own work, please cite our paper:
```
@article{cosmo2020limp,
  title={{LIMP: Learning Latent Shape Representations with Metric Preservation Priors}},
  author={Cosmo, Luca and Norelli, Antonio and Halimi, Oshri and Kimmel, Ron and Rodol{\`a}, Emanuele},
  journal={European Conference on Computer Vision (ECCV)},
  year={2020}
}
```

