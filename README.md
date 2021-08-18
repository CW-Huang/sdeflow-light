# sdeflow-light

This is a minimalist codebase for training score-based diffusion models (supporting MNIST and CIFAR-10) used in the following paper

> "A Variational Perspective on Diffusion-Based Generative Models and Score Matching"
by Chin-Wei Huang, Jae Hyun Lim and Aaron Courville
[[arXiv](https://arxiv.org/abs/2106.02808)]

Also see the [concurrent work](https://arxiv.org/abs/2101.09258) by Yang Song & Conor Durkan where they used the same idea to obtain state-of-the-art likelihood estimates. 


## Experiments on Swissroll
Here's a Colab notebook which contains an example for training a model on the Swissroll dataset. 

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1hzk-V_yby0KiNCKLBnu7WvXUhZg4-RKg)  

In this notebook, you'll see how to train the model using score matching loss, how to evaluate the ELBO of the plug-in reverse SDE, and how to sample from it. It also includes a snippet to sample from a family of plug-in reverse SDEs (parameterized by λ) mentioned in Appendix C of [the paper](https://arxiv.org/abs/2011.13456). 

Below are the trajectories of λ=0 (the reverse SDE used in [Song et al.](https://arxiv.org/abs/2011.13456)) and λ=1 (equivalent ODE) when we plug in the learned score / drift function. This corresponds to Figure 5 of [the paper](https://arxiv.org/abs/2011.13456). 
<img src="/assets/sdeflow_equivalent_sdes_lmbd0.png" alt="drawing" width="1000"/> 
<img src="/assets/sdeflow_equivalent_sdes_lmbd1.png" alt="drawing" width="1000"/>



## Experiments on MNIST and CIFAR-10
This repository contains one main training loop (`train_img.py`). The model is trained to minimize the denoising score matching loss by calling the `.dsm(x)` loss function, and evaluated using the following ELBO, by calling `.elbo_random_t_slice(x)`

![score-elbo](/assets/score-elbo.png)

where the divergence (sum of the diagonal entries of the Jacobian) is estimated using the Hutchinson trace estimator. 

It's a minimalist codebase in the sense that we do not use fancy optimizer (we only use Adam with the default setup) or learning rate scheduling. 
We use the modified U-net architecture from [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) by Jonathan Ho. 

A key difference from [Song et al.](https://arxiv.org/abs/2011.13456) is that instead of parameterizing the score function `s`, here we parameterize the drift term `a` (where they are related by `a=gs` and `g` is the diffusion coefficient). That is, `a` is the U-net. 

**Parameterization**: Our original generative & inference SDEs are
* dX = mu dt + sigma dBt
* dY = (-mu + sigma\*a) ds + sigma dBs

We reparameterize it as
* dX = (ga - f) dt + g dBt
* dY = f ds + g dBs

by letting `mu = ga - f`, and `sigma = g`. (since f and g are fixed, we only have one degree of freedom, which is `a`). 
Alternatively, one can parameterize `s` (e.g. using the U-net), and just let `a=gs`. 

### How it works
Here's an example command line for running an experiment

```
python train_img.py --dataroot=[DATAROOT] --saveroot=[SAVEROOT] --expname=[EXPNAME] \
    --dataset=cifar --print_every=2000 --sample_every=2000 --checkpoint_every=2000 --num_steps=1000 \
    --batch_size=128 --lr=0.0001 --num_iterations=100000 --real=True --debias=False
```

Setting `--debias` to be False uses uniform sampling for the time variable, whereas setting it to be True uses a non-uniform sampling strategy to debias the gradient estimate described in the paper. Below are the bits-per-dim and the corresponding standard error of the test set recorded during training (<span style="color:orange;">orange</span> for `--debias=True` and <span style="color:blue;">blue</span> for `--debias=False`).


<img src="/assets/bpd.svg" alt="drawing" width="300"/> <img src="/assets/bpd_std_err.svg" alt="drawing" width="300"/>

Here are some samples (debiased on the right)

<img src="/assets/uniform.png" alt="drawing" width="200"/> <img src="/assets/non-uniform.png" alt="drawing" width="200"/>


It takes about 14 hrs to finish 100k iterations on a V100 GPU. 
