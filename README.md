# Unpaired image-to-image translation
> A fastai/PyTorch package for unpaired image-to-image translation currently with CycleGAN implementation.


This is a package for training and testing unpaired image-to-image translation models. It currently only includes the [CycleGAN model](https://junyanz.github.io/CycleGAN/), but other models will be implemented in the future. 

This package uses [fastai](https://github.com/fastai/fastai) to accelerate deep learning experimentation. Additionally, [nbdev](https://github.com/fastai/nbdev) was used to develop the package and produce documentation based on a series of notebooks.

## Install

To install, use `pip`:

`pip install git+https://github.com/tmabraham/UPIT.git`

The package uses torch 1.6.0, torchvision 0.7.0, fastai 2.0.0 (and its dependencies), and tqdm. It also requires nbdev 0.2.26 if you would like to add features to the package. Finally, for creating a web app model interface, gradio 1.1.6 is used.

## How to use

Training a CycleGAN model is easy with UPIT! Given the paths of the images from the two domains `trainA_path` and `trainB_path`, you can do the following:

```python
#cuda
dls = get_dls(trainA_path, trainB_path)
cycle_gan = CycleGAN(3,3,64)
learn = cycle_learner(dls, cycle_gan,opt_func=partial(Adam,mom=0.5,sqr_mom=0.999))
learn.fit_flat_lin(100,100,2e-4)
```
