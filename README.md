Unpaired image-to-image translation
================

<!-- WARNING: THIS FILE WAS AUTOGENERATED! DO NOT EDIT! -->

This is a package for training and testing unpaired image-to-image
translation models. It currently only includes the
[CycleGAN](https://junyanz.github.io/CycleGAN/),
[DualGAN](https://arxiv.org/abs/1704.02510), and
[GANILLA](https://arxiv.org/abs/2002.05638) models, but other models
will be implemented in the future.

This package uses [fastai](https://github.com/fastai/fastai) to
accelerate deep learning experimentation. Additionally,
[nbdev](https://github.com/fastai/nbdev) was used to develop the package
and produce documentation based on a series of notebooks.


[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7889404.svg)](https://doi.org/10.5281/zenodo.7889404)

## Install

To install, use `pip`:

`pip install git+https://github.com/tmabraham/UPIT.git`

The package uses torch 1.7.1, torchvision 0.8.2, and fastai 2.3.0 (and
its dependencies). It also requires nbdev 1.1.13 if you would like to
add features to the package. Finally, for creating a web app model
interface, gradio 1.1.6 is used.

## How to use

Training a CycleGAN model is easy with UPIT! Given the paths of the
images from the two domains `trainA_path` and `trainB_path`, you can do
the following:

``` python
from upit.data.unpaired import *
from upit.models.cyclegan import *
from upit.train.cyclegan import *
```

``` python
dls = get_dls(trainA_path, trainB_path)
cycle_gan = CycleGAN(3,3,64)
learn = cycle_learner(dls, cycle_gan,opt_func=partial(Adam,mom=0.5,sqr_mom=0.999))
learn.fit_flat_lin(100,100,2e-4)
```

The GANILLA model is only a different generator model architecture
(that’s meant to strike a better balance between style and content), so
the same
[`cycle_learner`](https://tmabraham.github.io/UPIT/train.cyclegan.html#cycle_learner)
class can be used.

``` python
from upit.models.ganilla import *
```

``` python
ganilla = GANILLA(3,3,64)
learn = cycle_learner(dls, ganilla,opt_func=partial(Adam,mom=0.5,sqr_mom=0.999))
learn.fit_flat_lin(100,100,2e-4)
```

Finally, we provide separate functions/classes for
[`DualGAN`](https://tmabraham.github.io/UPIT/models.dualgan.html#dualgan)
model and training:

``` python
from upit.models.dualgan import *
from upit.train.dualgan import *
```

``` python
dual_gan = DualGAN(3,64,3)
learn = dual_learner(dls, dual_gan, opt_func=RMSProp)
learn.fit_flat_lin(100,100,2e-4)
```

Additionally, we provide metrics for quantitative evaluation of the
models, as well as experiment tracking with Weights and Biases. Check
the [documentation](https://tmabraham.github.io/UPIT) for more
information!

## Citing UPIT

If you use UPIT in your research please use the following BibTeX entry:
```
@software{Abraham_UPIT_-_A,
    author = {Abraham, Tanishq Mathew},
    doi = {10.5281/zenodo.7889405},
    title = {{UPIT - A fastai/PyTorch package for unpaired image-to-image translation.}},
    url = {https://github.com/tmabraham/UPIT},
    version = {0.2.3}
}
```
