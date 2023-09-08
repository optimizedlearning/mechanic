# Mechanic: black-box tuning of optimizers

[![PyPI - Version](https://img.shields.io/pypi/v/mechanic.svg)](https://pypi.org/project/mechanic)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/mechanic.svg)](https://pypi.org/project/mechanic)

-----

Based on the paper: https://arxiv.org/abs/2306.00144

Be aware that all experiments reported in the paper were run using the [JAX version of mechanic](https://github.com/google-deepmind/optax/blob/master/optax/_src/contrib/mechanic.py), which is available in [optax](https://optax.readthedocs.io/en/latest/) via `optax.contrib.mechanize`. 

Mechanic aims to remove the need for tuning a learning rate scalar (i.e. the maximum learning rate in a schedule).
You can use it with any pytorch optimizer and schedule. Simply replace:
```python
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
```
with:
```python
from mechanic_pytorch import mechanize
optimizer = mechanize(torch.optim.SGD)(model.parameters(), lr=1.0)
# you can set the lr to anything here, but excessivel small values may cause numerical precision issues.
```
That's it! The new optimizer should no longer require tuning the learning rate scale! That is, the optimizer should now be very robust to heavily mis-specified values of `lr`.

## Installation

```console
pip install mechanic-pytorch
```
Note that the package name is `mechanic-pytorch`, but you should `import mechanic_pytorch` (dash replaced with underscore).

## Options
It is possible to play with the configuration of mechanic, although this should be unecessary:
```python
optimizer = mechanize(torch.optim.SGD, s_decay=0.0, betas=(0.999,0.999999), store_delta=False)(model.parameters(), lr=0.01)
```
* The option `store_delta=False` is set to minimize memory usage. An minimum we currently keep one extra "slot" of memory (i.e. an extra copy of the weights). If you are ok keeping one more copy, you can set `store_delta=True`. This will make the first few iterations have a slightly more accurate update, and usually has negligible effect.
* The option `s_decay` is a bit like a weight-decay term that empirically is helpful for smaller datasets. We use a default of 0.01 in all our experiments. For larger datasets, smaller values (even 0.0) often worked as well.
* The option `betas` is a list of exponential weighting factors used internally in mechanic. They are NOT related to beta values found in Adam. In theory, it should be safe to provide a large list of possibilities here. The default settings of `(0.9,0.99,0.999,0.9999,0.99999,0.999999)` seem to work will in a range of tasks.
* `s_init` is the initial value for the mechanic learning rate. It should be an *underestimate* of the correct learning rate, and it can safely be set to a very small value (default 1e-8), although it cannot be set to zero. In particular, the  theoretical analysis of mechanic includes a log(1/s_init) term. This is very robust to small values, but will eventually blow up if you make `s_init` absurdly small.



## License

`mechanic` is distributed under the terms of the [Apache-2.0](https://spdx.org/licenses/Apache-2.0.html) license.