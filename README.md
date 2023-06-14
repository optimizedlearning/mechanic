# Mechanic: black-box tuning of optimizers

Based on the paper: https://arxiv.org/abs/2306.00144

Mechanic aims to remove the need for tuning a learning rate scalar (i.e. the maximum learning rate in a schedule).
You can use it with any pytorch optimizer and schedule. Simply replace:
```python
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
```
with:
```python
optimizer = mechanize(torch.optim.SGD)(model.parameters(), lr=0.01)
```
That's it! The new optimizer should no longer require tuning the learning rate scale!

## Options
It is possible to play with the configuration of Mechanic, although this should be unecessary:
```python
optimizer = mechanize(torch.optim.SGD, s_decay=0.0, betas=(0.999,0.999999), store_delta=False)(model.parameters(), lr=0.01)
```
* The option `store_delta=False` is set to minimize memory usage. An minimum we currently keep one extra "slot" of memory (i.e. an extra copy of the weights). If you are ok keeping one more copy, you can set `store_delta=True`. This will be a slightly more accurate update, but should only have an effect in the first few iterations, and should be negligible.
* The option `s_decay` is a bit like a weight-decay term that empirically is helpful for smaller datasets. We use a default of 0.01 in all our experiments. For larger datasets, smaller values (even 0.0) often worked as well.
* The option `betas` is a list of exponential weighting factors used internally in mechanic. They are NOT related to beta values found in Adam. In theory, it should be safe to provide a large list of possibilities here, and mechanic will automatically find the right one. Including some values very close to one is advised.
* `s_init` is the initial value for the mechanic learning rate. It should be an *underestimate* of the correct learning rate, and it can safely be set to a very small value (default 1e-8), although it cannot be set to zero. In particular, the  theoretical analysis of mechanic includes a log(1/s_init) term. This is very robust to small values, but will eventually blow up if you make s_init absurdly small.

