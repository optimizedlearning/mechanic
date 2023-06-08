# Mechanic: black-box tuning of optimizers

Based on the paper: https://arxiv.org/abs/2306.00144

Mechanic aims to remove the need for tuning a learning rate scalar (i.e. the maximum learning rate in a schedule).
You can use it with any pytorch optimizer and schedule. Simply replace:
```
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
```
with:
```
optimizer = mechanize(torch.optim.SGD)(model.parameters(), lr=1.0)
```
That's it! The new optimizer should no longer require tuning the learning rate scale!
