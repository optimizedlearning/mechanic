# Mechanic: black-box tuning of optimizers

Based on the paper: https://arxiv.org/abs/2306.00144

Mechanic aims to remove the need for tuning a learning rate scalar. You can use
it with any pytorch optimizer and schedule. Simply replace:
```
optimizer = optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
```
with:
```
optimizer = mechanize(torch.optim.SGD)(model.parameters(), lr=0.01)
```
That's it!