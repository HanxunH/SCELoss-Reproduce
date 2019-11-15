# SCELoss-Reproduce

Official Repo: https://github.com/YisenWang/symmetric_cross_entropy_for_noisy_labels \
Reproduce result for ICCV2019 paper ["Symmetric Cross Entropy for Robust Learning with Noisy Labels"](https://arxiv.org/abs/1908.06112)

## How To Run
##### Arguments
* --loss: 'SCE', 'CE'
* --nr: 0.0 to 1.0 specify the nosie rate.
* --train_cifar100: if train on CIFAR100
* --epoch

Example for 0.4 Symmetric noise rate with SCE loss
```console
$ python train.py --nr 0.4 --loss SCE --epoch 120
```
## Reporduced Results on CIFAR10
| Loss  | 0.0 | 0.2  | 0.4 | 0.6 | 0.8 |
| ----- |:---:|:---:|:---:|:---:|:---:|
| CE    |92.81|88.84|85.53|79.77|52.66|
| SCE   |92.86|90.87|87.92|82.75|56.26|
