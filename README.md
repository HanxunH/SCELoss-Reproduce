# SCELoss-PyTorch

Official Repo: https://github.com/YisenWang/symmetric_cross_entropy_for_noisy_labels \
Reproduce result for ICCV2019 paper ["Symmetric Cross Entropy for Robust Learning with Noisy Labels"](https://arxiv.org/abs/1908.06112)

## Update
In the tensorflow version [Official Repo](https://github.com/YisenWang/symmetric_cross_entropy_for_noisy_labels), the model uses l2 weight decay of 0.01 on model.fc1, which will gives a better results.
The code has been updated, now it should shows similar performance as in the paper.


## How To Run
##### Arguments
* --loss: 'SCE', 'CE'
* --nr: 0.0 to 1.0 specify the nosie rate.
* --dataset_type: 'cifar10' or 'cifar100'
* --alpha: alpha for SCE
* --beta: beta for SCE
* --seed: random seed
* --version: For experiment notes

Example for 0.4 Symmetric noise rate with SCE loss
```console
# CIFAR10
$ python3 -u train.py  --loss         SCE               \
	                     --dataset_type cifar10           \
                       --l2_reg       1e-2              \
                       --seed         123               \
                       --alpha        0.1               \
                       --beta         1.0               \
                       --version      SCE0.4_CIFAR10    \
                       --nr           0.4

# CIFAR100
$ python3 -u train.py  --lr           0.01              \
                       --loss         SCE               \
                       --dataset_type cifar100          \
                       --l2_reg       1e-2              \
                       --seed         123               \
                       --alpha        6.0               \
                       --beta         1.0               \
                       --version      SCE0.4_CIFAR100   \
                       --nr           0.4

```
## Results on CIFAR10
Result of best Epoch
| Loss  | 0.0 | 0.2  | 0.4 | 0.6 | 0.8 |
| ----- |:---:|:---:|:---:|:---:|:---:|
| CE    |92.68|84.70|72.77|54.14|31.23|
| SCE   |92.05|89.96|84.65|73.77|36.28|

## Results on CIFAR100
| Loss  | 0.0 | 0.2  | 0.4 | 0.6 | 0.8 |
| ----- |:---:|:---:|:---:|:---:|:---:|
| CE    |73.84|61.70|42.88|20.47|4.88|
| SCE   |73.57|62.31|46.50|24.00|12.51|
