This repository contains different techniques for solving unlearning problem. Effectivness and limitations of various methods are discussed in the [overleaf document.](https://www.overleaf.com/project/64ab1d0dfb84a676620a7ebe)

```python
python benchmark.py -d "cuda:2" -exp unlearning 
```
How to set up the file
1. git clone
2. conda env create -f environment.yml



## Ideas

### Modelling as a Overfitting Problem

The problem of unlearning can be thought as overfitting, generally faced by the Large models. To validate this hypothesis, first experiment to be performed is how the loss histogram changes as we increase the capcity of the model?
Results:

One parameter to wary is depth or number of layers in a ResNet model.



Though experiment, Suppose you have 2 class data from a circle clearly seperated, for example (-5,5) and (5,5) with radius 2. How does the model capacity affect the decision boundary? What is the contribution of each sample in the decision boundary, We can measure this by amount of change in model parameters, if we remove one point, keeping the seed fixed.


So we need to solve for the points which alter the decision boundary by a lot.



Key observations:
    Small models have less capacity so they dont suffer from the problem of 

### Adverserial Solution
Adverserial way of solving


Questions?
How to simulate data in 2D of a similar data distribution as CIFAR.


## Proposed solutions

### 

Now put more constraints, for example you dont have the retainset and only we need to remove samples on the fly, i.e streaming set of forget samples? 


minimise the entropy and the loss of the forget sample, This technique is useful only when the 
does variance of the model impact the loss manifold in train and test



GAN based solution (val and test and discriminator needs to be fooled. In case we dont want to use the retain set we can use a proxy of uniform loss on the test loss assuming the loss manifold is )
Hypernetworks based solution




Metric to measure unlearning
Maybe making a decision boundary based on just the loss charecterstics doesnot give more accuracy, what other charecterstics  we need to observe.
Loss, predictive manifold (probability manifold). 

Steps used to setup enivironment.yaml file
> conda env export > environment.yml

How to start unlearning from scratch

If you are using password based ssh connection you can access remote connection via vscode 
https://docs.icer.msu.edu/SSH_Key-Based_Authentication/

Setting up git
Add the same id_rsa.pub generated in the previous process to git.

> git clone https://github.com/sachit3022/unlearning.git

Installing anaconda
> wget https://repo.anaconda.com/archive/Anaconda3-2023.09-0-Linux-x86_64.sh
> bash Anaconda3-2023.09-0-Linux-x86_64.sh

Activating unlearning environment

> conda env create -f environment.yml
> conda activate unl

How to download CelebA dataset, As it it hosted on gdrive, we need to download manually due to the limitations of the google drive.
> https://drive.google.com/drive/folders/0B7EVK8r0v71pWEZsZE9oNnFzTm8?resourcekey=0-5BR16BdXnb8hVj6CNHKzLg
> Download locally and upload to the server.
uzip the zipped files
7zip e .....




