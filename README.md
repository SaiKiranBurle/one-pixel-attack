# One Pixel attack

Recently there has been a lot of interest in fooling Neural Networks that were trained to classify on a particular dataset.
Researchers have produced various methods that would perturb the natural image in way that the perturbation is 
imperceivable to the human eye, but disastrous to the correctness of the Neural Network in classifying the image.

This is an implementation of a recent paper which takes it to the extreme. How simple is it to fool a Neural network 
by changing just **one** pixel? It turns out, the current state-of-the-art Neural Networks are substantially vulnerable 
to this attack. There is a caveat. The adversary, who is trying to break the Neural Network should be given access to 
the black-box classification model for a fairly large number of experiments. In addition, the adversary needs to know 
the output probability distribution over the set of classes. These conditions are much weaker compared to the previous 
attacks where the adversary also has access to the gradients in the Neural Network.

The following project is a Keras reimplementation of ["One pixel attack for fooling deep neural networks"](https://arxiv.org/abs/1710.08864)

## Example successful attacks
### Targeted attack

Given an input image and an image classification model, the aim of a targeted attack is to maximize the probability 
label of the target class.

| Original Deer image        |  Perturbed Deer image |
| :---: | :---: |
| <img src="https://github.com/SaiKiranBurle/one-pixel-attack/blob/master/results/targeted/original_deer.jpg" width="100">  |  <img src="https://github.com/SaiKiranBurle/one-pixel-attack/blob/master/results/targeted/perturbed_deer.jpg" width="100"> |
|  Deer: 99.4%    |     Cat: 52.49%     |


### Non-targeted attack

Given an input image and an image classification model, the aim of a non-targeted attack is to minimize the probability 
label of the true class.

| Original dog image        |  Perturbed dog image |
| :---: | :---: |
| <img src="https://github.com/SaiKiranBurle/one-pixel-attack/blob/master/results/non-targeted/puppy_original.jpg" width="100"> | <img src="https://github.com/SaiKiranBurle/one-pixel-attack/blob/master/results/non-targeted/puppy_perturbed.jpg" width="100"> |
|  Dog: 94.8%    |     Bird: 90.6%     |

## Usage
### Targeted attack
```bash
python targeted.py --config config.yaml --input images/deer.jpg --target cat
```

### Non-targeted attack
```bash
python non_targeted.py --config config.yaml --input images/puppy.jpg
```

## Conclusions

* In my experiments, I found out that it is much easier to a fool a CIFAR-10 classifier than an ImageNet classifier. This 
is noted by the authors in the original paper as well.
* Although the success rate is quite low, this experiment was a great learning experience and demonstrates the fragile nature 
of various Deep Learning based image classifiers.
* Overall, I really liked the paper and enjoyed implementing for its simplicity and effectiveness conveying the point.

