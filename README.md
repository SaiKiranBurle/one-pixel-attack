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
