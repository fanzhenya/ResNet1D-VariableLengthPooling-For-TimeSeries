# Resnet 1D and Variable Length Pooling for Speech Analysis

This repo is the code I used during the Kaggle Contest https://www.kaggle.com/c/11785-hw2pt2/,
which is a class project for CMU 11-785 course.

The reason for sharing is that I found very resource online for the below two topics.
It will be helpful for others who has similar problem.

## Pooling from variable length of activations
This is useful for dealing features of various length in time dimension.
E.g., phonemes in speech data.

```
Softmax             o1        o2        o3
                    |         |         |

VarLenAvgPooling    p1        p2        p3
                   / \     /     \     / \
                  |   |   |       |   |   |

Activation        a1  a2  a3  a4  a5  a6  a7
                  |   |   |   |   |   |   |

Conv1D            o   o   o   o   o   o   o
                 /|\ /|\ /|\ /|\ /|\ /|\ /|\

Activation        a1  a2  a3  a4  a5  a6  a7
                  |   |   |   |   |   |   |

Conv1D            o   o   o   o   o   o   o
                 /|\ /|\ /|\ /|\ /|\ /|\ /|\

Time             --------------------------->

```

## ResNet for time series data

Vanilla ResNet uses Conv2D for image data. However this architecture may be useful for deep Conv1D networks as well. 

I tried two approaches in my code:

- use rectangular filters (different H, W) directly in ResNet2D
- shift to Conv1D entirely

It depends on your specific problem to answer which approach is better.

# Credits

Some code are copied from pytorch official examples and CMU classes. Credits to the original authors.

- https://github.com/cmudeeplearning11785/deep-learning-tutorials/blob/master/recitation-4/pytorch-mnist-cnn-example.py
- https://github.com/pytorch/examples/tree/master/mnist
- ...
