# Resnet 1D and Variable Length Pooling for time series data

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

- https://github.com/pytorch/examples/tree/master/mnist
- ...
