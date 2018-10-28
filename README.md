# Pytorch Helpers

This repo can be cloned as a submodule into any project in order to provide:

    - helper functions
    - grapher module
    - metrics
    - layers
    - fid

## Helper functions

These functions include things like:

    - ones_like / zeros_like functions
    - directory creation / csv appender
    - expand / squeeze dims
    - zero padding
    - one_hot generataion
    - normalization
    - int_type / long_type / float_type for cuda vs. regular tensors

## Metrics

Include functions like:

    - compute softmax / bce accuracies
    - frechet distance calculations
    - compute EWC
    - compute FID

## Layers

Include layers such as :

    - dense with many sequential layers & bn
    - conv stack
    - dense / conv encoder + decoder stack
    - bw2rgb module
    - Identity, View layers, and more!


## FID:

To compute FID use `train_fid_model` from `fid.py`.
You can use a simple `conv` model or `inceptionv3`.
The FID batch size can smaller than or equal to the model you train.
For inceptionv3 you need small batch sizes unless you have a badass P100 or something.

After this you can use `calculate_fid` from metrics.


## Grapher

The grapher can plot to visdom or tensorboardX
Currently there exists only a grapher, the visdom grapher.
This helper utilizes a matplotlib style API for sending data to visdom.

```python
from helpers.grapher import Grapher

# for visdom:
grapher = Grapher('visdom', env='my_experiment',
                  server='http://localhost',
                  port=8097)

# for tensorboardX
grapher = Grapher('tensorboard', 'my_experiment')

# to add a scalar
grapher.add_scalar('my_scalar', value, epoch)
```
