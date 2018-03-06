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

Currently there exists only a grapher, the visdom grapher.
This helper utilizes a matplotlib style API for sending data to visdom.

```python
from helpers.grapher import Grapher

grapher = Grapher("unique_experiment_uuid", "http://localhost", port=8097)
grapher.register_single(  # format is [xarr, yarr]
    {'loss': [[epoch], [loss.data[0]]]},
    plot_type='line'
)
grapher.show()
```
