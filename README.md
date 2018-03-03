# Pytorch Helpers

This repo can be cloned as a submodule into any project in order to provide:

    - helper functions
    - grapher module

## Helper functions

These functions include things like:

    - compute softmax / bce accuracies
    - frechet distance calculations
    - ones_like / zeros_like functions
    - directory creation / csv appender
    - expand / squeeze dims
    - zero padding
    - one_hot generataion
    - normalization
    - int_type / long_type / float_type for cuda vs. regular tensors

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
```
