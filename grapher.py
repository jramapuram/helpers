from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import gc
import six
import asyncio

from functools import partial

from tensorboardX.writer import SummaryWriter
from .visdom_writer import VisdomWriter


# Supports both TensorBoard and Visdom (no embedding or graph visualization with Visdom)
vis_formats = {'tensorboard': SummaryWriter, 'visdom': VisdomWriter}


class Grapher:
    def __init__(self, *args, **kwargs):
        self.subscribers = {}
        self.register(*args, **kwargs)

    def register(self, *args, **kwargs):
        # Sets tensorboard as the default visualization format if not specified
        formats = ['tensorboard'] if not args else args
        for format in formats:
            if self.subscribers.get(format) is None and format in vis_formats.keys():
                self.subscribers[format] = vis_formats[format](**kwargs)

    def unregister(self, *args):
        for format in args:
            self.subscribers[format].close()
            del self.subscribers[format]
            gc.collect()

    def __getattr__(self, attr):
        for _, subscriber in six.iteritems(self.subscribers):
            def wrapper(*args, **kwargs):
                for subscriber_name, subscriber in six.iteritems(self.subscribers):
                    if hasattr(subscriber, attr):
                        # Don't use async for specific fns or if we are using tensorboard
                        non_async_fns = ['pickle_data', 'close', 'reconnect_and_replay_log', '__init__', '_connect']
                        if attr not in non_async_fns and 'tensorboard' not in subscriber_name:
                            fn = partial(getattr(subscriber, attr), *args, **kwargs)
                            asyncio.get_event_loop().run_in_executor(None, fn)
                        else:
                            getattr(subscriber, attr)(*args, **kwargs)
                    elif 'sync' in attr and hasattr(subscriber, attr.replace('sync_', '')):
                        # Special logic if we want to force sync functions. eg: grapher.sync_add_scalar()
                        fn_name = attr.replace('sync_', '')
                        getattr(subscriber, fn_name)(*args, **kwargs)

            return wrapper
        raise AttributeError

    # Handle writer management (open/close) for the user
    def __del__(self):
        for _, subscriber in six.iteritems(self.subscribers):
            subscriber.close()
