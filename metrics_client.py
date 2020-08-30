import rpyc
import numpy as np


def setter_function(lbda):
    """Uses a setter pattern to do an rpyc obtain on the scalar value and calls underlying lbda function."""

    def setter(metrics_map):
        """Tacks on _mean to ensure proper posting to grapher."""
        metrics_map = {k + '_mean': np.float32(rpyc.classic.obtain(v))
                       for k, v in metrics_map.items()}
        lbda(metrics_map)

    return setter


class MetricsClient(object):
    def __init__(self, host, port, normalize_imgs=True, force_cpu=False):
        """ A simple remote metrics client that talks to the metrics server and posts fid / prec / recall etc.

        :param host: the metrics server hostname or ip
        :param port: the metrics server port
        :param normalize_imgs: whether to normalize the images
        :param force_cpu: force calculations on CPU (not recommended)
        :returns: MetricsClient Object
        :rtype: object

        """
        self.cfg = {'allow_pickle': True, "sync_request_timeout": 30000}
        self.conn = rpyc.connect(host, port, config=self.cfg)
        self.bgsrv = rpyc.BgServingThread(self.conn)
        self.fid = self.conn.root.FID(normalize=normalize_imgs, force_cpu=force_cpu)

    def add_dataset(self, dataset_str, path):
        """ Helper to add a dataset to the metrics server

        :param dataset_str: name of the dataset
        :param path: where the dataset is stored on the server
        :returns: nothing
        :rtype: None

        """
        self.fid.add_dataset(dataset_str, path)

    def post_with_images(self, fake_images, real_images, lbda):
        """ Post fake + real images to the metrics server and set lbda to be the callback.

        :param fake_images: the fake numpy images with channels being the last dimension.
        :param real_images: the true numpy images with channels being the last dimension.
        :param lbda: the function that takes metrics dict as ONLY param, eg: lambda metrics_dict: print(metrics_dict).
        :returns: nothing, but executes the callback lbda asynchronously.
        :rtype: None

        """
        assert fake_images.shape[-1] == 3 or fake_images.shape[-1] == 1, "[fake] channel dim needs to be at dim=-1"
        assert real_images.shape[-1] == 3 or real_images.shape[-1] == 1, "[real] channel dim needs to be at dim=-1"
        self.fid.post_with_images(fake_images, real_images, setter_function(lbda))

    def post(self, fake_images, lbda, dataset_str):
        """ Post fake_images to the metrics server and set lbda to be the callback.

        :param fake_images: the fake numpy images with channels being the last dimension.
        :param lbda: the function that takes metrics dict as ONLY param, eg: lambda metrics_dict: print(metrics_dict).
        :param dataset_str: the name of the dataset to use
        :returns: nothing, but executes the callback lbda asynchronously.
        :rtype: None

        """
        assert fake_images.shape[-1] == 3 or fake_images.shape[-1] == 1, "channel dim needs to be at dim=-1"
        self.fid.post(fake_images, setter_function(lbda), dataset_str)
