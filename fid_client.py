import rpyc


class FIDClient(object):
    def __init__(self, host, port, normalize_imgs=True, force_cpu=False):
        """ A simple FID-client that talks to the FID server and posts fid calculation requests

        :param host: the FID-server hostname or ip
        :param port: the FID-server port
        :param normalize_imgs: whether to normalize the images
        :param force_cpu: force calculations on CPU (not recommended)
        :returns: FIDClient Object
        :rtype: object

        """
        self.cfg = {'allow_pickle': True, "sync_request_timeout": 180}
        self.conn = rpyc.connect(host, port, config=self.cfg)
        self.bgsrv = rpyc.BgServingThread(self.conn)
        self.fid = self.conn.root.FID(normalize=normalize_imgs, force_cpu=force_cpu)

    def add_dataset(self, dataset_str, path):
        """ Helper to add a dataset to the FID server

        :param dataset_str: name of the dataset
        :param path: where the dataset is stored on the server
        :returns: nothing
        :rtype: None

        """
        self.fid.add_dataset(dataset_str, path)

    def post_with_images(self, fake_images, real_images, lbda):
        """ Post fake + real images to the FID-server and set lbda to be the callback.

        :param fake_images: the fake numpy images with channels being the last dimension.
        :param real_images: the true numpy images with channels being the last dimension.
        :param lbda: the function that takes fid_score as ONLY param, eg: lambda x: print("fid is ", x).
        :returns: nothing, but executes the callback lbda asynchronously.
        :rtype: None

        """
        assert fake_images.shape[-1] == 3 or fake_images.shape[-1] == 1, "[fake] channel dim needs to be at dim=-1"
        assert real_images.shape[-1] == 3 or real_images.shape[-1] == 1, "[real] channel dim needs to be at dim=-1"
        self.fid.post_with_images(fake_images, real_images, lbda)

    def post(self, fake_images, lbda, dataset_str):
        """ Post fake_images to the FID-server and set lbda to be the callback.

        :param fake_images: the fake numpy images with channels being the last dimension.
        :param lbda: the function that takes fid_score as ONLY param, eg: lambda x: print("fid is ", x).
        :param dataset_str: the name of the dataset to use
        :returns: nothing, but executes the callback lbda asynchronously.
        :rtype: None

        """
        assert fake_images.shape[-1] == 3 or fake_images.shape[-1] == 1, "channel dim needs to be at dim=-1"
        self.fid.post(fake_images, lbda, dataset_str)
