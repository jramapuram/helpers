import os
import wandb

from tensorboardX.utils import figure_to_image
from tensorboardX.x2num import make_np

from .utils import hash_to_size


class WandBWriter:
    """Simple adaptor for weights & biases logger."""

    def __init__(self, env, server, config, model, port=8080, log_folder=None):
        self.env = env
        self.server = server
        self.port = port
        server_format_str = 'http://{}:{}'
        os.environ['WANDB_BASE_URL'] = server_format_str.format(
            server.replace('http://', ''), port)

        # Where to save the logs
        self.log_folder = os.path.expanduser(os.path.join(log_folder, env))
        if self.log_folder is not None and not os.path.isdir(self.log_folder):
            os.makedirs(self.log_folder)

        # create the wandb object
        self.server = wandb.init(
            # name=env, id=env,
            # id=env,
            # id=hash_to_size(env, size=64),

            name=hash_to_size(env, size=16),
            resume=True,
            config=config,
            dir=self.log_folder,
        )
        wandb.watch(model)

    def add_scalar(self, tag, scalar_value, global_step=None):
        """Add scalar data to Visdom. Plots the values in a plot titled
           {main_tag}-{tag}.

        Args:
            tag (string): Data identifier
            scalar_value (float or string/blobname): Value to save
            global_step (int): Global step value to record
        """
        wandb.log({tag: scalar_value}, step=global_step, commit=False)

    def add_scalars(self, tag_scalar_dict, global_step=None):
        """Adds many scalar data to summary.

        Note that this function also keeps logged scalars in memory. In extreme case it explodes your RAM.

        Args:
            tag_scalar_dict (dict): Key-value pair storing the tag and corresponding values
            global_step (int): Global step value to record

        Examples::

            writer.add_scalars({'xsinx':i*np.sin(i/r),
                                'xcosx':i*np.cos(i/r),
                                'arctanx': numsteps*np.arctan(i/r)}, i)
            This function adds three plots:
                'xsinx',
                'xcosx',
                'arctanx'
            with the corresponding values.
        """
        wandb.log(tag_scalar_dict, step=global_step, commit=False)

    def add_histogram(self, tag, values, global_step=None, bins='tensorflow'):
        """Add histogram to summary.

        Args:
            tag (string): Data identifier
            values (torch.Tensor, numpy.array, or string/blobname): Values to build histogram
            global_step (int): Global step value to record
            bins (string): one of {'tensorflow', 'auto', 'fd', ...}, this determines how the bins are made. You can find
              other options in: https://docs.scipy.org/doc/numpy/reference/generated/numpy.histogram.html
        """
        wandb.log({tag: wandb.Histogram(values)}, step=global_step, commit=False)

    def add_image(self, tag, img_tensor, global_step=None, caption=None):
        """Add image data to summary.

        Note that this requires the ``pillow`` package.

        Args:
            tag (string): Data identifier
            img_tensor (torch.Tensor, numpy.array, or string/blobname): Image data
            global_step (int): Global step value to record
        Shape:
            img_tensor: :math:`(C, H, W)`. Use ``torchvision.utils.make_grid()`` to prepare it is a good idea.
            C = colors (can be 1 - grayscale, 3 - RGB, 4 - RGBA)
        """
        # img_tensor = make_np(img_tensor).transpose((2, 1, 0))
        # print('img_tensor = ', img_tensor.shape, " | dtype = ", img_tensor.dtype, type(img_tensor))
        wandb.log({tag: wandb.Image(img_tensor, caption=caption)}, step=global_step, commit=False)

    def add_figure(self, tag, figure, global_step=None, close=True):
        """Render matplotlib figure into an image and add it to summary.

        Note that this requires the ``matplotlib`` package.

        Args:
            tag (string): Data identifier
            figure (matplotlib.pyplot.figure) or list of figures: figure or a list of figures
            global_step (int): Global step value to record
            close (bool): Flag to automatically close the figure
        """
        self.add_image(tag, figure_to_image(figure, close), global_step)

    def add_video(self, tag, vid_tensor, global_step=None, fps=4):
        """Add video data to summary.

        Note that this requires the ``moviepy`` package.

        Args:
            tag (string): Data identifier
            vid_tensor (torch.Tensor): Video data
            global_step (int): Global step value to record
            fps (float or int): Frames per second
        Shape:
            vid_tensor: :math:`(B, C, T, H, W)`. (if following tensorboardX format)
            vid_tensor: :math:`(T, H, W, C)`. (if following visdom format)
            B = batches, C = colors (1, 3, or 4), T = time frames, H = height, W = width
        """
        wandb.log({tag: wandb.Video(vid_tensor, fps=4, format="gif")}, step=global_step, commit=False)

    def add_audio(self, tag, snd_tensor, global_step=None, sample_rate=44100, caption=None):
        """Add audio data to summary.

        Args:
            tag (string): Data identifier
            snd_tensor (torch.Tensor, numpy.array, or string/blobname): Sound data
            global_step (int): Global step value to record
            sample_rate (int): sample rate in Hz

        Shape:
            snd_tensor: :math:`(1, L)`. The values should lie between [-1, 1].
        """
        wandb.log({tag: [wandb.Audio(snd_tensor, caption=caption, sample_rate=sample_rate/1000.)]},
                  step=global_step, commit=False)

    def add_text(self, tag, text_string, global_step=None, append=False):
        """Add text data to summary.

        Args:
            tag (string): Data identifier
            text_string (string): String to save
            global_step (int): Global step value to record
        Examples::
            writer.add_text('lstm', 'This is an lstm', 0)
            writer.add_text('rnn', 'This is an rnn', 10)
        """
        # TODO(jramapuram): we don't really need this since kwargs are cached
        wandb.run.summary[tag] = text_string

    def add_pr_curve(self, tag, labels, predictions, global_step=None, num_thresholds=127, weights=None):
        """Adds precision recall curve.

        Args:
            tag (string): Data identifier
            labels (torch.Tensor, numpy.array, or string/blobname): Ground truth data. Binary label for each element.
            predictions (torch.Tensor, numpy.array, or string/blobname):
            The probability that an element be classified as true. Value should in [0, 1]
            global_step (int): Global step value to record
            num_thresholds (int): Number of thresholds used to draw the curve.

        """
        labels, predictions = make_np(labels), make_np(predictions)
        wandb.log({tag: wandb.plots.precision_recall(y_true=labels, y_probas=predictions)},
                  step=global_step, commit=False)

    def save(self):
        # TODO(jramapuram): do we even need this?
        wandb.log({})

    def close(self):
        pass
