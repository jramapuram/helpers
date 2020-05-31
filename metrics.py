import numpy as np
import torch
import torch.nn.functional as F

from .utils import to_data, float_type, int_type, \
    zero_pad_smaller_cat, zeros_like
from pytorch_msssim import ms_ssim, ssim


def topk(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k.
       From: https://bit.ly/2Y1MOAq
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def softmax_correct(preds, targets, dim=-1):
    preds_max = to_data(preds).max(dim=dim)[1]  # get the index of the max log-probability
    assert targets.shape == preds_max.shape, \
        "target[{}] shape does not match preds[{}]".format(targets.shape, preds_max.shape)
    targ = to_data(targets)
    return preds_max.eq(targ).cpu().type(torch.FloatTensor)


def softmax_accuracy(preds, targets, size_average=True, dim=-1):
    reduction_fn = torch.mean if size_average is True else torch.sum
    return reduction_fn(softmax_correct(preds, targets, dim=dim))


def all_or_none_accuracy(preds, targets, size_average=True, dim=-1):
    preds_max = to_data(preds).max(dim=dim)[1]  # get the index of the max log-probability
    assert targets.shape == preds_max.shape, \
        "target[{}] shape does not match preds[{}]".format(targets.shape, preds_max.shape)
    targ = to_data(targets)
    reduction_fn = torch.mean if size_average is True else torch.sum
    return reduction_fn(preds_max.eq(targ).cpu().all(dim=dim).type(torch.float32))


def bce_accuracy(pred_logits, targets, size_average=True):
    cuda = pred_logits.is_cuda
    pred = torch.round(torch.sigmoid(to_data(pred_logits)))
    pred = pred.type(int_type(cuda))
    targets = targets.type(int_type(cuda))
    reduction_fn = torch.mean if size_average is True else torch.sum
    return reduction_fn(pred.eq(targets).cpu().type(torch.FloatTensor))


def calculate_mssim(minibatch, reconstr_image, size_average=True):
    """ compute the ms-sim between an image and its reconstruction

    :param minibatch: the input minibatch
    :param reconstr_image: the reconstructed image
    :returns: the msssim score
    :rtype: float

    """
    smallest_dim = min(minibatch.shape[-1], minibatch.shape[-2])
    if minibatch.dtype != reconstr_image.dtype:
        minibatch = minibatch.type(reconstr_image.dtype)

    if smallest_dim < 160:  # Limitation of ms-ssim library due to 4x downsample
        return 1 - ssim(X=minibatch, Y=reconstr_image, data_range=1,
                        size_average=size_average, nonnegative_ssim=True)

    return 1 - ms_ssim(X=minibatch, Y=reconstr_image, data_range=1, size_average=size_average)


def calculate_consistency(model, loader, reparam_type, vae_type, cuda=False):
    ''' sum (z_d(teacher)) == z_d(student) for all test samples '''
    consistency = 0.0

    if model.current_model > 0 and (reparam_type == 'mixture'
                                    or reparam_type == 'discrete'):
        model.eval()  # prevents data augmentation
        consistency, samples_seen = [], 0

        for img, _ in loader.test_loader:
            with torch.no_grad():
                img = img.cuda() if cuda else img

                output_map = model(img)
                if vae_type == 'parallel':
                    teacher_posterior = output_map['teacher']['params']['discrete']['logits']
                    student_posterior = output_map['student']['params']['discrete']['logits']
                elif vae_type == 'sequential':
                    if 'discrete' not in output_map['teacher']['params']['params_0']:
                        # we need the first reparam to be discrete to calculate
                        # the consistency metric
                        return consistency

                    teacher_posterior = output_map['teacher']['params']['params_0']['discrete']['logits']
                    student_posterior = output_map['student']['params']['params_0']['discrete']['logits']
                else:
                    raise Exception('unknown VAE consistency requested')

                teacher_posterior = F.softmax(teacher_posterior, dim=-1)
                student_posterior = F.softmax(student_posterior, dim=-1)
                teacher_posterior, student_posterior \
                    = zero_pad_smaller_cat(teacher_posterior, student_posterior)

                correct = to_data(teacher_posterior).max(1)[1] \
                    == to_data(student_posterior).max(1)[1]
                consistency.append(torch.mean(correct.type(float_type(cuda))).item())
                # print("teacher = ", teacher_posterior)
                # print("student = ", student_posterior)
                # print("consistency[-1]=", correct)
                samples_seen += img.size(0)

        consistency = np.mean(consistency)
        print("Consistency [#samples: {}]: {}\n".format(samples_seen,
                                                        consistency))
    return np.asarray([consistency])


def estimate_fisher(model, data_loader, batch_size, sample_size=10000, cuda=False):
    """Estimates the ELBO based FIM."""
    model.eval()  # lock BN / dropout, etc
    diag_fisher = {k: zeros_like(param) for (k, param) in model.named_parameters()}
    num_observed_samples = 0

    for x, _ in data_loader.train_loader:  # can't use torch.no_grad because we need grads.
        model.zero_grad()
        x = x.cuda() if cuda else x
        reconstr_x, params = model(x)
        loss = model.loss_function(reconstr_x, x, params)
        for i in range(x.size(0)):
            model.zero_grad()
            loss['loss'][i].backward(retain_graph=True)
            for k, v in model.named_parameters():
                diag_fisher[k] += v.grad.data ** 2  # / num_samples_in_dataset

            num_observed_samples += 1
            if num_observed_samples > sample_size:
                break

    for k in diag_fisher.keys():
        diag_fisher[k] /= num_observed_samples
        diag_fisher[k].requires_grad = False

    return diag_fisher
