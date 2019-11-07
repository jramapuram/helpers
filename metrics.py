import numpy as np
import torch
import torch.nn.functional as F
import torch.distributions as D

from scipy import linalg
from torchvision import transforms
from torch.autograd import Variable

from .utils import to_data, float_type, int_type, \
    num_samples_in_loader, zero_pad_smaller_cat, zeros_like
from .msssim import MultiScaleSSIM


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


def calculate_mssim(reconstr_image, minibatch):
    """ compute the ms-sim between an image and its reconstruction

    :param reconstr_image: the reconstructed image
    :param minibatch: the input minibatch
    :returns: a score
    :rtype: float

    """
    xform = transforms.ToPILImage()
    reconstr_image_np = np.vstack([np.expand_dims(np.array(xform(ri)), 0) for ri in reconstr_image.detach().cpu()])
    minibatch_np = np.vstack([np.expand_dims(np.array(xform(mi)), 0) for mi in minibatch.detach().cpu()])
    reconstr_image_np = np.concatenate([np.expand_dims(reconstr_image_np, -1) for _ in range(3)], axis=-1) \
        if len(reconstr_image_np.shape) == 3 else reconstr_image_np
    minibatch_np = np.concatenate([np.expand_dims(minibatch_np, -1) for _ in range(3)], axis=-1) \
        if len(minibatch_np.shape) == 3 else minibatch_np
    return MultiScaleSSIM(reconstr_image_np, minibatch_np)


def calculate_consistency(model, loader, reparam_type, vae_type, cuda=False):
    ''' \sum z_d(teacher) == z_d(student) for all test samples '''
    consistency = 0.0

    if model.current_model > 0 and (reparam_type == 'mixture'
                                    or reparam_type == 'discrete'):
        model.eval() # prevents data augmentation
        consistency, samples_seen = [], 0

        for img, _ in loader.test_loader:
            with torch.no_grad():
                img = Variable(img).cuda() if cuda else Variable(img)

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

        num_test_samples = num_samples_in_loader(loader.test_loader)
        consistency = np.mean(consistency)
        print("Consistency [#samples: {}]: {}\n".format(samples_seen,
                                                        consistency))
    return np.asarray([consistency])


def estimate_fisher(model, data_loader, batch_size, sample_size=10000, cuda=False):
    model.eval() # lock BN / dropout, etc
    diag_fisher = {k: zeros_like(param) for (k, param) in model.named_parameters()}
    #num_samples_in_dataset = num_samples_in_loader(data_loader.train_loader)
    num_observed_samples = 0

    for x, _ in data_loader.train_loader:
        model.zero_grad()
        x = Variable(x).cuda() if cuda else Variable(x)
        reconstr_x, params = model(x)
        loss = model.loss_function(reconstr_x, x, params)
        for i in range(x.size(0)):
            model.zero_grad()
            loss['loss'][i].backward(retain_graph=True)
            for k, v in model.named_parameters():
                diag_fisher[k] += v.grad.data ** 2 #/ num_samples_in_dataset

            num_observed_samples += 1
            if num_observed_samples > sample_size:
                break

    for k in diag_fisher.keys():
        diag_fisher[k] /= num_observed_samples
        diag_fisher[k].requires_grad = False

    return diag_fisher
