import numpy as np
import torch
import torch.nn.functional as F
import torch.distributions as D

from torch.autograd import Variable

from models.layers import Submodel
from helpers.utils import to_data, float_type, \
    num_samples_in_loader, zero_pad_smaller_cat


def softmax_correct(preds, targets):
    pred = to_data(preds).max(1)[1] # get the index of the max log-probability
    targ = to_data(targets)
    return pred.eq(targ).cpu().type(torch.FloatTensor)


def softmax_accuracy(preds, targets, size_average=True):
    reduction_fn = torch.mean if size_average is True else torch.sum
    return reduction_fn(softmax_correct(preds, targets))


def bce_accuracy(pred_logits, targets, size_average=True):
    cuda = is_cuda(pred_logits)
    pred = torch.round(F.sigmoid(to_data(pred_logits)))
    pred = pred.type(int_type(cuda))
    reduction_fn = torch.mean if size_average is True else torch.sum
    return reduction_fn(pred.data.eq(to_data(targets)).cpu().type(torch.FloatTensor), -1)


def frechet_gauss_gauss_np(synthetic_features, test_features):
    # calculate the statistics required for frechet distance
    # https://github.com/bioinf-jku/TTUR/blob/master/fid.py
    mu_synthetic = np.mean(synthetic_features, axis=0)
    sigma_synthetic = np.cov(synthetic_features, rowvar=False)
    mu_test = np.mean(test_features, axis=0)
    sigma_test = np.cov(test_features, rowvar=False)

    m = np.square(mu_synthetic - mu_test).sum()
    s = sp.linalg.sqrtm(np.dot(sigma_synthetic, sigma_test))
    dist = m + np.trace(sigma_synthetic + sigma_synthetic - 2*s)
    if np.isnan(dist):
        raise Exception("nan occured in FID calculation.")

    return dist


def frechet_gauss_gauss(dist_a, dist_b):
    ''' d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)). '''
    m = torch.pow(dist_a.loc - dist_b.loc, 2).sum()
    s = torch.sqrt(dist_a.scale * dist_b.scale)
    return torch.mean(m + dist_a.scale + dist_b.scale - 2*s)


def calculate_fid(fid_model, model, loader, grapher, batch_size, cuda=False):
    # evaluate and cache away the FID score
    fid = np.inf
    fid = calculate_fid_from_generated_images(fid_model, model, loader, batch_size, cuda=cuda)
    grapher.vis.text(str(fid), opts=dict(title="FID"))
    return fid

def calculate_fid_from_generated_images(fid_model, model, data_loader, batch_size, fid_layer_index=-4, cuda=False):
    ''' Extract features and computes the FID score for the VAE vs. the classifier
        NOTE: expects a trained fid classifier and model '''
    fid_submodel = Submodel(fid_model, layer_index=fid_layer_index)
    fid_submodel.eval()
    model.eval()

    # calculate how many synthetic images from the student model
    num_test_samples = num_samples_in_loader(data_loader.test_loader)
    num_synthetic = int(np.ceil(num_test_samples // batch_size))
    fid, count = 0.0, 0

    with torch.no_grad():
        synthetic = [model.generate_synthetic_samples(model.student, batch_size)
                     for _ in range(num_synthetic + 1)]
        for (data, _), generated in zip(data_loader.test_loader, synthetic):
            data = Variable(data).cuda() if cuda else Variable(data)
            fid += frechet_gauss_gauss(
                D.Normal(torch.mean(fid_submodel(generated), dim=0), torch.var(fid_submodel(generated), dim=0)),
                D.Normal(torch.mean(fid_submodel(data), dim=0), torch.var(fid_submodel(data), dim=0))
            ).cpu().numpy()
            count += 1

    frechet_dist = fid / count
    print("frechet distance [ {} samples ]: {}\n".format(
        (num_test_samples // batch_size) * batch_size, frechet_dist)
    )
    return frechet_dist


def calculate_consistency(model, loader, reparam_type, vae_type, cuda=False):
    ''' \sum z_d(teacher) == z_d(student) for all test samples '''
    consistency = 0.0

    if model.current_model > 0 and (reparam_type == 'mixture'
                                    or reparam_type == 'discrete'):
        model.eval() # prevents data augmentation
        consistency = []

        for img, _ in loader.test_loader:
            with torch.no_grad():
                img = Variable(img).cuda() if cuda else Variable(img)

                output_map = model(img)
                if vae_type == 'parallel':
                    teacher_posterior = output_map['teacher']['params']['discrete']['logits']
                    student_posterior = output_map['student']['params']['discrete']['logits']
                else: # sequential
                    if 'discrete' not in output_map['teacher']['params']['params_0']:
                        # we need the first reparam to be discrete to calculate
                        # the consistency metric
                        return consistency

                    teacher_posterior = output_map['teacher']['params']['params_0']['discrete']['logits']
                    student_posterior = output_map['student']['params']['params_0']['discrete']['logits']

                teacher_posterior = F.softmax(teacher_posterior, dim=-1)
                student_posterior = F.softmax(student_posterior, dim=-1)
                teacher_posterior, student_posterior \
                    = zero_pad_smaller_cat(teacher_posterior, student_posterior)

                correct = to_data(teacher_posterior).max(1)[1] \
                          == to_data(student_posterior).max(1)[1]
                consistency.append(torch.mean(correct.type(float_type(cuda))))
                # print("teacher = ", teacher_posterior)
                # print("student = ", student_posterior)
                # print("consistency[-1]=", correct)

        num_test_samples = num_samples_in_loader(loader.test_loader)
        consistency = np.mean(consistency)
        print("Consistency [#samples: {}]: {}\n".format(num_test_samples,
                                                      consistency))
    return np.asarray([consistency])


def estimate_fisher(model, data_loader, batch_size, sample_size=1024, cuda=False):
    # modified from github user kuc2477's code
    # sample loglikelihoods from the dataset.
    loglikelihoods = []
    for x, _ in data_loader.train_loader:
        x = Variable(x).cuda() if cuda else Variable(x)
        reconstr_x, params = model.teacher(x)
        loss_teacher = model.teacher.loss_function(reconstr_x, x, params)
        loglikelihoods.append(
            loss_teacher['loss']
        )
        if len(loglikelihoods) >= sample_size // batch_size:
            break

    # estimate the fisher information of the parameters.
    loglikelihood = torch.cat(loglikelihoods, 0).mean(0)
    loglikelihood_grads = torch.autograd.grad(
        loglikelihood, model.teacher.parameters()
    )
    parameter_names = [
        n.replace('.', '__') for n, p in model.teacher.named_parameters()
    ]
    return {n: g**2 for n, g in zip(parameter_names, loglikelihood_grads)}
