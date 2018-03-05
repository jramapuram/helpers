import numpy as np
import torch
import torch.nn.functional as F
import torch.distributions as D

from torch.autograd import Variable

from models.layers import Submodel
from helpers.utils import to_data, float_type, \
    frechet_gauss_gauss, frechet_gauss_gauss_np, \
    num_samples_in_loader, zero_pad_smaller_cat



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
