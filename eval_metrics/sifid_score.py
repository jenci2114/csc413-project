#!/usr/bin/env python3
"""Calculates ***Single Image*** Frechet Inception Distance (SIFID) to evalulate Single-Image-GANs
Code was adapted from:
https://github.com/mseitzer/pytorch-fid.git
Which was adapted from the TensorFlow implementation of:

                                 
 https://github.com/bioinf-jku/TTUR

The FID metric calculates the distance between two distributions of images.
The SIFID calculates the distance between the distribution of deep features of a single real image and a single fake image.
Copyright 2018 Institute of Bioinformatics, JKU Linz
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.                                                               
"""

import os
import pathlib
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import numpy as np
import torch
from scipy import linalg
#from scipy.misc import imread
from matplotlib.pyplot import imread
from torch.nn.functional import adaptive_avg_pool2d

try:
    from tqdm import tqdm
except ImportError:
    # If not tqdm is not available, provide a mock version of it
    def tqdm(x): return x

from inception import InceptionV3
import torchvision
import numpy
import scipy
import pickle

import sys
current_dir = os.path.abspath(os.path.dirname(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument('--path2real', type=str, help=('Path to the real images'))
parser.add_argument('--path2fake', type=str, help=('Path to generated images'))
parser.add_argument('--gpu', default='', type=str, help='GPU to use (leave blank for CPU only)')
parser.add_argument('--images_suffix', default='jpg', type=str, help='image file suffix')
parser.add_argument('--exp_name', type=str, help='experiment name')
parser.add_argument('--dims', type=int, help='experiment name')


def get_activations(files, model, batch_size=1, dims=64,
                    cuda=True, verbose=False):
    """Calculates the activations of the pool_3 layer for all images.

    Params:
    -- files       : List of image files paths
    -- model       : Instance of inception model
    -- batch_size  : Batch size of images for the model to process at once.
                     Make sure that the number of samples is a multiple of
                     the batch size, otherwise some samples are ignored. This
                     behavior is retained to match the original FID score
                     implementation.
    -- dims        : Dimensionality of features returned by Inception
    -- cuda        : If set to True, use GPU
    -- verbose     : If set to True and parameter out_step is given, the number
                     of calculated batches is reported.
    Returns:
    -- A numpy array of dimension (num images, dims) that contains the
       activations of the given tensor when feeding inception with the
       query tensor.
    """
    model.eval()

    if len(files) % batch_size != 0:
        print(('Warning: number of images is not a multiple of the '
               'batch size. Some samples are going to be ignored.'))
    if batch_size > len(files):
        print(('Warning: batch size is bigger than the data size. '
               'Setting batch size to data size'))
        batch_size = len(files)

    n_batches = len(files) // batch_size
    n_used_imgs = n_batches * batch_size

    pred_arr = np.empty((n_used_imgs, dims))

    for i in range(n_batches):
        if verbose:
            print('\rPropagating batch %d/%d' % (i + 1, n_batches),
                  end='', flush=True)
        start = i * batch_size
        end = start + batch_size

        images = np.array([imread(str(f)).astype(np.float32)
                           for f in files[start:end]])

        images = images[:,:,:,0:3]
        # Reshape to (n_images, 3, height, width)
        images = images.transpose((0, 3, 1, 2))
        #images = images[0,:,:,:]
        images /= 255

        batch = torch.from_numpy(images).type(torch.FloatTensor)
        if cuda:
            batch = batch.cuda()

        pred = model(batch)[0]

        # If model output is not scalar, apply global spatial average pooling.
        # This happens if you choose a dimensionality not equal 2048.

        #if pred.shape[2] != 1 or pred.shape[3] != 1:
        #    pred = adaptive_avg_pool2d(pred, output_size=(1, 1))


        pred_arr = pred.cpu().data.numpy().transpose(0, 2, 3, 1).reshape(batch_size*pred.shape[2]*pred.shape[3],-1)


    if verbose:
        print(' done')

    return pred_arr


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

    Stable version by Dougal J. Sutherland.

    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.

    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean)


def calculate_activation_statistics(files, model, batch_size=1,
                                    dims=64, cuda=True, verbose=False):
    """Calculation of the statistics used by the FID.
    Params:
    -- files       : List of image files paths
    -- model       : Instance of inception model
    -- batch_size  : The images numpy array is split into batches with
                     batch size batch_size. A reasonable batch size
                     depends on the hardware.
    -- dims        : Dimensionality of features returned by Inception
    -- cuda        : If set to True, use GPU
    -- verbose     : If set to True and parameter out_step is given, the
                     number of calculated batches is reported.
    Returns:
    -- mu    : The mean over samples of the activations of the inception model.
    -- sigma : The covariance matrix of the activations of the inception model.
    """
    act = get_activations(files, model, batch_size, dims, cuda, verbose)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma


def _compute_statistics_of_path(files, model, batch_size, dims, cuda):
    if path.endswith('.npz'):
        f = np.load(path)
        m, s = f['mu'][:], f['sigma'][:]
        f.close()
    else:
        path = pathlib.Path(path)
        files = sorted(list(path.glob('*.jpg'))+ list(path.glob('*.png')))
        m, s = calculate_activation_statistics(files, model, batch_size,
                                               dims, cuda)

    return m, s


def calculate_sifid_given_paths(path1, path2, batch_size, cuda, dims, suffix):
    """Calculates the SIFID of two paths"""

    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]

    model = InceptionV3([block_idx])
    if cuda:
        model.cuda()

    path1 = pathlib.Path(path1)
    files1 = sorted(list(path1.glob('*.%s' %suffix)))

    path2 = pathlib.Path(path2)
    files2 = sorted(list(path2.glob('*.%s' %suffix)))

    fid_values = []
    Im_ind = []

    for i in range(len(files2)):
        m1, s1 = calculate_activation_statistics([files1[i]], model, batch_size, dims, cuda)
        m2, s2 = calculate_activation_statistics([files2[i]], model, batch_size, dims, cuda)
        fid_values.append(calculate_frechet_distance(m1, s1, m2, s2))
        file_num1 = files1[i].name
        file_num2 = files2[i].name
        Im_ind.append(int(file_num1[:-5]))
        Im_ind.append(int(file_num2[:-5]))
    return fid_values


def calculate_sifid_ti(path1, path2, batch_size, cuda, dims, suffix):
    """Similar to SIFID, except the path1 contains a single image andb path2 contains 
    generated batch of images"""

    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]

    model = InceptionV3([block_idx])
    if cuda:
        model.cuda()

    path1 = pathlib.Path(path1)
    files1 = sorted(list(path1.glob('*.%s' %suffix)))
    files1.extend(sorted(list(path1.glob('*.%s' %'jpg'))))

    path2 = pathlib.Path(path2)
    files2 = sorted(list(path2.glob('*.%s' %suffix)))
    fid_values = []
    # breakpoint()
    idx = 0
    for i in range(len(files2)):
        if not os.path.isdir(files2[i]):
            m1, s1 = calculate_activation_statistics([files1[0]], model, batch_size, dims, cuda)
            m2, s2 = calculate_activation_statistics([files2[idx]], model, batch_size, dims, cuda)
            fid_values.append(calculate_frechet_distance(m1, s1, m2, s2))
            idx += 1
    return fid_values

if __name__ == '__main__':
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    path1 = args.path2real
    path2 = args.path2fake
    suffix = args.images_suffix
    exp_name = args.exp_name
    dims = args.dims
    category = args.category

    # eval single vs batch
    sifid_values = calculate_sifid_ti(path1, path2, 1, args.gpu != '', dims, suffix)  # 64, 192, 768
    # take average
    sifid_values = np.asarray(sifid_values, dtype=np.float32)
    exp_name += f"_inception_{dims}"
    sifid_mean = sifid_values.mean()

    with open('fid_scores_{}.txt'.format(category), 'a') as f:
        f.write(f"{exp_name}\t{sifid_mean}\n")
 
    
# EXAMPLE CALL
# CUDA_VISIBLE_DEVICES=0 python ti_inference.py --category clock --init_token null --eval_step 0 --num_rows 10 --num_cols 4 
# CUDA_VISIBLE_DEVICES=1 python ti_inference.py --category clock --init_token clock --eval_step 0 --num_rows 10 --num_cols 4

# python sifid_score.py --path2real ../textual_inversion/images/clock_small --path2fake ../clock/null_init_results/step_0 --images_suffix jpeg --exp_name clock_null --gpu 0

# python sifid_score.py --path2real ../textual_inversion/images/clock_small --path2fake ../clock/clock_init_results/step_0 --images_suffix jpeg --exp_name clock_clock --gpu 1