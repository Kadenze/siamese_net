# -*- coding: utf-8 -*-
"""Siamese Network for performing training of a Deep Convolutional
Network for Face Verification on the Olivetti and LFW Faces datasets.

Dependencies:

python 3.4+, numpy>=1.10.4, sklearn>=0.17, scipy>=0.17.0, theano>=0.7.0, lasagne>=0.1, cv2, dlib>=18.18 (only required if using the 'trees' crop mode).

Part of the package siamese_net:
siamese_net/
siamese_net/faces.py
siamese_net/datasets.py
siamese_net/normalization.py
siamese_net/siamese_net.py


Copyright 2016 Kadenze, Inc.
Kadenze(R) and Kannu(R) are Registered Trademarks of Kadenze, Inc.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

Apache License

Version 2.0, January 2004

http://www.apache.org/licenses/
"""

import sys
import pickle
import os
# base_compiledir = os.path.expandvars("$HOME/.theano/slot-%d" % (os.getpid()))
# os.environ['THEANO_FLAGS'] = "base_compiledir=%s" % base_compiledir

import matplotlib.pyplot as plt
import numpy as np
import theano
import theano.tensor as T
import time
import lasagne

# For training the final output network
from sklearn import cross_validation
from sklearn import metrics
from sklearn.linear_model import LogisticRegression

# Custom code for parsing datasets and normalizing images
from datasets import Datasets
from normalization import LCN, ZCA

# plt.style.use('ggplot')
theano.config.floatX = 'float32'


def montage(x):
    if x.shape[1] == 1 or x.shape[1] == 3:
        num_img = x.shape[0]
        num_img_per_dim = np.ceil(np.sqrt(num_img)).astype(int)
        montage_img = np.zeros((
            num_img_per_dim * x.shape[3],
            num_img_per_dim * x.shape[2], x.shape[1]))
    else:
        num_img_per_dim = np.ceil(np.sqrt(x.shape[1])).astype(int)
        montage_img = np.zeros((
            num_img_per_dim * x.shape[3],
            num_img_per_dim * x.shape[2]))
        num_img = x.shape[1]

    for img_i in range(num_img_per_dim):
        for img_j in range(num_img_per_dim):
            if img_i * num_img_per_dim + img_j < num_img:
                if x.shape[0] == 1:
                    montage_img[
                        img_i * x.shape[3]: (img_i + 1) * x.shape[2],
                        img_j * x.shape[3]: (img_j + 1) * x.shape[2]
                    ] = np.squeeze(np.squeeze(
                        x[0, img_i * num_img_per_dim + img_j, ...]
                    ) / (np.max(x[0, img_i * num_img_per_dim + img_j, ...]
                                ) + 1e-15))
                else:
                    montage_img[
                        img_i * x.shape[3]: (img_i + 1) * x.shape[2],
                        img_j * x.shape[3]: (img_j + 1) * x.shape[2],
                        :
                    ] = np.swapaxes(np.squeeze(
                        x[img_i * num_img_per_dim + img_j, ...]
                    ) / (np.max(x[img_i * num_img_per_dim + img_j, ...]
                                ) + 1e-15), 0, 2)
    return montage_img


def get_image_manifold(images, features, res=64, n_neighbors=5):
    '''Creates a montage of the images based on a TSNE
    manifold of the associated image features.
    '''

    from sklearn import manifold
    mapper = manifold.SpectralEmbedding()
    transform = mapper.fit_transform(features)
    nx = int(np.ceil(np.sqrt(len(transform))))
    ny = int(np.ceil(np.sqrt(len(transform))))
    montage_img = np.zeros((res * nx, res * ny, 3))
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors()
    nn.fit(transform)
    min_x = np.mean(transform[:, 0]) - np.std(transform[:, 0]) * 3.0
    max_x = np.mean(transform[:, 0]) + np.std(transform[:, 0]) * 3.0
    min_y = np.mean(transform[:, 1]) - np.std(transform[:, 1]) * 3.0
    max_y = np.mean(transform[:, 1]) + np.std(transform[:, 1]) * 3.0

    for n_i in range(nx):
        for n_j in range(ny):
            x = min_x + (max_x - min_x) / nx * n_i
            y = min_y + (max_y - min_y) / ny * n_j
            idx = nn.kneighbors([x, y], n_neighbors=n_neighbors)[1][0][:]
            for neighbor_i in idx:
                montage_img[
                    n_i * res: (n_i + 1) * res, n_j * res: (n_j + 1) * res, :] += images[neighbor_i]
            montage_img[
                n_i * res: (n_i + 1) * res, n_j * res: (n_j + 1) * res, :] /= float(len(idx))
    montage_img = montage_img / np.max(montage_img)
    return montage_img


def make_image_pairs(X, y, unique_labels):
    '''For each person in unique_labels (P people):

    1. combine all matched pairs the images of that person (N images):

    N_matched = (P choose 2) * (N choose 2)

    2. combine all imposter pairs.  N_unmatched = (P choose 2) * (N * N)

    Returns an array of matched and unmatched images and their targets
    ------------------------------------------------------------------
    X_matched, y_matched, X_unmatched, y_unmatched

    where the dimensions of the Xs are (with 2 being each image in the pair):
        [(N_matched + N_unmatched) x 2 x W x H]

    and ys are
    ----------
        [(N_matched + N_unmatched),]

    Args
    ----
    X : TYPE
        Description
    y : TYPE
        Description
    unique_labels : TYPE
        Description

    Deleted Parameters
    ------------------
    X (TYPE) : Description
    y (TYPE) : Description
    unique_labels (TYPE) : Description
    '''
    from itertools import combinations

    X_pairs_matched = list()
    y_pairs_matched = list()

    # Iterate over all actual pairs
    # 32 choose 2 = 496 people pairs. 496 * (10 images choose 2) = 496 * 45 =
    # 1440
    for person in unique_labels:
        # Find images of those people
        im_idx = np.where(person == y)[0]
        for el in combinations(im_idx, 2):
            X_pairs_matched.append(
                np.concatenate((X[el[0], ...], X[el[1], ...]),
                               axis=0)[np.newaxis, ...])
            y_pairs_matched.append(1)

    X_pairs_unmatched = list()
    y_pairs_unmatched = list()

    # Iterate over all imposter pairs of people
    # (32 choose 2 = 496 people pairs. 496 * 10 * 10 image pairs =
    # 49600 imposter pairs)
    # (157 * 0.4 = 63), 63 choose 2 = 1953, 1953 * 100 = 195300
    for pair in combinations(unique_labels, 2):
        # Find images of those people
        im1_idx = np.where(pair[0] == y)[0]
        im2_idx = np.where(pair[1] == y)[0]
        for im1_idx_it in im1_idx:
            for im2_idx_it in im2_idx:
                X_pairs_unmatched.append(np.concatenate(
                    (X[im1_idx_it, ...], X[im2_idx_it, ...]),
                    axis=0)[np.newaxis, ...])
                y_pairs_unmatched.append(0)

    return (np.concatenate(X_pairs_matched),
            np.array(y_pairs_matched),
            np.concatenate(X_pairs_unmatched),
            np.array(y_pairs_unmatched))


def make_image_pair_idxs(y, unique_labels):
    '''For each person in unique_labels (P people):

    1. combine all matched pairs the images of that person (N images):
    N_matched = (P choose 2) * (N choose 2)

    2. combine all imposter pairs.  N_unmatched = (P choose 2) * (N * N)

    Returns an array of matched and unmatched images and their targets
    ------------------------------------------------------------------
    X_matched, y_matched, X_unmatched, y_unmatched

    where the dimensions of the Xs are [(N_matched + N_unmatched) x 2]
    (with 2 being the index into X defining the image in the pair),
    and ys are [(N_matched + N_unmatched),]

    Args
    ----
    y : TYPE
        Description
    unique_labels : TYPE
        Description

    Deleted Parameters
    ------------------
    y (TYPE) : Description
    unique_labels (TYPE) : Description
    '''
    from itertools import combinations

    X_pairs_matched = list()
    y_pairs_matched = list()

    # Iterate over all actual pairs
    # 32 choose 2 = 496 people pairs. 496 * (10 images choose 2) = 496 * 45 =
    # 1440
    for person in unique_labels:
        # Find images of those people
        im_idx = np.where(person == y)[0]
        for el in combinations(im_idx, 2):
            X_pairs_matched.append(np.array([el[0], el[1]])[np.newaxis, ...])
            y_pairs_matched.append(1)

    X_pairs_unmatched = list()
    y_pairs_unmatched = list()

    # Iterate over all imposter pairs of people
    # (32 choose 2 = 496 people pairs. 496 * 10 * 10 image pairs = 49600 imposter pairs)
    # (157 * 0.4 = 63), 63 choose 2 = 1953, 1953 * 100 = 195300
    for pair_i, pair in enumerate(combinations(unique_labels, 2)):
        # Find images of those people
        im1_idx = np.where(pair[0] == y)[0]
        im2_idx = np.where(pair[1] == y)[0]
        for im1_idx_it in im1_idx:
            for im2_idx_it in im2_idx:
                X_pairs_unmatched.append(
                    np.array([im1_idx_it, im2_idx_it])[np.newaxis, ...])
                y_pairs_unmatched.append(0)

    return (np.concatenate(X_pairs_matched),
            np.array(y_pairs_matched),
            np.concatenate(X_pairs_unmatched),
            np.array(y_pairs_unmatched))


def draw_image_pair(X, y, idx=None):
    '''Given X of N x 2 x W x H, and the associated label matrix, plot
    a random pair, or a given idx.

    Keyword arguments
    -----------------

    idx -- Integer - Which pair to show.  If none is given, then a
    idx -- Integer - Which pair to show.  If none is given, then a
    random one is picked. [None]

    Args
    ----
    X : TYPE
        Description
    y : TYPE
        Description
    idx : TYPE, optional
        Description

    Deleted Parameters
    ------------------
    X (TYPE) : Description
    y (TYPE) : Description
    idx (TYPE, optional) : Description
    '''
    if idx is None:
        idx = np.random.randint(len(X) - 2)
    if X.shape[1] == 1:
        idx = idx + (idx % 2)
    fig, [ax1, ax2] = plt.subplots(1, 2, figsize=(10, 8))
    if X.shape[1] == 2:
        ax1.imshow(np.squeeze(X[idx, 0, ...]), cmap='gray')
        ax2.imshow(np.squeeze(X[idx, 1, ...]), cmap='gray')
    else:
        ax1.imshow(np.squeeze(X[idx, ...]), cmap='gray')
        ax2.imshow(np.squeeze(X[idx + 1, ...]), cmap='gray')
    ax1.grid(False)
    ax2.grid(False)
    if y[idx] == 0:
        fig.suptitle('Unmatched: %d' % idx, fontsize=30)
    else:
        fig.suptitle('Matched: %d' % idx, fontsize=30)


def load_pairs(
        dataset='lfw',
        normalization='LCN',
        split=(0.8, 0.1, 0.1),
        resolution=(128, 128),
        crop_style='none',
        crop_factor=1.2,
        n_files_per_person=5,
        path_to_data=None,
        b_load_idxs_only=True,
        b_convert_to_grayscale=True):
    '''
    Given a dataset name, generate the training, validation, and testing
    data of matched and unmatched pairs, optionally applying normalization
    to each image.

    Note this method only returns the idxs of the original dataset.

    Parameters
    ----------
    dataset -- string
        The name of the dataset to load, 'olivetti', ['lfw'].

    normalization -- string
        The type of normalization to apply, if any of ['LCN'], 'LCN-',
        'ZCA', or '-1:1'.

    split -- tuple
        The (train, valid, test) split fractions [(0.6, 0.2, 0.2)].

    num_files -- int
        Number of files to load for each person.
    '''
    ds = None
    if dataset == 'olivetti':
        from sklearn.datasets import fetch_olivetti_faces
        ds = fetch_olivetti_faces()
        # TODO: Apply processing options to olivetti
    elif dataset == 'lfw':
        ds = Datasets(
            crop_style=crop_style,
            crop_factor=crop_factor,
            resolution=resolution,
            n_files_per_person=n_files_per_person,
            n_min_files_per_person=(n_files_per_person / 2),
            b_convert_to_grayscale=b_convert_to_grayscale
        )
        ds = ds.get_parsed_dataset(dataset=dataset, path_to_data=path_to_data)
    elif dataset.__class__ is dict and 'target' in dataset.keys() and 'images' in dataset.keys():
        ds = dataset
    else:
        raise ValueError(
            'Dataset should be either olivetti, lfw, or a dict defining images and target from get_parsed_dataset')

    # Split up the dataset into unique targets for train/test,
    # making sure not to repeat any subjects between train/test
    # Should get 32 subjects train, 8 test, with a 0.8 split

    y = ds['target']

    total = len(np.unique(y))

    train_split = int(total * split[0])
    valid_split = train_split + int(total * split[1])
    test_split = total - int(total * split[2])

    unique_train_labels = np.unique(y)[:train_split]
    unique_valid_labels = np.unique(y)[train_split:valid_split]
    unique_test_labels = np.unique(y)[-test_split:]

    # X = (400, 1, 64, 64);  y = (400,), 40 subjects w/ 10 examples each of 64
    # x 64 pixels
    if b_convert_to_grayscale:
        X = np.concatenate([img[np.newaxis, np.newaxis, ...]
                            for img in ds['images']], axis=0)
    else:
        X = np.concatenate([img[np.newaxis, ...]
                            for img in ds['images']], axis=0)
    print(X.shape)

    if normalization == 'LCN':
        lcn = LCN(sigma=round(0.0625 * X.shape[2]), subtractive=False)
        lcn.fit(X[:len(y) * split[0], ...])
        X = lcn.transform(X)
    elif normalization == 'LCN-':
        lcn = LCN(sigma=round(0.0625 * X.shape[2]), subtractive=True)
        lcn.fit(X[:len(y) * split[0], ...])
        X = lcn.transform(X)
    elif normalization == 'ZCA':
        zca = ZCA(bias=0.1)
        zca.fit(X[:len(y) * split[0], ...])
        X = zca.transform(X)
    elif normalization == '-1:1':
        for idx in range(len(X)):
            X[idx, ...] = (X[idx, ...] - np.min(X[idx, ...])) / \
                (np.max(X[idx, ...]) - np.min(X[idx, ...])) * 2.0 - 1.0

    if b_load_idxs_only:
        # Make pairs of actual and imposter faces, returning the indexes to
        # create them
        print('train')
        X_train_matched, y_train_matched, X_train_unmatched, y_train_unmatched = make_image_pair_idxs(
            y, unique_train_labels)
        print('valid')
        X_valid_matched, y_valid_matched, X_valid_unmatched, y_valid_unmatched = make_image_pair_idxs(
            y, unique_valid_labels)
        print('test')
        X_test_matched, y_test_matched, X_test_unmatched, y_test_unmatched = make_image_pair_idxs(
            y, unique_test_labels)

        return {
            'X': lasagne.utils.floatX(X),
            'y': y.astype(np.int32),
            'X_train_matched_idxs': X_train_matched.astype(np.int32),
            'y_train_matched_idxs': y_train_matched.astype(np.int32),
            'X_train_unmatched_idxs': X_train_unmatched.astype(np.int32),
            'y_train_unmatched_idxs': y_train_unmatched.astype(np.int32),
            'X_valid_matched_idxs': X_valid_matched.astype(np.int32),
            'y_valid_matched_idxs': y_valid_matched.astype(np.int32),
            'X_valid_unmatched_idxs': X_valid_unmatched.astype(np.int32),
            'y_valid_unmatched_idxs': y_valid_unmatched.astype(np.int32),
            'X_test_matched_idxs': X_test_matched.astype(np.int32),
            'y_test_matched_idxs': y_test_matched.astype(np.int32),
            'X_test_unmatched_idxs': X_test_unmatched.astype(np.int32),
            'y_test_unmatched_idxs': y_test_unmatched.astype(np.int32)
        }
    else:
        # Make pairs of actual and imposter faces
        X_train_matched, y_train_matched, X_train_unmatched, y_train_unmatched = make_image_pairs(
            X, y, unique_train_labels)
        X_valid_matched, y_valid_matched, X_valid_unmatched, y_valid_unmatched = make_image_pairs(
            X, y, unique_valid_labels)
        X_test_matched, y_test_matched, X_test_unmatched, y_test_unmatched = make_image_pairs(
            X, y, unique_test_labels)

        return {
            'X_train_matched': lasagne.utils.floatX(X_train_matched),
            'y_train_matched': y_train_matched.astype(np.int32),
            'X_train_unmatched': lasagne.utils.floatX(X_train_unmatched),
            'y_train_unmatched': y_train_unmatched.astype(np.int32),
            'X_valid_matched': lasagne.utils.floatX(X_valid_matched),
            'y_valid_matched': y_valid_matched.astype(np.int32),
            'X_valid_unmatched': lasagne.utils.floatX(X_valid_unmatched),
            'y_valid_unmatched': y_valid_unmatched.astype(np.int32),
            'X_test_matched': lasagne.utils.floatX(X_test_matched),
            'y_test_matched': y_test_matched.astype(np.int32),
            'X_test_unmatched': lasagne.utils.floatX(X_test_unmatched),
            'y_test_unmatched': y_test_unmatched.astype(np.int32)
        }


def interleave_dataset(X_split, y_split):
    '''Take paired observations in the channel dimension and convert them

    to alternating batches
    ----------------------

    N x 2 x W x H  -->  2*N x 1 x W x H

    Args
    ----
    X_split : TYPE
        Description
    y_split : TYPE
        Description

    Deleted Parameters
    ------------------
    X_split (TYPE) : Description
    y_split (TYPE) : Description
    '''

    # TODO: account for color images
    n_batch, n_channels, n_height, n_width = X_split.shape
    n_obs = n_batch * n_channels
    n_feats = n_height * n_width

    X_interleaved = np.empty((n_obs, n_feats), dtype=theano.config.floatX)
    y_interleaved = np.empty((n_obs,), dtype=np.int32)

    X_interleaved[0::2] = X_split[:, 0, ...].reshape(n_batch, n_feats)
    X_interleaved[1::2] = X_split[:, 1, ...].reshape(n_batch, n_feats)

    y_interleaved[0::2] = y_split.copy()
    y_interleaved[1::2] = y_split.copy()

    return X_interleaved.reshape(n_obs, 1, n_height, n_width), y_interleaved


def shuffle_dataset(X, y):
    '''Randomly permute the order of the observations and their associated labels

    Parameters
    ----------
    X : TYPE
        Description
    y : TYPE
        Description
    '''
    indices = np.random.permutation(len(y))
    return X[indices, ...], y[indices, ...]


def get_balanced_shuffled_dataset(X_matched, y_matched, X_unmatched, y_unmatched):
    '''Shuffles dataset, producing training data with similar number of matched
    and unmatched observations. There are often much more unmatched
    observations, so this method is used to sample from the larger set of
    unmatched observations, while using as many matched observations as
    there are, but randomly permuting their order.

    Parameters
    ----------
    X_matched : TYPE
        Description
    y_matched : TYPE
        Description
    X_unmatched : TYPE
        Description
    y_unmatched : TYPE
        Description
    '''
    npairs = X_matched.shape[0]

    # Shuffle order
    X_matched, y_matched = shuffle_dataset(X_matched, y_matched)
    X_unmatched, y_unmatched = shuffle_dataset(X_unmatched, y_unmatched)

    # Sample same number of unmatched data
    X_train = np.concatenate((X_matched, X_unmatched[:npairs]))
    y_train = np.concatenate((y_matched, y_unmatched[:npairs]))

    # Shuffle again so that batches aren't all matched/unmatched
    X_train, y_train = shuffle_dataset(X_train, y_train)

    return X_train, y_train


def make_shared(X, dtype):
    '''Convert `X` to a theano shared variable with the given type.

    Parameters
    ----------
    X : TYPE
        Description
    dtype : TYPE
        Description
    '''
    return theano.shared(np.asarray(X, dtype=dtype), borrow=True)


def generate_new_dataset_batch(X_matched, y_matched, X_unmatched, y_unmatched, batch_size):
    '''Generator which loops through a randomly permuted ordering of the dataset.
    This method requires the generated pairs of the data, which is a much
    higher number of observations than the original dataset.

    If you cannot fit the entire dataset into memory, use the slower method:
    `generate_new_dataset_batch_from_idxs`

    Returns X_train, y_train

    Parameters
    ----------
    X_matched : TYPE
        Description
    y_matched : TYPE
        Description
    X_unmatched : TYPE
        Description
    y_unmatched : TYPE
        Description
    batch_size : TYPE
        Description
    '''
    # Generate a new shuffled, balanced dataset
    X_train, y_train = get_balanced_shuffled_dataset(
        X_matched, y_matched, X_unmatched, y_unmatched)

    # Interleave pairs into sequential batches which will be used in the
    # distance/loss functions appropriately
    X_train, y_train = interleave_dataset(X_train, y_train)

    nobs = len(X_train)

    # Make sure it is even
    batch_size = batch_size + (batch_size % 2)

    # Loop until we're out of observations
    batch_start = 0
    batch_end = batch_size
    while batch_start < np.min((nobs, (nobs - batch_size))):
        yield X_train[batch_start:batch_end, ...], y_train[batch_start:batch_end, ...]
        batch_start = batch_end
        batch_end = batch_start + batch_size


def generate_new_dataset_batch_from_idxs(
        X, y, X_matched_idxs, y_matched_idxs,
        X_unmatched_idxs, y_unmatched_idxs, batch_size):
    '''Generator which loops through a randomly permuted ordering of the dataset.
    This method requires the generated pairs of the data as indexes.

    Returns X_train, y_train

    Parameters
    ----------
    X : TYPE
        Description
    y : TYPE
        Description
    X_matched_idxs : TYPE
        Description
    y_matched_idxs : TYPE
        Description
    X_unmatched_idxs : TYPE
        Description
    y_unmatched_idxs : TYPE
        Description
    batch_size : TYPE
        Description
    '''
    # Generate a new shuffled, balanced dataset
    X_train, y_train = get_balanced_shuffled_dataset(
        X_matched_idxs, y_matched_idxs, X_unmatched_idxs, y_unmatched_idxs)

    # Interleave pairs into sequential batches which will be used in the distance/loss functions appropriately
    # TODO: account for color images
    X_train, y_train = interleave_dataset(
        X_train[..., np.newaxis, np.newaxis], y_train)
    X_train = np.squeeze(X_train).astype(np.int32)
    y_train = np.squeeze(y_train).astype(np.int32)

    nobs = len(X_train)

    # Make sure it is even
    batch_size = batch_size + (batch_size % 2)

    # Loop until we're out of observations
    batch_start = 0
    batch_end = batch_size
    while batch_start < np.min((nobs, (nobs - batch_size))):
        yield X[X_train[batch_start:batch_end, ...], ...], y_train[batch_start:batch_end, ...]
        batch_start = batch_end
        batch_end = batch_start + batch_size


class SiameseNetPredictor(object):

    '''Loads a pre-trained Deep Net for Face Verification which uses a
    Siamese Net distance function + LogisticRegression on the final feature
    layer. Requires the pretrained model in the directory results

    Attributes
    ----------
    clf : TYPE
        Description
    fn : TYPE
        Description
    lcn : TYPE
        Description
    result : TYPE
        Description
    '''

    def __init__(self, images, filename='./lfw.pkl'):
        """Summary"""
        # Load the pretrained model
        self.result = pickle.load(open(filename, 'rb'))
        print(self.result['params'])
        self.grayscale = self.result['params']['b_convert_to_grayscale']
        self.normalization = self.result['params']['normalization']
        self.net = ConvSiameseNet(
            input_channels=(1
                            if self.grayscale
                            else 3),
            input_width=self.result['params']['resolution'][0],
            input_height=self.result['params']['resolution'][1],
            n_out=self.result['params']['n_features'],
            distance_fn=self.result['params']['distance_fn'],
            nonlinearity=self.result['params']['nonlinearity'])

        if self.result['params']['model_type'] == 'custom':
            self.net.use_custom_model()
        elif self.result['params']['model_type'] == 'hani':
            self.net.use_hani_model()
        elif self.result['params']['model_type'] == 'chopra':
            self.net.use_chopra_model()
        else:
            print('Unrecognized model!')

        self.net.set_from_parameters(
            pickle.loads(self.result['model_parameters']))
        pred = lasagne.layers.get_output(self.net.model, self.net.x,
                                         deterministic=True)

        # Compile
        self.fn = theano.function([self.net.x], [pred])

        # We'll hash functions for every layer if/when user asks for them
        self.fns = {}

        # Train final regressor on entire dataset
        # (cheating, but...¯\_(ツ)_/¯)
        Xs = self.result['prediction']['X']
        ys = self.result['prediction']['y']
        Xs_L1 = np.abs(Xs[:, :self.net.n_out] - Xs[:, self.net.n_out:])
        self.clf = LogisticRegression()
        self.clf.fit(Xs_L1, ys)

        # Load normalization kernel
        # (previously created using LCN on the training set)
        # self.lcn = pickle.loads(self.result['LCN'])
        if self.grayscale:
            X = np.concatenate([img[np.newaxis, np.newaxis, ...]
                                for img in images], axis=0)
        else:
            X = np.concatenate([img[np.newaxis, ...]
                                for img in images], axis=0)
        print(X.shape)

        if self.normalization == 'LCN':
            lcn = LCN(
                sigma=round(0.0625 * self.result['params']['resolution'][0]),
                subtractive=False)
            lcn.fit(X)
            self.norm = lcn
        elif self.normalization == 'LCN-':
            lcn = LCN(
                sigma=round(0.0625 * self.result['params']['resolution'][0]),
                subtractive=True)
            lcn.fit(X)
            self.norm = lcn
        elif self.normalization == 'ZCA':
            zca = ZCA(bias=0.1)
            zca.fit(X)
            self.norm = zca
        elif self.normalization == '-1:1':
            self.norm = lambda x: ((x - np.min(x)) / (np.max(x) - np.min(x)) * 2.0 - 1.0)

    def preprocess(self, X):
        '''Take an image in X, and transform it with local contrast normalization.

        Parameters
        ----------
        X : numpy.ndarray
            image to perform local contrast normalization on

        Returns
        -------
        img : numpy.ndarray
            Local contrast normalized image
        '''
        res = None
        try:
            res = self.norm.transform(X)
        except:
            res = self.norm(X)
            pass
        return res

    def features_for_layer(self, X, layer_num):
        if layer_num in self.fns.keys():
            fn = self.fns[layer_num]
        else:
            layer_output = lasagne.layers.get_output(
                lasagne.layers.get_all_layers(
                    self.net.model)[layer_num],
                self.net.x, deterministic=True)
            fn = theano.function([self.net.x], [layer_output])
            self.fns[layer_num] = fn
        out = fn(lasagne.utils.floatX(X))
        return out

    def features(self, X):
        '''Get siamese net features for the images in X.

        Parameters
        ----------
        X  :  numpy.ndarray
            N x C x W x H tensor defining the N images of W x H.
            For colorscale, C = 3, while for grayscale, C = 1.

        Returns
        -------
        features  :  numpy.ndarray
            N x M array of features
        '''
        return self.fn(X)

    def predict(self, X):
        '''Predict whether images contain the same face or not.

        Parameters
        ----------
        X  :  numpy.ndarray
            2*N x C x W x H tensor defining the N sequence of image pairs W x H.
            For colorscale, C = 3, while for grayscale, C = 1.

        Returns
        -------
        predictions  :  numpy.ndarray
            N x 1 vector of True/False predictions of whether the image
            pairs contain the same face or not.
        '''
        features = self.fn(X)
        Xs_L1 = np.abs(features[0][0::2] - features[0][1::2])
        final = self.clf.predict(Xs_L1)
        return final

    def get_normalization(self):
        '''Return the normalization type of the pre-trained network.

        Returns
        -------
        normalization_type  :  string
            'LCN', 'LCN-', '-1:1', 'ZCA'
        '''
        return self.result['params']['normalization']

    def get_crop(self):
        '''Return the crop type of the pre-trained network.'''
        return self.result['params']['crop']

    def get_resolution(self):
        '''Return the resolution of the images required by the pre-trained network.

        Returns
        -------
        (%d, %d)  :  tuple
            Resolution of the image
        '''
        return self.result['params']['resolution']

    def get_colorscale(self):
        '''Return the colorscale of the images required by the pre-trained network

        Returns
        -------
        is_grayscale  :  bool
            True if grayscale, else, False for RGB color.
        '''
        return self.result['params']['b_convert_to_grayscale']


class ConvSiameseNet:

    """Builds an object used for training a siamese net
    with different types of models and options.

    Attributes
    ----------
    batch_size : TYPE
        Description
    batch_slice : TYPE
        Description
    distance_fn : TYPE
        Description
    hyperparameter_margin : TYPE
        Description
    hyperparameter_threshold : TYPE
        Description
    index : TYPE
        Description
    input_channels : TYPE
        Description
    input_height : TYPE
        Description
    input_width : TYPE
        Description
    l_in : TYPE
        Description
    learning_rate : TYPE
        Description
    loss_fn : TYPE
        Description
    model : TYPE
        Description
    n_out : TYPE
        Description
    nonlinearity : TYPE
        Description
    srng : TYPE
        Description
    test_x : TYPE
        Description
    train_x : TYPE
        Description
    update : TYPE
        Description
    validation_x : TYPE
        Description
    weight_init : TYPE
        Description
    x : TYPE
        Description
    y : TYPE
        Description
    """

    def __init__(self,
                 input_channels,
                 input_width,
                 input_height,
                 n_out,
                 batch_size=None,
                 distance_fn='l1',
                 nonlinearity='scaled_tanh'):
        """Builds a ConvSiameseNet for training.

        Parameters
        ----------
        input_channels : TYPE
            Description
        input_width : TYPE
            Description
        input_height : TYPE
            Description
        n_out : TYPE
            Description
        batch_size : TYPE, optional
            Description
        distance_fn : str, optional
            Description
        nonlinearity : str, optional
            Description

        Raises
        ------
        ValueError
            Description
        """
        self.input_channels = input_channels
        self.input_width = input_width
        self.input_height = input_height
        self.n_out = n_out
        self.batch_size = batch_size

        self.l_in = lasagne.layers.InputLayer(
            shape=(None, input_channels, input_width, input_height))
        self.n_out = n_out

        self.srng = theano.sandbox.rng_mrg.MRG_RandomStreams()

        self.loss_fn = contrastive_loss
        if distance_fn.lower() == 'cosine':
            self.distance_fn = distance_cosine
        elif distance_fn.lower() == 'l1':
            self.distance_fn = distance_L1
        elif distance_fn.lower() == 'l2':
            self.distance_fn = distance_L2
        else:
            raise ValueError(
                'Must specify distance as either "cosine", "l1", or "l2".')

        self.x = T.tensor4('x')
        self.y = T.ivector('y')

        if nonlinearity == 'scaled_tanh':
            self.nonlinearity = lasagne.nonlinearities.ScaledTanH(
                scale_in=2. / 3, scale_out=1.7159)
        elif nonlinearity == 'rectify':
            self.nonlinearity = lasagne.nonlinearities.rectify
        else:
            raise ValueError(
                'Must specify nonlinearity as either "scaled_tanh" or "rectify".')

        self.weight_init = lasagne.init.Normal(std=0.05, mean=0.0)

    def use_hani_model(self, dropout_pct=0.0, b_spatial=False):
        """Summary

        Parameters
        ----------
        dropout_pct : float, optional
            Description
        b_spatial : bool, optional
            Description

        Returns
        -------
        name : TYPE
            Description
        """
        self.model = self.get_hani_2014_net(
            self.l_in, dropout_pct=dropout_pct, b_spatial=b_spatial)

    def use_custom_model(self, b_spatial=False):
        """Summary

        Parameters
        ----------
        b_spatial : bool, optional
            Description

        Returns
        -------
        name : TYPE
            Description
        """
        self.model = self.get_custom_net(self.l_in, b_spatial=b_spatial)

    def use_chopra_model(self, dropout_pct=0.0, b_spatial=False):
        """Summary

        Parameters
        ----------
        dropout_pct : float, optional
            Description
        b_spatial : bool, optional
            Description

        Returns
        -------
        name : TYPE
            Description
        """
        self.model = self.get_chopra_net(
            self.l_in, dropout_pct=dropout_pct, b_spatial=b_spatial)

    def use_deepid_model(self, b_spatial=False):
        """Summary

        Parameters
        ----------
        b_spatial : bool, optional
            Description

        Returns
        -------
        name : TYPE
            Description
        """
        self.model = self.get_deep_id_net(self.l_in, b_spatial=b_spatial)

    def get_spatial_transform_net(self, input_layer):
        """Summary

        Parameters
        ----------
        input_layer : TYPE
            Description

        Returns
        -------
        name : TYPE
            Description
        """
        # http://lasagne.readthedocs.org/en/latest/modules/layers/special.html?highlight=trainable#lasagne.layers.TransformerLayer
        # Localization network
        # Spatial Transformer Networks Max Jaderberg, Karen Simonyan, Andrew Zisserman, Koray Kavukcuoglu Submitted on 5 Jun 2015
        # Here we set up the layer to initially do the identity transform,
        # similarly to [R34]. Note that you will want to use a localization
        # with linear output. If the output from the localization networks
        # is [t1, t2, t3, t4, t5, t6] then t1 and t5 determines zoom, t2
        # and t4 determines skewness, and t3 and t6 move the center
        # position.

        b = np.zeros((2, 3), dtype=theano.config.floatX)
        b[0, 0] = 1
        b[1, 1] = 1
        b = b.flatten()
        loc_l1 = lasagne.layers.MaxPool2DLayer(input_layer, pool_size=(2, 2))
        loc_l2 = lasagne.layers.Conv2DLayer(
            loc_l1,
            num_filters=20,
            filter_size=(5, 5),
            W=self.weight_init
        )
        loc_l3 = lasagne.layers.MaxPool2DLayer(loc_l2, pool_size=(2, 2))
        loc_l4 = lasagne.layers.Conv2DLayer(
            loc_l3,
            num_filters=20,
            filter_size=(5, 5),
            W=self.weight_init
        )
        loc_l5 = lasagne.layers.DenseLayer(
            loc_l4,
            num_units=50,
            W=self.weight_init
        )
        loc_out = lasagne.layers.DenseLayer(
            loc_l5,
            num_units=6,
            b=b,
            W=self.weight_init,
            nonlinearity=lasagne.nonlinearities.identity
        )

        # Transformer network
        transformed_input_layer = lasagne.layers.TransformerLayer(
            input_layer, loc_out, downsample_factor=2.0)
        print('Transformed Input Shape: ',
              transformed_input_layer.output_shape)
        return transformed_input_layer

    def get_chopra_net(self, input_layer, dropout_pct=0.0, b_spatial=False):
        '''Return a lasagne network defining the siamese network
        Chopra, S., Hadsell, R., & Y., L. (2005). Learning a similiarty
        metric discriminatively, with application to face verification.
        Proceedings of IEEE Conference on Computer Vision and Pattern
        Recognition, 349–356.

        Modifications
        -------------

        dropout_pct -- Instead of a fixed connection layer, use dropout
        with this much percentage [0.5]

        b_spatial -- Prepend a spatial transformer network which applies
        an affine transformation and a 2x crop [False]

        Args
        ----
        input_layer : TYPE
            Description
        dropout_pct : float, optional
            Description
        b_spatial : bool, optional
            Description

        Deleted Parameters
        ------------------
        input_layer (TYPE) : Description
        dropout_pct (float : Description
        optional), b_spatial (bool : Description
        '''

        l_conv1 = None

        if b_spatial:
            # returns a 15x40x40
            l_conv1 = lasagne.layers.Conv2DLayer(
                self.get_spatial_transform_net(input_layer),
                num_filters=15,
                filter_size=(7, 7),
                nonlinearity=self.nonlinearity,
                W=self.weight_init
            )

        else:
            # returns a 15x40x40
            l_conv1 = lasagne.layers.Conv2DLayer(
                input_layer,
                num_filters=15,
                filter_size=(7, 7),
                nonlinearity=self.nonlinearity,
                W=self.weight_init
            )

        # returns a 15x20x20
        l_pool1 = lasagne.layers.MaxPool2DLayer(l_conv1, pool_size=(2, 2))

        # returns a 45x15x15
        l_conv2 = lasagne.layers.Conv2DLayer(
            l_pool1,
            num_filters=45,
            filter_size=(6, 6),
            nonlinearity=self.nonlinearity,
            W=self.weight_init
        )

        # returns a 45x5x5
        l_pool2 = lasagne.layers.MaxPool2DLayer(l_conv2, pool_size=(3, 3))
        l_pool2_dropout = lasagne.layers.DropoutLayer(l_pool2, p=dropout_pct)

        # returns a 250x1x1
        l_conv3 = lasagne.layers.Conv2DLayer(
            l_pool2_dropout,
            num_filters=250,
            filter_size=(5, 5),
            nonlinearity=self.nonlinearity,
            W=self.weight_init
        )

        l_hidden = lasagne.layers.DenseLayer(
            l_conv3,
            num_units=self.n_out,
            nonlinearity=self.nonlinearity,
            W=self.weight_init
        )

        model = lasagne.layers.DenseLayer(
            l_hidden,
            num_units=self.n_out,
            nonlinearity=self.nonlinearity,
            W=self.weight_init
        )

        return model

    def get_custom_net(self, input_layer, b_spatial=False):
        '''Return a lasagne network defining a custom siamese network

        Modifications
        -------------

        dropout_pct -- Instead of a fixed connection layer, use dropout
        with this much percentage [0.5]

        b_spatial -- Prepend a spatial transformer network which applies an
        affine transformation and a 2x crop [False]

        Args
        ----
        input_layer : TYPE
            Description
        b_spatial : bool, optional
            Description

        Deleted Parameters
        ------------------
        input_layer (TYPE) : Description
        b_spatial (bool, optional) : Description
        '''

        l_conv1a = None

        if b_spatial:
            l_conv1a = lasagne.layers.Conv2DLayer(
                self.get_spatial_transform_net(input_layer),
                num_filters=16,
                filter_size=(3, 3),
                nonlinearity=self.relu,
                W=self.weight_init
            )

        else:
            l_conv1a = lasagne.layers.Conv2DLayer(
                input_layer,
                num_filters=16,
                filter_size=(3, 3),
                nonlinearity=self.nonlinearity,
                W=self.weight_init
            )

        l_conv1b = lasagne.layers.Conv2DLayer(
            l_conv1a,
            num_filters=32,
            filter_size=(3, 3),
            nonlinearity=self.nonlinearity,
            W=self.weight_init
        )
        l_pool1 = lasagne.layers.MaxPool2DLayer(l_conv1b, pool_size=(2, 2))

        l_conv2a = lasagne.layers.Conv2DLayer(
            l_pool1,
            num_filters=32,
            filter_size=(3, 3),
            nonlinearity=self.nonlinearity,
            W=self.weight_init
        )
        l_conv2b = lasagne.layers.Conv2DLayer(
            l_conv2a,
            num_filters=64,
            filter_size=(3, 3),
            nonlinearity=self.nonlinearity,
            W=self.weight_init
        )
        l_pool2 = lasagne.layers.MaxPool2DLayer(l_conv2b, pool_size=(2, 2))

        l_conv3a = lasagne.layers.Conv2DLayer(
            l_pool2,
            num_filters=64,
            filter_size=(3, 3),
            nonlinearity=self.nonlinearity,
            W=self.weight_init
        )

        l_conv3b = lasagne.layers.Conv2DLayer(
            l_conv3a,
            num_filters=128,
            filter_size=(3, 3),
            nonlinearity=self.nonlinearity,
            W=self.weight_init
        )

        l_pool3 = lasagne.layers.MaxPool2DLayer(l_conv3b, pool_size=(2, 2))
        l_full4 = lasagne.layers.DenseLayer(
            l_pool3,
            num_units=self.n_out,
            nonlinearity=self.nonlinearity,
            W=self.weight_init
        )

        model = lasagne.layers.DenseLayer(
            l_full4,
            num_units=self.n_out,
            nonlinearity=self.nonlinearity,
            W=self.weight_init
        )

        return model

    # this model actually requires a different training procedure, of
    # recognition then verification
    def get_deep_id_net(self, input_layer, b_spatial=False):
        """Summary

        Parameters
        ----------
        input_layer : TYPE
            Description
        b_spatial : bool, optional
            Description

        Returns
        -------
        name : TYPE
            Description
        """
        l_conv1 = None
        # flip = False

        # returns a 20x52x44
        if b_spatial:
            l_conv1 = lasagne.layers.Conv2DLayer(
                self.get_spatial_transform_net(input_layer),
                num_filters=20,
                filter_size=(4, 4),
                stride=(1, 1),
                nonlinearity=self.nonlinearity,
                W=self.weight_init
            )

        else:
            l_conv1 = lasagne.layers.Conv2DLayer(
                input_layer,
                num_filters=20,
                filter_size=(4, 4),
                stride=(1, 1),
                nonlinearity=self.nonlinearity,
                W=self.weight_init
            )
        l_pool1 = lasagne.layers.MaxPool2DLayer(l_conv1, pool_size=(3, 3))

        l_conv2 = lasagne.layers.Conv2DLayer(
            l_pool1,
            num_filters=40,
            filter_size=(3, 3),
            stride=(1, 1),
            nonlinearity=self.nonlinearity,
            W=self.weight_init
        )
        l_pool2 = lasagne.layers.MaxPool2DLayer(l_conv2, pool_size=(3, 3))

        l_conv3 = lasagne.layers.Conv2DLayer(
            l_pool2,
            num_filters=60,
            filter_size=(3, 3),
            stride=(1, 1),
            nonlinearity=self.nonlinearity,
            W=self.weight_init
        )
        l_pool3 = lasagne.layers.MaxPool2DLayer(l_conv3, pool_size=(3, 3))

        l_conv4 = lasagne.layers.Conv2DLayer(
            l_pool3,
            num_filters=80,
            filter_size=(2, 2),
            stride=(1, 1),
            nonlinearity=self.nonlinearity,
            W=self.weight_init
        )

        model = lasagne.layers.DenseLayer(
            l_conv4,
            num_units=self.n_out,
            nonlinearity=self.nonlinearity,
            W=self.weight_init
        )

        return model

    def get_deep_id2_plus_net(self, input_layer, b_spatial=False):
        """Summary

        Parameters
        ----------
        input_layer : TYPE
            Description
        b_spatial : bool, optional
            Description

        Returns
        -------
        name : TYPE
            Description
        """
        l_conv1 = None
        # flip = False

        # returns a 20x52x44
        if b_spatial:
            l_conv1 = lasagne.layers.Conv2DLayer(
                self.get_spatial_transform_net(input_layer),
                num_filters=128,
                filter_size=(3, 4, 4),
                stride=(1, 1),
                nonlinearity=self.nonlinearity,
                W=self.weight_init
            )

        else:
            l_conv1 = lasagne.layers.Conv2DLayer(
                input_layer,
                num_filters=128,
                filter_size=(3, 4, 4),
                stride=(1, 1),
                nonlinearity=self.nonlinearity,
                W=self.weight_init
            )
        l_pool1 = lasagne.layers.MaxPool2DLayer(l_conv1, pool_size=(3, 3))

        l_conv2 = lasagne.layers.Conv2DLayer(
            l_pool1,
            num_filters=128,
            filter_size=(3, 3),
            stride=(1, 1),
            nonlinearity=self.nonlinearity,
            W=self.weight_init
        )
        l_pool2 = lasagne.layers.MaxPool2DLayer(l_conv2, pool_size=(3, 3))

        l_conv3 = lasagne.layers.Conv2DLayer(
            l_pool2,
            num_filters=128,
            filter_size=(3, 3),
            stride=(1, 1),
            nonlinearity=self.nonlinearity,
            W=self.weight_init
        )
        l_pool3 = lasagne.layers.MaxPool2DLayer(l_conv3, pool_size=(3, 3))

        l_conv4 = lasagne.layers.Conv2DLayer(
            l_pool3,
            num_filters=128,
            filter_size=(2, 2),
            stride=(1, 1),
            nonlinearity=self.nonlinearity,
            W=self.weight_init
        )

        model = lasagne.layers.DenseLayer(
            l_conv4,
            num_units=self.n_out,
            nonlinearity=self.nonlinearity,
            W=self.weight_init
        )

        return model

    def get_hani_2014_net(self, input_layer, dropout_pct=0.5, b_spatial=False):
        '''
        Return a lasagne network defining the siamese network in
        --------------------------------------------------------
        Khalil-Hani, M., & Sung, L. S. (2014). A convolutional neural
        network approach for face verification. High Performance Computing
        & Simulation (HPCS), 2014 International Conference on, (3), 707–714.
        doi:10.1109/HPCSim.2014.6903759

        Modifications
        -------------

        dropout_pct -- Instead of a fixed connection layer, use dropout with
        this much percentage [0.5]

        b_spatial -- Prepend a spatial transformer network which applies an
        affine transformation and a 2x crop [False]

        Args
        ----
        input_layer : TYPE
            Description
        dropout_pct : float, optional
            Description
        b_spatial : bool, optional
            Description

        Deleted Parameters
        ------------------
        input_layer (TYPE) : Description
        dropout_pct (float : Description
        optional), b_spatial (bool : Description
        '''

        # from lasagne.layers.corrmm import Conv2DMMLayer

        l_conv1 = None
        # flip = False

        if b_spatial:
            # returns a 5x21x21
            l_conv1 = lasagne.layers.Conv2DLayer(
                self.get_spatial_transform_net(input_layer),
                num_filters=5,
                filter_size=(6, 6),
                stride=(2, 2),
                nonlinearity=self.nonlinearity,
                # flip_filters=flip,
                W=self.weight_init
            )

        else:
            # returns a 5x21x21
            l_conv1 = lasagne.layers.Conv2DLayer(
                input_layer,
                num_filters=5,
                filter_size=(6, 6),
                stride=(2, 2),
                nonlinearity=self.nonlinearity,
                # flip_filters=flip,
                W=self.weight_init
            )

        # returns a 14x6x6
        l_conv2 = lasagne.layers.Conv2DLayer(
            l_conv1,
            num_filters=14,
            filter_size=(6, 6),
            stride=(2, 2),
            nonlinearity=self.nonlinearity,
            # flip_filters=flip,
            W=self.weight_init
        )

        l_dropout2 = lasagne.layers.DropoutLayer(l_conv2, p=dropout_pct)

        # returns a 60x1x1
        l_conv3 = lasagne.layers.Conv2DLayer(
            l_dropout2,
            num_filters=60,
            filter_size=(6, 6),
            nonlinearity=self.nonlinearity,
            # flip_filters=flip,
            W=self.weight_init
        )

        l_hidden = lasagne.layers.DenseLayer(
            l_conv3,
            num_units=self.n_out,
            nonlinearity=self.nonlinearity,
            W=self.weight_init
        )

        model = lasagne.layers.DenseLayer(
            l_hidden,
            num_units=self.n_out,
            nonlinearity=lasagne.nonlinearities.identity,
            W=lasagne.init.Uniform()
        )

        return model

    def build_model(self,
                    X_train,
                    y_train,
                    X_valid,
                    y_valid,
                    X_test,
                    y_test,
                    update=lasagne.updates.adam,
                    hyperparameter_margin=2.0,
                    hyperparameter_threshold=5.0,
                    learning_rate=0.0001):
        '''Given data for train, valid, and test, apply update with the
        given hyperparameters and learning rates, returning the models
        for train, valid, and test.

        Parameters
        ----------
        X_train : numpy.ndarray
            Example input data used to set network shape
        y_train : numpy.ndarray
            Example labels used to set network shape
        X_valid : numpy.ndarray
            Example input data used to set network shape
        y_valid : TYPE
            Example labels used to set network shape
        X_test : TYPE
            Example input data used to set network shape
        y_test : TYPE
            Example labels used to set network shape
        update : Attribute, optional
            Lasagne update rule to apply
        hyperparameter_margin : float, optional
            Total energy expected in the contrastive loss function
        hyperparameter_threshold : float, optional
            Simple thresholding of the final loss used for approximating the label
        learning_rate : float, optional
            How much to move in the gradient
        '''

        self.learning_rate = theano.shared(lasagne.utils.floatX(learning_rate))
        self.hyperparameter_threshold = lasagne.utils.floatX(
            hyperparameter_threshold)
        self.hyperparameter_margin = lasagne.utils.floatX(
            hyperparameter_margin)

        self.train_x = X_train
        self.validation_x = X_valid
        self.test_x = X_test

        self.update = update

        self.index = T.iscalar('index')
        self.batch_slice = slice(
            self.index * self.batch_size, (self.index + 1) * self.batch_size)

        # Training Loss
        y_pred = lasagne.layers.get_output(
            self.model, self.x, deterministic=False)
        avg_loss = self.loss_fn(y_pred, self.y, self.hyperparameter_margin)
        loss = avg_loss / self.batch_size

        # Validation Loss
        y_pred_eval = lasagne.layers.get_output(
            self.model, self.x, deterministic=True)
        avg_loss = self.loss_fn(
            y_pred_eval, self.y, self.hyperparameter_margin)
        loss_eval = avg_loss / self.batch_size
        # loss_eval = loss_eval.mean()

        # Validation Accuracy
        pred = self.distance_fn(y_pred_eval)
        accuracy = T.mean(T.eq(T.lt(pred, self.hyperparameter_threshold), self.y[
                          0::2]), dtype=theano.config.floatX)

        # Find weight params to update during backprop, and use adam updater
        all_params = lasagne.layers.get_all_params(self.model, trainable=True)
        updates = lasagne.updates.adam(
            loss, all_params, learning_rate=self.learning_rate)

        # Setup each model and return
        train_model = theano.function(
            [self.x, self.y], [loss, y_pred], updates=updates)
        validate_model = theano.function(
            [self.x, self.y], [loss_eval, accuracy, y_pred_eval])
        test_model = theano.function(
            [self.x, self.y], [loss_eval, accuracy, y_pred_eval])

        return train_model, validate_model, test_model

    def get_evaluation_model(self):
        """Return a theano function allowing you to directly compute
        the siamese net features.

        Returns
        -------
        fn : theano.function
            The theano function which expects the input layer and returns the
            siamese net features.  Does not require pairs (e.g. N can = 1).
        """
        y_pred = lasagne.layers.get_output(
            self.model, self.x, deterministic=True)
        fn = theano.function([self.x], [y_pred])
        return fn

    def retrieve_parameters(self):
        """Get stored parameters from the theano model.
        This function can be used in conjunction with set_from_parameters to save
        and restore model parameters.

        Returns
        -------
        model_parameters : list of numpy.array
            A list of numpy arrays representing the parameter values.
        """
        return lasagne.layers.get_all_param_values(self.model)

    def set_from_parameters(self, parameters):
        """Set the stored parameters of the internal theano model.
        This function can be used in conjunction with retrieve_parameters to save
        and restore model parameters.

        Parameters
        ----------
        parameters : list of numpy.array
            A list of numpy arrays representing the parameter values, must match
            the number of parameters.
            Every parameter's shape must match the shape of its new value.
        """
        lasagne.layers.set_all_param_values(self.model, parameters)

    def load_model(self, filename='model.pkl'):
        """Set the stored parameters of the internal theano model.
        This function can be used in conjunction with save_model to save
        and restore model parameters.

        Parameters
        ----------
        filename : str, optional
            Location of pickle file containing the model parameters.
        """
        params = pickle.load(open(filename, 'rb'))
        lasagne.layers.set_all_param_values(self.model, params)

    def save_model(self, filename='model.pkl'):
        """Get stored parameters from the theano model and store in the given filename.
        This function can be used in conjunction with load_model to save
        and restore model parameters.

        Parameters
        ----------
        filename : str, optional
            Location of pickle file containing the model parameters.
        """
        params = lasagne.layers.get_all_param_values(self.model)
        pickle.dump(params, open(filename, 'wb'))

    def get_learning_rate(self):
        return self.learning_rate.get_value()

    def set_learning_rate(self, lr):
        self.learning_rate.set_value(lasagne.utils.floatX(lr))


def distance_L2(x):
    """L2 distance for the Siamese architecture.

    Batches should be fed in pairs of images which the loss
    helps to optimize the distance of.  This is the siamese part of the architecture
    which fakes having two networks and just uses the batch's dimension to help
    define the parallel networks.

    Parameters
    ----------
    x : theano.tensor
        Tensor with pairs of batches

    Returns
    -------
    l2_dist : theano.tensor
        L2 Distance between pairs
    """
    x_a = x[0::2]
    x_b = x[1::2]
    return T.sum((x_a - x_b)**2, axis=1)


def distance_L1(x):
    """L1 distance for the Siamese architecture.

    Batches should be fed in pairs of images which the loss
    helps to optimize the distance of.  This is the siamese part of the architecture
    which fakes having two networks and just uses the batch's dimension to help
    define the parallel networks.

    Parameters
    ----------
    x : theano.tensor
        Tensor with pairs of batches

    Returns
    -------
    l1_dist : theano.tensor
        L1 Distance between pairs
    """
    x_a = x[0::2]
    x_b = x[1::2]
    return T.sum(T.abs_(x_a - x_b), axis=1)


def l2norm(x):
    """L2 norm.

    Parameters
    ----------
    x : theano.tensor
        Vector to take norm of.

    Returns
    -------
    l2_norm : theano.tensor
        L2 norm of vector in x
    """
    return T.sqrt(T.sum(T.sqr(x), axis=1))


def distance_cosine(x, e=1e-6):
    """Cosine distance for the Siamese architecture.

    Batches should be fed in pairs of images which the loss
    helps to optimize the distance of.  This is the siamese part of the architecture
    which fakes having two networks and just uses the batch's dimension to help
    define the parallel networks.

    Parameters
    ----------
    x : theano.tensor
        Description
    e : float, optional
        Epsilon to prevent divide by zero

    Returns
    -------
    distance : theano.tensor
        Cosine distance between pairs
    """
    x_a = x[0::2]
    x_b = x[1::2]
    return T.sum(x_a * x_b, axis=1) / T.maximum(l2norm(x_a) * l2norm(x_b), e)


# def contrastive_loss(y_pred, y_true, Q=20.0):
# eq. 8
#     E_w = distance_L1(y_pred)
#     y = y_true[0::2]
# eq 9
# Decrease energy for matched pair: (0 = unmatched, 1 = matched)
#     L_G = (1.0 - y) * (2.0 / Q) * (E_w ** 2)
#     L_I = (y) * 2.0 * Q * T.exp((-2.7726 * E_w) / Q)
#     L = L_G + L_I
#     avg_loss = T.mean(L)
#     return avg_loss


def contrastive_loss(y_pred, y_true, margin=20.0):
    """Contrastive loss for the Siamese Architecture.

    Batches should be fed in pairs of images which the loss helps to optimize
    the distance of.  This is the siamese part of the architecture which fakes
    having two networks and just uses the batch's dimension to help define the
    parallel networks.

    Parameters
    ----------
    y_pred : theano.tensor
        Predicted features (n_batch x n_features)
    y_true : theano.tensor
        Actual features (n_batch x n_features)
    margin : float, optional
        Hyperparameter defining total free energy

    Returns
    -------
    loss : theano.tensor
        Mean loss
    """
    x1 = y_pred[0::2]
    x2 = y_pred[1::2]

    d = T.sum((x1 - x2)**2, axis=1)
    y = y_true[0::2]
    return T.mean(y * d + (1.0 - y) * T.maximum(margin - d, 0.0))


def continue_siamese_net_training(filename):
    '''Continues the training of the siamese network with parameters defined
    in the given filename.

    Parameters
    ----------
    filename : string
        The path to the file defining the trained siamese network in progress.
    '''
    results_file = pickle.load(open(filename, 'rb'))
    params = results_file['params']
    print(params)
    run_siamese_net_training(
        dataset=params['dataset'],
        spatial=params['spatial_transform'],
        batch_size=params['batch_size'],
        n_out=params['n_features'],
        model_type=params['model_type'],
        n_epochs=params['n_epochs'],
        num_files=params['n_files'],
        learning_rate=params['learning_rate'],
        normalization=params['normalization'],
        crop_factor=params['crop'],
        resolution=params['resolution'][0],
        hyperparameter_margin=params['hyperparameter_margin'],
        hyperparameter_threshold=params['hyperparameter_threshold'],
        dropout_pct=params['dropout_pct'],
        nonlinearity=params['nonlinearity'],
        distance_fn=params['distance_fn'],
        b_convert_to_grayscale=params['b_convert_to_grayscale'],
        filename=filename + 'continued.pkl'
    )


def run_siamese_net_training(dataset,
                             spatial,
                             batch_size,
                             learning_rate,
                             model_type,
                             n_epochs,
                             n_out,
                             num_files,
                             normalization,
                             resolution,
                             crop_factor,
                             hyperparameter_margin,
                             hyperparameter_threshold,
                             nonlinearity,
                             distance_fn,
                             b_convert_to_grayscale,
                             filename=None,
                             path_to_data=None,
                             b_load_idxs_only=True):
    '''Run training of a siamese net for the given parameters, saving
    results to a pickle file defined by the given parameters:

    Parameters
    ----------
    dataset : string
        Name of the dataset to use: 'lfw', 'olivetti'
    spatial : bool
        Whether to prepent a spatial transformer network or not
    batch_size : int
        Number of observations in a batch
    learning_rate : float
        Learning Rate
    model_type : string
        Which model to use: 'hani', 'chopra', or 'custom'
    n_epochs : Integer
        Number of epochs to train for.
    n_out : int
        Number of neurons in the final output layer of the Siamese Network
    num_files : int
        Number of files to load for each person
    normalization : string
        Method of normalization to apply: '-1:1', 'LCN', 'LCN-', 'ZCA'
    resolution : int
        Image resolution to scale to (square pixels only)
    crop_factor : float
        Factor to scale bounds of the detected face.
        1.0 means the face is tightly cropped,
        < 1.0, the face is cropped even tighter
        > 1.0, more of the outside of the face is included.
    hyperparameter_margin : float
        Total free energy of the contrastive loss equation
    hyperparameter_threshold : float
        Threshold to apply to L1 norm of final output layers defining
        whether faces match or not
    nonlinearity : string
        "rectify" or "scaled_tanh"
    distance_fn : string
        "L1", "L2", or "Cosine"
    b_convert_to_grayscale : bool
        Color images are automatically converted to grayscale (C = 1)
    filename : str, optional
        Where to store results
    path_to_data : string
        Where to find the dataset (defaults to current working directory)
    b_load_idxs_only : bool
        If False, the entire dataset's pairs are loaded into memory
        Advised to load idxs only for lfw as it requires > 60 GB.

    Deleted Parameters
    ------------------
    bCropToHaarBBox : bool
        Crop images to frontal face cascade
    '''

    if filename is None:
        filename = str('dataset_%s' % dataset +
                       '_transform_%d' % int(spatial) +
                       '_batch_%d' % batch_size +
                       '_lr_%f' % learning_rate +
                       '_model_%s' % model_type +
                       '_epochs_%d' % n_epochs +
                       '_normalization_%s' % normalization +
                       '_cropfactor_%0.02f' % crop_factor +
                       '_nout_%d' % n_out +
                       '_resolution_%d' % resolution +
                       '_numfiles_%d' % num_files +
                       '_q_%2.02f' % hyperparameter_margin +
                       '_t_%2.02f' % hyperparameter_threshold +
                       '_nonlinearity_%s' % nonlinearity +
                       '_distancefn_%s' % distance_fn +
                       '_grayscale_%d.pkl' % b_convert_to_grayscale)
        filename = os.path.join('results', filename)

    results = None
    model = None
    if os.path.isfile(filename):
        try:
            results = pickle.load(open(filename, 'rb'))
            if 'epochs' in results.keys():
                if len(results['epochs']) >= n_epochs:
                    print('Already process(ing/ed); exiting.')
                    return
                # else:
                # continue where it left off
                #     if 'model' in results.keys():
                #         model = pickle.loads(results['model'])
                #         model.set_from_parameters(pickle.loads(results['model_parameters']))
        except:
            pass

    print("""Dataset: %s
        \rSpatial: %d
        \rBatch Size: %d
        \rNum Features: %d
        \rModel Type: %s
        \rNum Epochs: %d
        \rNum Files: %d
        \rLearning Rate: %f
        \rNormalization: %s
        \rCrop Factor: %f
        \rResolution: %d
        \rHyperparameter Margin: %f
        \rHyperparameter Threshold: %f
        \rNon-Linearity: %s
        \rGrayscale: %d
        \rDistance Function: %s\n
        \rWriting results to: %s\n""" % (dataset,
                                         int(spatial),
                                         batch_size,
                                         n_out,
                                         model_type,
                                         n_epochs,
                                         num_files,
                                         learning_rate,
                                         normalization,
                                         crop_factor,
                                         resolution,
                                         hyperparameter_margin,
                                         hyperparameter_threshold,
                                         nonlinearity,
                                         int(b_convert_to_grayscale),
                                         distance_fn,
                                         filename))

    if model_type == 'deepid':
        b_convert_to_grayscale = False

    if b_convert_to_grayscale:
        input_channels = 1
    else:
        input_channels = 3

    # TODO: if continuing a result from a left off epoch, the dataset will
    # have been generated differently. how should I handle this?  store the
    # pairs, too big a file?  store the rng, what about parameters?
    print('Loading dataset...')
    data = load_pairs(
        dataset=dataset,
        normalization=normalization,
        resolution=(resolution, resolution),
        split=(0.8, 0.2, 0.2),
        crop_factor=1.2,
        n_files_per_person=num_files,
        path_to_data=path_to_data,
        b_load_idxs_only=b_load_idxs_only,
        b_convert_to_grayscale=b_convert_to_grayscale)

    print('Initializing Siamese Network...')
    print(data['X'].shape)
    X_train = np.zeros(np.hstack((batch_size, data['X'].shape[1:])))
    y_train = np.zeros(np.hstack((batch_size, data['y'].shape[1:])))

    if model is None:
        model = ConvSiameseNet(input_channels=input_channels,
                               input_width=X_train.shape[2],
                               input_height=X_train.shape[3],
                               n_out=n_out,
                               batch_size=batch_size,
                               nonlinearity=nonlinearity,
                               distance_fn=distance_fn)

        if model_type == 'hani':
            model.use_hani_model(dropout_pct=0.0, b_spatial=spatial)
        elif model_type == 'custom':
            model.use_custom_model(b_spatial=spatial)
        elif model_type == 'chopra':
            model.use_chopra_model(
                dropout_pct=0.0, b_spatial=spatial)
        elif model_type == 'deepid':
            model.use_deepid_model(b_spatial=spatial)
        else:
            print(
                'Unrecognized model type! Choose between \'hani\', \'chopra\', or \'custom\'')
            sys.exit(2)

    train_model, validate_model, test_model = model.build_model(
        X_train, y_train,
        X_train, y_train,
        X_train, y_train,
        hyperparameter_margin=hyperparameter_margin,
        hyperparameter_threshold=hyperparameter_threshold,
        learning_rate=learning_rate
    )
    if results is None:
        results = {
            'params':
            {
                'dataset': dataset,
                'spatial_transform': spatial,
                'batch_size': batch_size,
                'n_features': n_out,
                'model_type': model_type,
                'n_epochs': n_epochs,
                'n_files': num_files,
                'learning_rate': learning_rate,
                'normalization': normalization,
                'crop': crop_factor,
                'resolution': (resolution, resolution),
                'hyperparameter_margin': hyperparameter_margin,
                'hyperparameter_threshold': hyperparameter_threshold,
                'nonlinearity': nonlinearity,
                'distance_fn': distance_fn,
                'b_convert_to_grayscale': b_convert_to_grayscale
            },
            'epochs': [],
            'prediction':
            {
                'X': None,
                'y': None,
                'imgs': None,
                'auc': [],
                'F1': [],
                'log_loss': [],
                'W': []
            },
            'model': None,
            'model_parameters': None
        }

    delta_loss = 1.0
    epoch = len(results['epochs'])
    prev_loss = 0
    while delta_loss > 1e-6 and epoch < n_epochs:

        # Training
        clf = LogisticRegression()
        X_train, y_train = np.zeros((0, n_out * 2)), []
        X_test, y_test = np.zeros((0, n_out * 2)), []

        train_err = 0
        train_batches = 0
        start_time = time.time()
        if b_load_idxs_only:
            for X, y in generate_new_dataset_batch_from_idxs(
                data['X'],
                data['y'],
                data['X_train_matched_idxs'],
                data['y_train_matched_idxs'],
                data['X_train_unmatched_idxs'],
                data['y_train_unmatched_idxs'],
                batch_size
            ):
                err, y_pred = train_model(X, y)
                X_train = np.r_[
                    (X_train, np.reshape(y_pred, (batch_size / 2, n_out * 2)))]
                y_train = np.r_[(y_train, y[::2])]
                train_batches += 1
                train_err += err
        else:
            for X, y in generate_new_dataset_batch(
                data['X_train_matched'],
                data['y_train_matched'],
                data['X_train_unmatched'],
                data['y_train_unmatched'],
                batch_size
            ):
                err, y_pred = train_model(X, y)
                X_train = np.r_[
                    (X_train, np.reshape(y_pred, (batch_size / 2, n_out * 2)))]
                y_train = np.r_[(y_train, y[::2])]
                train_batches += 1
                train_err += err
        # Validation
        val_err = 0
        val_acc = 0
        val_batches = 0

        if b_load_idxs_only:
            for X, y in generate_new_dataset_batch_from_idxs(
                data['X'],
                data['y'],
                data['X_valid_matched_idxs'],
                data['y_valid_matched_idxs'],
                data['X_valid_unmatched_idxs'],
                data['y_valid_unmatched_idxs'],
                batch_size
            ):
                err, acc, y_pred = validate_model(X, y)
                X_test = np.r_[
                    (X_test, np.reshape(y_pred, (batch_size / 2, n_out * 2)))]
                y_test = np.r_[(y_test, y[::2])]
                val_err += err
                val_acc += acc
                val_batches += 1
        else:
            for X, y in generate_new_dataset_batch(
                data['X_valid_matched'],
                data['y_valid_matched'],
                data['X_valid_unmatched'],
                data['y_valid_unmatched'],
                batch_size
            ):
                err, acc, y_pred = validate_model(X, y)
                X_test = np.r_[
                    (X_test, np.reshape(y_pred, (batch_size / 2, n_out * 2)))]
                y_test = np.r_[(y_test, y[::2])]
                val_err += err
                val_acc += acc
                val_batches += 1

        # Measure Performance
        X_train_L1 = np.abs(X_train[:, :n_out] - X_train[:, n_out:])
        X_test_L1 = np.abs(X_test[:, :n_out] - X_test[:, n_out:])
        clf.fit(X_train_L1, y_train)
        y_true, y_pred = y_test, clf.predict(X_test_L1)
        auc = metrics.roc_auc_score(y_true, y_pred)
        f1 = metrics.f1_score(y_true, y_pred)

        # Report
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, n_epochs, time.time() - start_time))
        print("\ttraining loss:\t\t\t{:.6f}".format(train_err / train_batches))
        print("\tvalidation loss:\t\t{:.6f}".format(val_err / val_batches))
        print("\tvalidation AUC:\t\t\t{:.2f}".format(auc))
        print("\tvalidation F1:\t\t\t{:.2f}".format(f1))

        if (prev_loss - train_err / train_batches) < 0:
            model.set_learning_rate(model.get_learning_rate() * 0.5)

        delta_loss = np.abs(prev_loss - train_err / train_batches)
        prev_loss = train_err / train_batches

        results['epochs'].append(
            {
                'epoch': epoch,
                'training_loss': train_err / train_batches,
                'validation_loss': val_err / val_batches,
                'accuracy': val_acc / val_batches * 100,
                'f1': f1,
                'auc': auc
            }
        )
        # results['model'] = pickle.dumps(model)
        results['model_parameters'] = pickle.dumps(model.retrieve_parameters())

        pickle.dump(results, open(filename, 'wb'))
        epoch = epoch + 1

    # Now train a logistic regression classifer which will use the
    # siamese network embedding of an image
    # First generate stacked features from image pairs which will be used as
    # input features
    Xs, ys, imgs = np.zeros(
        (0, n_out * 2)), [], np.zeros((0, 1, data['X'].shape[2], data['X'].shape[3]))
    if b_load_idxs_only:
        for split in ['train', 'valid', 'test']:
            for X, y in generate_new_dataset_batch_from_idxs(
                data['X'],
                data['y'],
                data['X_' + split + '_matched_idxs'],
                data['y_' + split + '_matched_idxs'],
                data['X_' + split + '_unmatched_idxs'],
                data['y_' + split + '_unmatched_idxs'],
                batch_size
            ):
                err, acc, pred = test_model(X, y)
                Xs = np.r_[(Xs, np.reshape(pred, (batch_size / 2, n_out * 2)))]
                ys = np.r_[(ys, y[::2])]
                # imgs = np.r_[(X, imgs)]
    else:
        for split in ['train', 'valid', 'test']:
            for X, y in generate_new_dataset_batch(
                data['X_' + split + '_matched'],
                data['y_' + split + '_matched'],
                data['X_' + split + '_unmatched'],
                data['y_' + split + '_unmatched'],
                batch_size
            ):
                err, acc, pred = test_model(X, y)
                Xs = np.r_[(Xs, np.reshape(pred, (batch_size / 2, n_out * 2)))]
                ys = np.r_[(ys, y[::2])]
                # imgs = np.r_[(X, imgs)]

    results['prediction']['X'] = Xs
    results['prediction']['y'] = ys
    # results['prediction']['imgs'] = imgs

    Xs_L1 = np.abs(Xs[:, :n_out] - Xs[:, n_out:])

    skf = cross_validation.StratifiedKFold(ys, n_folds=4, random_state=0)
    for train_index, test_index in skf:
        X_train, y_train = Xs_L1[train_index], ys[train_index]
        X_test, y_test = Xs_L1[test_index], ys[test_index]

        # if False:
        #     scaler = StandardScaler()
        #     scaler.fit(X_train)
        #     X_train = scaler.transform(X_train)
        #     X_test = scaler.transform(X_test)

        clf = LogisticRegression()
        clf.fit(X_train, y_train)
        y_true, y_pred = y_test, clf.predict(X_test)
        # collect a few metrics:
        results['prediction']['auc'].append(
            metrics.roc_auc_score(y_true, y_pred))
        results['prediction']['F1'].append(metrics.f1_score(y_true, y_pred))
        results['prediction']['log_loss'].append(
            metrics.log_loss(y_true, y_pred))

        # store the resulting model
        results['prediction']['W'].append(clf.coef_)

    print('AUC: ', np.mean(results['prediction']['auc']))
    print('F1: ', np.mean(results['prediction']['F1']))
    results['model'] = pickle.dumps(model)
    results['model_parameters'] = pickle.dumps(model.retrieve_parameters())

    pickle.dump(results, open(filename, 'wb'))


def main(argv):
    """Summary

    Parameters
    ----------
    argv : TYPE
        Description

    Returns
    -------
    name : TYPE
        Description
    """
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    # parser.add_argument('-h', '--help', help='Display help.', action=printUsage())
    parser.add_argument('-m', '--model_type',
                        help='Choose the Deep Network to use. ["hani"], "chopra", or "custom"',
                        default='hani', dest='model_type')
    parser.add_argument('-of', '--output_features',
                        help='Number of features in the final siamese network layer',
                        default=40, dest='n_out')
    parser.add_argument('-bs', '--batch_size',
                        help='Number of observations per batch.',
                        default=100, dest='batch_size')
    parser.add_argument('-e', '--epochs',
                        help='Number of epochs to train for.',
                        default=5, dest='n_epochs')
    parser.add_argument('-lr', '--learning_rate',
                        help='Initial learning rate to apply to the gradient update.',
                        default=1e-4, dest='learning_rate')
    parser.add_argument('-norm', '--normalization',
                        help='Normalization of the dataset using either ["-1:1"], "LCN", "LCN-", or "ZCA".',
                        default='-1:1', dest='normalization')
    parser.add_argument('-f', '--filename',
                        help='Resulting pickle file to store results. If none is given, a filename is created based on the combination of all parameters.',
                        default=None, dest='filename')
    parser.add_argument('-path', '--path_to_data',
                        help='Path to the dataset.  If none is given it is assumed to be in the current working directory',
                        default=None, dest='path_to_data')
    parser.add_argument('-hm', '--hyperparameter_margin',
                        help='Contrastive Loss parameter describing the total free energy.',
                        default=2.0, dest='hyperparameter_margin')
    parser.add_argument('-ht', '--hyperparameter_threshold',
                        help='Threshold to apply to the difference in the final output layer.',
                        default=5.0, dest='hyperparameter_threshold')
    parser.add_argument('-ds', '--dataset',
                        help='The dataset to train/test with. Choose from ["lfw"], or "olivetti"',
                        default='lfw', dest='dataset')
    parser.add_argument('-nl', '--nonlinearity',
                        help='Non-linearity to apply to convolution layers.',
                        default='rectify', dest='nonlinearity')
    parser.add_argument('-fn', '--distance_fn',
                        help='Distance function to apply to final siamese layer.',
                        default='l2', dest='distance_fn')
    parser.add_argument('-cf', '--cropfactor',
                        help='Scale factor of amount of image around the face to use.',
                        default=1.0, dest='crop_factor')
    parser.add_argument('-sp', '--spatial_transform',
                        help='Whether or not to prepend a spatial transform network',
                        default=False, dest='spatial')
    parser.add_argument('-r', '--resolution',
                        help='Rescale images to this fixed square pixel resolution (e.g. 64 will mean images, after any crops, are rescaled to 64 x 64). ',
                        default=64, dest='resolution')
    parser.add_argument('-nf', '--num_files',
                        help='Number of files to load for each person.',
                        default=2, dest='num_files')
    parser.add_argument('-gray', '--grayscale',
                        help='Convert images to grayscale.',
                        default=True, dest='b_convert_to_grayscale')

    args = parser.parse_args()
    print(args)

    run_siamese_net_training(dataset=args.dataset,
                             spatial=args.spatial,
                             batch_size=int(args.batch_size),
                             learning_rate=float(args.learning_rate),
                             model_type=args.model_type,
                             n_epochs=int(args.n_epochs),
                             n_out=int(args.n_out),
                             crop_factor=float(args.crop_factor),
                             num_files=int(args.num_files),
                             resolution=int(args.resolution),
                             normalization=args.normalization,
                             nonlinearity=args.nonlinearity,
                             distance_fn=args.distance_fn,
                             b_convert_to_grayscale=int(
                                 args.b_convert_to_grayscale),
                             hyperparameter_margin=float(
                                 args.hyperparameter_margin),
                             hyperparameter_threshold=float(
                                 args.hyperparameter_threshold),
                             path_to_data=args.path_to_data,
                             filename=args.filename)


if __name__ == "__main__":
    main(sys.argv[1:])
