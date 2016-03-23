"""
Siamese Network for performing training of a Deep Convolutional
Network for Face Verification on the Olivetti, LFW, and Kadenze
Faces datasets.

Parag K. Mital. Copyright Kadenze, Inc. 2015.

Dependencies
------------
numpy, sklearn, scipy, theano, lasagne, pickle, cv2, dlib

Part of the package siamese_net:
siamese_net/
siamese_net/faces.py
siamese_net/datasets.py
siamese_net/normalization.py

Loading Olivetti, LFW, and Kadenze Faces datasets with optional preprocessing steps.

Copyright 2016 Kadenze, Inc.
Kadenze(R) and Kannu(R) are Registered Trademarks
of Kadenze, Inc.


Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

Apache License

Version 2.0, January 2004

http://www.apache.org/licenses/
"""
import pickle
import os
import numpy as np
import scipy as sp
import cv2
from bson.binary import Binary


def scale_crop(x, y, w, h, img_w, img_h, crop_factor=1.0):
    '''scale_crop(x, y, w, h, img_w, img_h, crop_factor=1.0)

    Utility to take a bbox and image bounds and scale the bbox by crop_factor

    Parameters
    ----------
    x : float
        x position of bbox
    y : float
        y position of bbox
    w : float
        width of bbox
    h : float
        height of bbox
    img_w : float
        Image width, used for defining the maximum extent of the bbox
    img_h : float
        Image height, used for defining the maximum extent of the bbox
    crop_factor : float
        x position of bbox

    Returns
    -------
    x, y, w, h : float, float, float, float
        Scaled bounding box
    '''
    w = w - (x - crop_factor * x) * 2
    h = h - (y - crop_factor * y) * 2
    x = x + x - crop_factor * x
    y = y + y - crop_factor * y

    if (x + w) > img_w:
        x -= ((x + w) - img_w)
    if (y + h) > img_h:
        y -= ((y + h) - img_h)

    x = np.clip(x, 0, img_w - w)
    y = np.clip(y, 0, img_h - h)

    # w = np.clip(w, 0, img_w)
    # h = np.clip(h, 0, img_h)
    return x, y, w, h


class Datasets(object):

    """Summary

    Attributes
    ----------
    area_threshold : TYPE
        Description
    b_augment_w_affine : TYPE
        Description
    b_augment_w_flips : TYPE
        Description
    b_convert_img_to_serializable : TYPE
        Description
    b_convert_to_grayscale : TYPE
        Description
    bCropToHaar : TYPE
        Description
    bCropToHOG : TYPE
        Description
    clm : TYPE
        Description
    crop_factor : TYPE
        Description
    crop_style : TYPE
        Description
    detector : TYPE
        Description
    face_cascade : TYPE
        Description
    model : TYPE
        Description
    n_files_per_person : int
        Description
    n_min_files_per_person : TYPE
        Description
    n_warps : int
        Description
    path_to_data : TYPE
        Description
    resolution : TYPE
        Description
    similar_img_threshold : TYPE
        Description
    """

    def __init__(self,
                 path_to_data=None,
                 n_min_files_per_person=None,
                 n_files_per_person=None,
                 b_convert_to_grayscale=False,
                 crop_style='none',
                 crop_factor=1.0,
                 resolution=(64, 64),
                 b_convert_img_to_serializable=False,
                 b_augment_w_flips=True,
                 b_augment_w_affine=True,
                 area_threshold=1.0,
                 similar_img_threshold=0.1):
        '''Create a Dataset from various face datasets for train/test
            with various options.

        Parameters
        ----------
        path_to_data : TYPE, optional
            Description
        n_min_files_per_person : TYPE, optional
            Description
        n_files_per_person : TYPE, optional
            Description
        b_convert_to_grayscale : bool, optional
            Description
        crop_style : string
            Either ['trees'], 'hog+clm', 'haar+clm', 'haar', or 'hog'.
            Trees will use HOG to localize the face then use regression
            trees to fit shape model.  CLM will use HOG or Haar to localize
            the face, then find the shape points, and crop to the shape
            points.
        crop_factor : float
            percentage of the cropped image to use.  1.0 is unchanged,
            <1.0 is less of the crop, >1.0 is more of the image included.
        resolution : (int, int)
            The final image resolution to be resized to.
        b_convert_img_to_serializable : bool
            Image will be converted to a serializable representation (e.g.
                useful for adding to a database like MongoDB)
        b_augment_w_flips : bool, optional
            Description
        b_augment_w_affine : bool, optional
            Description
        area_threshold : float
            Reject images above this percent of the entire image (meaning
                it is likely not a face),  1.0 means nothing will be rejected.
        similar_img_threshold : float
            The minimum L1 distance another image needs to have to all other
            images of the same label to be added to the dataset

        Raises
        ------
        ValueError
            Description
        '''

        self.path_to_data = path_to_data
        self.n_min_files_per_person = n_min_files_per_person
        if n_files_per_person is None:
            self.n_files_per_person = 2
        else:
            self.n_files_per_person = n_files_per_person
        self.b_convert_to_grayscale = b_convert_to_grayscale
        self.crop_style = crop_style if crop_factor != 1.0 else 'none'
        self.crop_factor = crop_factor
        self.resolution = resolution
        self.b_convert_img_to_serializable = b_convert_img_to_serializable
        self.similar_img_threshold = similar_img_threshold
        self.bCropToHOG = 'hog' in self.crop_style
        self.bCropToHaar = 'haar' in self.crop_style
        self.b_augment_w_flips = b_augment_w_flips
        self.b_augment_w_affine = b_augment_w_affine
        self.n_warps = 20
        self.area_threshold = area_threshold

        if crop_style == 'haar':
            self.face_cascade = cv2.CascadeClassifier(
                'haarcascade_frontalface_default.xml')
        elif crop_style == 'hog':
            import dlib
            self.detector = dlib.get_frontal_face_detector()
        elif 'clm' in crop_style:
            import menpo_clm as mp
            self.clm = mp.CLM()
        elif 'trees' in crop_style:
            import faces as f
            self.model = f.FaceShapeModel()
        elif crop_style == 'none':
            pass
        else:
            raise ValueError(
                'Unknown crop style.  Known options: ["trees"], "hog+clm", "haar+clm", "hog", and "haar"')

    def get_parsed_dataset(self, dataset='lfw', path_to_data=None):
        '''Returns the requested dataset, either 'lfw' or 'kadenze'

        Parameters
        ----------
        dataset : string
            Choose from 'lfw' or 'kadenze'
        path_to_data : string
            Where to find the data?  Default of None will search the current
            working directory.

        Returns
        -------
        ds : {
          'images' : numpy.ndarray((num_images, num_channels, width, height)),
          'target' : numpy.ndarray((num_images,))
        }
        '''

        filename = None

        filename = 'ds-{ds}_files-{n_files_per_person}_crop_style-{crop_style}_flips-{b_augment_w_flips}_warps-{b_augment_w_affine}_crop_factor-{crop_factor}_resolution-{x}x{y}_grayscale-{grayscale}.pkl'.format(
            ds=dataset,
            n_files_per_person=self.n_files_per_person,
            crop_style=self.crop_style,
            crop_factor=self.crop_factor,
            x=self.resolution[0],
            y=self.resolution[1],
            grayscale=self.b_convert_to_grayscale,
            b_augment_w_affine=self.b_augment_w_affine,
            b_augment_w_flips=self.b_augment_w_flips)

        # Check if we already have processed this dataset into a pickle file
        ds = None
        if os.path.exists(filename):
            try:
                with open(filename, 'rb') as f:
                    ds = pickle.load(f)
            except:
                pass

        if ds is None:
            print('Preprocessing dataset')
            # nope, load it
            if dataset == 'lfw':
                ds = self._get_lfw_dataset(path_to_data=path_to_data)
            else:
                ds = self._get_kad_dataset(path_to_data=path_to_data)

            # keep it cached as a pickle file so we don't have to load it again
            try:
                with open(filename, 'wb') as f:
                    pickle.dump(ds, f, pickle.HIGHEST_PROTOCOL)
            except:
                os.remove(filename)
                print('Could not pickle!')

        # Check that we have the right number of images per person
        labels = [d['y'] for d in ds]
        good_idxs = None
        n_bins = len(np.unique(labels))
        if self.n_files_per_person is None:
            good_idxs = range(n_bins)
        else:
            hist, edges = np.histogram(labels, bins=n_bins)
            good_idxs = np.where(hist >= self.n_files_per_person)[0]

        # Randomly permute the images and select the right number of images
        images = []
        labels = []
        for idx in good_idxs:
            this_imgs = np.concatenate([d['X'] for d in ds if idx == d['y']])
            this_lbls = np.array([d['y'] for d in ds if idx == d['y']])
            indices = None
            if self.n_files_per_person is None:
                indices = np.random.permutation(len(this_lbls))
            else:
                indices = np.random.permutation(
                    len(this_lbls))[:np.min((
                        self.n_files_per_person,
                        len(this_lbls)
                    ))]
            images.append(this_imgs[indices, ...])
            labels.append(this_lbls[indices, ...])

        # Return dict in same format as olivetti dataset
        return {'images': np.concatenate(images), 'target': np.hstack(labels)}

    def _get_lfw_dataset(self, path_to_data=None):
        """Summary

        Parameters
        ----------
        path_to_data : TYPE, optional
            Description

        Returns
        -------
        name : TYPE
            Description
        """
        import os
        if self.path_to_data is None:
            path_to_data = os.path.join(os.getcwd(), 'lfw')
        if os.path.exists(path_to_data):
            return self._get_one_level_deep(
                path_to_data=path_to_data, token1='', token2='.jpg'
            )
        else:
            print('Cannot find dataset at: %s' % (path_to_data))
            print(
                'Downloading dataset from http://vis-www.cs.umass.edu/lfw/lfw.tgz...')
            os.system('wget http://vis-www.cs.umass.edu/lfw/lfw.tgz')
            print('Extracting dataset...')
            os.system('tar -xvf lfw.tgz')
            return self._get_one_level_deep(
                path_to_data='./lfw', token1='', token2='.jpg'
            )

    def _get_kad_dataset(self, path_to_data=None):
        """Summary

        Parameters
        ----------
        path_to_data : TYPE, optional
            Description

        Returns
        -------
        name : TYPE
            Description
        """
        import os
        if self.path_to_data is None:
            path_to_data = os.path.join(os.getcwd(), 'kadenze_faces')
        return self._get_two_level_deep(
            path_to_data=path_to_data, token1='captured', token2='.jpg'
        )

    def _augment_img(self, img):
        """Summary

        Parameters
        ----------
        img : TYPE
            Description

        Returns
        -------
        name : TYPE
            Description
        """
        ds = []
        ds.append(img)

        if self.b_convert_to_grayscale:
            orig_img = np.squeeze(img['X'])
        else:
            orig_img = np.rollaxis(np.squeeze(img['X']), 0, 3)

        img_flipped = img.copy()

        if self.b_augment_w_flips:

            if self.b_convert_to_grayscale:
                x_aug = cv2.flip(orig_img, flipCode=1)
                img_flipped['X'] = x_aug[np.newaxis, ...]
            else:
                x_aug = np.concatenate((
                    cv2.flip(orig_img[:, :, 0], flipCode=1)[..., np.newaxis],
                    cv2.flip(orig_img[:, :, 1], flipCode=1)[..., np.newaxis],
                    cv2.flip(orig_img[:, :, 2], flipCode=1)[..., np.newaxis]
                ), axis=2)
                img_flipped['X'] = np.rollaxis(x_aug, -1)[np.newaxis, ...]

            ds.append(img_flipped)

        if self.b_augment_w_affine:

            for i in range(self.n_warps):

                img_warped = img.copy()

                # rotate w/ N(0, 2.0) degrees and scale w/ N(1.0, 0.067)
                # percent of the image
                M = cv2.getRotationMatrix2D(
                    (img['X'].shape[2] / 2.0, img['X'].shape[1] / 2.0),
                    angle=np.random.normal(0.0, 2.0),
                    scale=np.random.normal(1.0, 0.067)
                )
                img_warped['X'] = cv2.warpAffine(
                    orig_img,
                    M,
                    dsize=(orig_img.shape[1], orig_img.shape[0]),
                    borderMode=cv2.BORDER_WRAP
                )

                # translate w/ N(1.0, 0.067) percent of the image
                M = np.array([[np.random.normal(1.0, 0.067), 0.0, 0.0], [
                             0.0, np.random.normal(1.0, 0.067), 0.0]])
                img_warped['X'] = cv2.warpAffine(
                    img_warped['X'],
                    M, dsize=(orig_img.shape[1], orig_img.shape[0]),
                    borderMode=cv2.BORDER_WRAP
                )

                if self.b_convert_to_grayscale:
                    img_warped['X'] = img_warped['X'][np.newaxis, ...]
                else:
                    img_warped['X'] = np.rollaxis(
                        img_warped['X'], -1)[np.newaxis, ...]

                ds.append(img_warped)
        return ds

    def _get_one_level_deep(self, path_to_data, token1='', token2='.jpg'):
        '''Return the unparsed original LFW dataset as a listof dictionaries 
        defining `X`, the image, `y` the label as a numeric integer unique to
        each person, `str_y`, the path of the person's image data, `shape` 
        the size of the image, and `filename`, the original file's name.

        Note this function could be used to load any dataset that is 
        similarly structured:
        e.g.:

        /path_to_data
        /path_to_data/person1
        /path_to_data/person1/image1.jpg
        /path_to_data/person1/image2.jpg
        /path_to_data/person1/image3.jpg
        ...
        /path_to_data/person2
        /path_to_data/person2/image1.jpg
        /path_to_data/person2/image2.jpg
        /path_to_data/person2/image3.jpg
        ...
        /path_to_data/person3
        ...

        Parameters
        ----------
        path_to_data : string
            Location of the dataset.  Defaults to current working directory.
        token1 : string
            Token required to add the file.  Empty string passes all files.
        token2 : string
            Token required to add the file.  Empty string passes all files.
        '''

        print('Loading data in %s' % path_to_data)
        ds = []
        dirs = [direc for direc in os.listdir(path_to_data) if os.path.isdir(
            os.path.join(path_to_data, direc))]
        # Randomize directories
        indices = np.random.permutation(len(dirs))
        for label_i, direc in enumerate([dirs[dir_i] for dir_i in indices]):
            print('Person: {current}/{total}'.format(
                current=label_i, total=len(dirs)), end='\r')
            files = os.listdir(os.path.join(path_to_data, direc))
            good_files = [
                file_i for file_i in files
                if token1 in file_i and
                token2 in file_i
            ]
            for file_i, filename in enumerate(good_files):
                img = self._preprocess_img(
                    file_i=filename,
                    label_i=label_i,
                    pathToFile=os.path.join(path_to_data, direc),
                    ds=ds
                )
                if img is not None:
                    for img_augmented in self._augment_img(img):
                        ds.append(img_augmented)
        return ds

    def _get_two_level_deep(self, path_to_data=None, token1='', token2='.jpg'):
        '''Return the unparsed original Kadenze dataset as a listof dictionaries
        defining `X`, the image, `y` the label as a numeric integer unique to
        each person, `str_y`, the path of the person's image data,
        `shape` the size of the image, and `filename`, the original file's name

        Parameters
        ----------
        path_to_data : string
            Location of the dataset.  Defaults to current working directory.
        token1 : string
            Token required to add the file.  Empty string passes all files.
        token2 : string
            Token required to add the file.  Empty string passes all files.
        '''
        import os
        if path_to_data is None:
            path_to_data = os.path.join(os.getcwd(), 'kadenze_faces')
        print('Loading data in %s' % path_to_data)
        ds = []
        dirs = [direc for direc in os.listdir(path_to_data) if os.path.isdir(
            os.path.join(path_to_data, direc))]
        for label_i, direc in enumerate(dirs):
            print(
                'Person: {current}/{total}'.format(
                    current=label_i, total=len(dirs)
                ),
                end='\r'
            )
            sub_direcs = [
                subdirec
                for subdirec in os.listdir(
                    os.path.join(path_to_data, direc)
                )
                if os.path.isdir(
                    os.path.join(
                        os.path.join(path_to_data, direc), subdirec)
                )
            ]
            for pair in sub_direcs:
                files = os.listdir(
                    os.path.join(os.path.join(path_to_data, direc), pair))
                good_files = [
                    file_i for file_i in files
                    if token1 in file_i and token2 in file_i
                ]
                if self.n_min_files_per_person is None:
                    this_n_files = len(good_files)
                else:
                    this_n_files = np.min(
                        (len(good_files), self.n_min_files_per_person))
                if len(good_files) >= this_n_files:
                    for file_i, filename in enumerate(good_files):
                        img = self._preprocess_img(
                            file_i=filename,
                            label_i=label_i,
                            pathToFile=os.path.join(
                                os.path.join(path_to_data, direc), pair),
                            ds=ds
                        )
                        if img is not None:
                            for img_augmented in self._augment_img(img):
                                ds.append(img_augmented)
        return ds

    def _preprocess_img(self,
                        file_i,
                        label_i,
                        pathToFile,
                        ds=None
                        ):
        '''Preprocesses an image with the defined options; used by the
        dataset aggregation methods (getLFWDataset, getKadDataset) and should
        probably not be used directly.

        Note, this function differs from getLFWDataset only in that the Kadenze
        dataset is structured with multiple directories for each person, and
        each person has some processed image in their directory which shouldn't
        be included.

        Parameters
        ----------
        file_i : TYPE
            Description
        label_i : TYPE
            Description
        pathToFile : TYPE
            Description
        ds : TYPE, optional
            Description

        Returns
        -------

        ds - dictionary
        {
            'filename': file_i,
            'y': label_i,
            'str_y': pathToFile,
            'shape': img.shape,
            'X': img
        }

        Deleted Parameters
        ------------------

        file_i - string
            The filename of the image
        label_i - string
            The directory containing the image, or something unique to the 
            person in the image
        pathToFile - string
            The full path to the filename
        ds - list of dictionaries
            The dataset of images added so far; used for computing
            similarity threshold.  Set to None if not computing 
            similarity to other images.
        '''

        # First process the crop
        img = sp.misc.imread(os.path.join(pathToFile, file_i)).astype(np.uint8)

        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        x, y, w, h = None, None, None, None

        if self.crop_style == 'haar':
            faces = self.face_cascade.detectMultiScale(gray_img, 1.2)
            if len(faces):
                face = faces[0]
                x, y, w, h = face[0], face[1], face[2], face[3]
        elif self.crop_style == 'hog':
            faces = self.detector(gray_img)
            if len(faces):
                face = faces[0]
                x, y, w, h = face.left(), face.top(), face.right() - \
                    face.left(), face.bottom() - face.top()
        elif 'clm' in self.crop_style:
            fr = self.clm.fit(
                gray_img,
                bCropToHOG=self.bCropToHOG,
                bCropToHaar=self.bCropToHaar
            )
            if fr is not None:
                x, y, w, h = fr['bbox']
        elif 'trees' in self.crop_style:
            result = self.model.fit(gray_img)
            if result is not None:
                s = np.max(
                    (result['face-bbox'][2:], result['pts-bbox'][2:])) / 2
                x = result['nose'][0] - s
                y = result['nose'][1] - s
                w = s * 2
                h = s * 2
        elif self.crop_style == 'none':
            pass
        else:
            raise ValueError(
                'Unknown crop style.  Known options: ["trees"], "hog+clm", "haar+clm", "hog", and "haar"')

        if x is not None:
            # test whether the detected image is above the area threshold
            if (w * h) / (img.shape[0] * img.shape[1]) >= self.area_threshold:
                print('rejecting')
                return None
            # Scale the crop and apply it to the grayscale image
            x, y, w, h = scale_crop(
                x, y, w, h,
                gray_img.shape[1], gray_img.shape[0],
                self.crop_factor)
            if self.b_convert_to_grayscale:
                crop = gray_img[y:y + h, x:x + w]
                img = sp.misc.imresize(crop, self.resolution)[np.newaxis, ...]
            else:
                crop = img[y:y + h, x:x + w, ...]
                img = np.rollaxis(
                    sp.misc.imresize(crop, self.resolution),
                    -1)[np.newaxis, ...]
        else:
            # Didn't require crop
            img = sp.misc.imresize(
                sp.misc.imread(os.path.join(pathToFile, file_i),
                               flatten=self.b_convert_to_grayscale),
                self.resolution
            )
            if not self.b_convert_to_grayscale:
                img = np.rollaxis(img, -1)
            img = img[np.newaxis, ...]

        if np.max(img) > 0:
            if ds is not None and self.similar_img_threshold > 0:
                # check all other images with the same label
                other_imgs = [d['X'] for d in ds if d['y'] == label_i]
                for other_img in other_imgs:
                    # calculate the l1 distance to that image
                    l1dist = np.sum(
                        np.abs(img.astype(float) / 255.0 -
                               other_img.astype(float) / 255.0)
                    ) / (img.shape[0] * img.shape[1] * img.shape[2])
                    # if less than 5% of the pixels are different, it's
                    # probably the same image
                    if l1dist < self.similar_img_threshold:
                        return None
            if self.b_convert_img_to_serializable:
                return {
                    'filename': file_i,
                    'y': label_i,
                    'str_y': pathToFile,
                    'shape': img.shape,
                    'X': Binary(pickle.dumps(img, protocol=2), subtype=128)
                }
            else:
                return {
                    'filename': file_i,
                    'y': label_i,
                    'str_y': pathToFile,
                    'shape': img.shape,
                    'X': img
                }
        return None
