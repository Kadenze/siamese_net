"""
Siamese Network for performing training of a Deep Convolutional
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

http://www.apache.org/licenses/"""
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
import dlib
from dlib import rectangle


class FaceShapeModel(object):

    """Summary.

    Attributes
    ----------
    detector : TYPE
        Description
    model : TYPE
        Description
    """

    def __init__(self):
        """Summary."""
        model_filename = 'shape_predictor_68_face_landmarks.dat'
        if not os.path.exists(model_filename):
            os.system(
                'wget http://sourceforge.net/projects/dclib/files/dlib/v18.10/shape_predictor_68_face_landmarks.dat.bz2')
            os.system('bunzip2 shape_predictor_68_face_landmarks.dat.bz2')
        self.model = dlib.shape_predictor(model_filename)
        self.detector = dlib.get_frontal_face_detector()

    def localize(self, image):
        """Summary.

        Parameters
        ----------
        image : TYPE
            Description

        Raises
        ------
        ValueError
            Description

        Returns
        -------
        name : TYPE
            Description
        """
        if image.dtype != np.uint8:
            image = image.astype(np.uint8)

        gray_img = None
        if image.ndim == 3:
            gray_img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        elif image.ndim == 2:
            gray_img = image
        else:
            raise ValueError(
                'Unsupported number of image dimensions: %d' % image.ndim)

        faces = self.detector(gray_img)
        if len(faces):
            return [faces[0].left(), faces[0].top(), faces[0].right() - faces[0].left(), faces[0].bottom() - faces[0].top()]
        else:
            return None

    def fit(self, image):
        """Summary.

        Parameters
        ----------
        image : TYPE
            Description

        Returns
        -------
        name : TYPE
            Description
        """
        bbox = self.localize(image)
        if bbox is not None:
            face = rectangle(
                bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3])
            shape = self.model(image, face)
            xs = [p.x for p in shape.parts()]
            ys = [p.y for p in shape.parts()]
            x = np.min(xs)
            y = np.min(ys)
            w = np.max(xs) - np.min(xs)
            h = np.max(ys) - np.min(ys)
            return {
                'pts': zip(xs, ys),
                'xs': xs,
                'ys': ys,
                'pts-bbox': [x, y, w, h],
                'face-bbox': bbox,
                'nose': (shape.parts()[30].x, shape.parts()[30].y)
            }
        else:
            return None

    def draw(self, image, fr=None):
        """Summary.

        Parameters
        ----------
        image : TYPE
            Description
        fr : TYPE, optional
            Description

        Returns
        -------
        name : TYPE
            Description
        """
        if fr is None:
            fr = self.fit(image=image)

        plt.clf()
        plt.imshow(image)
        for pt in fr['pts']:
            plt.plot(pt[0], pt[1], 'ro')
