# Siamese Net

The siamese network is a method for training a distance function discriminatively.  Its use is popularized in many facial detection/recognition models including ones developed by Facebook and Google.  The basic idea is to run a deep net using pairs of images describing either matched or unmatched pairs.  The same network is run separately for the left and right images, but the loss is computed on the pairs of images rather than a single image.  This is done by making use of the "batch" dimension of the input tensor, and computing loss on interleaved batches.  If the left image is always the even idx (0, 2, 4, ...) and the right image is always the odd idxs, (1, 3, 5, ...), then the loss is computed on the alternating batches: `loss = output[::2] - output[1::2]`, for instance.  By feeding in pairs of images that are either true or false pairs, the output of the networks should try to push similar matching pairs closer to together, while keeping unmatched pairs farther away.

This package shows how to train a siamese network using Lasagne and Theano and includes network definitions for state-of-the-art networks including: DeepID, DeepID2, Chopra et. al, and Hani et. al.  We also include one pre-trained model using a custom convolutional network.

We are releasing all of this to the community in the hopes that it will encourage more models to be shared and appropriated for other possible uses.  The framework we share here should allow one to train their own network, compute results, and visualize the results.  We encourage the community to explore its use, submit pull requests on any issues within the package, and to contribute pre-trained models.

# References

Chopra, S., Hadsell, R., & Y., L. (2005). Learning a similiarty metric discriminatively, with application to face verification. Proceedings of IEEE Conference on Computer Vision and Pattern Recognition, 349–356.

Donahue, J., Jia, Y., Vinyals, O., Hoffman, J., Zhang, N., Tzeng, E., & Darrell, T. (2014). DeCAF: A Deep Convolutional Activation Feature for Generic Visual Recognition. arXiv Preprint. Retrieved from http://arxiv.org/abs/1310.1531

El-bakry, H. M., & Zhao, Q. (2005). Fast Object / Face Detection Using Neural Networks and Fast Fourier Transform, 8580(11), 503–508.

Huang, G. B., Mattar, M. a., Lee, H., & Learned-Miller, E. (2012). Learning to Align from Scratch. Proc. Neural Information Processing Systems, 1–9.

Khalil-Hani, M., & Sung, L. S. (2014). A convolutional neural network approach for face verification. High Performance Computing & Simulation (HPCS), 2014 International Conference on, (3), 707–714. doi:10.1109/HPCSim.2014.6903759

Kostinger, M., Hirzer, M., Wohlhart, P., Roth, P. M., & Bischof, H. (2012). Large scale metric learning from equivalence constraints. Proceedings of the IEEE Computer Society Conference on Computer Vision and Pattern Recognition, (Ldml), 2288–2295. doi:10.1109/CVPR.2012.6247939

Li, H., & Hua, G. (2015). Hierarchical-PEP Model for Real-world Face Recognition, 4055–4064. doi:10.1109/CVPR.2015.7299032

Parkhi, O. M., Vedaldi, A., Zisserman, A., Vedaldi, A., Lenc, K., Jaderberg, M., … others. (2015). Deep face recognition. Proceedings of the British Machine Vision, (Section 3).

Sun, Y., Wang, X., & Tang, X. (2014). Deep Learning Face Representation by Joint Identification-Verification. Nips, 1–9. doi:10.1109/CVPR.2014.244

Taigman, Y., Yang, M., Ranzato, M., & Wolf, L. (2014). DeepFace: Closing the Gap to Human-Level Performance in Face Verification. Conference on Computer Vision and Pattern Recognition (CVPR), 8. doi:10.1109/CVPR.2014.220

Wheeler, F. W., Liu, X., & Tu, P. H. (2007). Multi-Frame Super-Resolution for Face Recognition. 2007 First IEEE International Conference on Biometrics: Theory, Applications, and Systems, 1–6. doi:10.1109/BTAS.2007.4401949

Yi, D., Lei, Z., Liao, S., & Li, S. Z. (2014). Learning Face Representation from Scratch. arXiv.

# License

Copyright 2016 Kadenze, Inc.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.