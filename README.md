# lartpc_minkowski

Deep learning based LArTPC event reconstruction using MinkowskiEngine sparse convolution APIs.

This repository is an extension/update of [`DeepLearnPhysics/lartpc_mlreco3d`](https://github.com/DeepLearnPhysics/lartpc_mlreco3d) using the [`StanfordVL/MinkowskiEngine`](MinkowskiEngine) generalized spatio-temporal sparse convolution library. 

This repository uses the generalized sparse convolution library provided by StanfordVL:   
Github Link: [MinkowskiEngine](https://github.com/StanfordVL/MinkowskiEngine)  
Paper (arXiv link): [4D Spatio-Temporal ConvNets: Minkowski Convolutional Neural Networks](https://arxiv.org/abs/1904.08755)

## I. Contributing

 1. Please keep the 80 column limit, unless absolutely necessary.

## II. Repository Structure

 * `mlreco`: contains all ML/DL algorithms for reconstructing LArTPC event.

  * `iotools`:

  * `nn`: shorthand for 'neural network'. This directory contains all neural network implementations that does not define a standalone model.

  * `models`: contains all standalone models that could be trained by calling `bin/run.py`.

  * `post_processing`: directory for post_processing algorithms, such as NMS and thresholding for PPN.

  * `utils`: miscellaneous utility functions.

 * `bin`:

 * `test`: contains unit tests.

 * `config`: configurations (`*.cfg`) files for training and validation.

## III. Installation

We will use `singularity` containers to ship software.
