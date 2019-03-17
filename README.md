# Similarity Encoders (SimEc)

This repository contains the code for the Similarity Encoder (SimEc) neural network architecture based on the `keras` library. Several Jupyter notebooks with examples should give you an idea of how to use this code. A basic setup of the model is also implemented using the `torch` NN library.
For further details on the model and experiments please refer to the [paper](https://arxiv.org/abs/1702.01824) - and of course if any of this code was helpful for your research, please consider citing it:
```
@article{horn2018simec,
  title={Predicting Pairwise Relations with Neural Similarity Encoders},
  author={Horn, Franziska and Müller, Klaus-Robert},
  journal={Bulletin of the Polish Academy of Sciences: Technical Sciences},
  volume={66},
  number={6},
  pages={821--830},
  year={2018},
  publisher={Polish Academy of Sciences}
}
```

This code is still work in progress and intended for research purposes. It was programmed for Python 3 but should also work in Python 2.7.

#### dependencies
- *main simec code:* `numpy`, `keras` (with `tensorflow` backend) (or `torch`)
- *examples:* `scipy`, `sklearn`, `matplotlib`, [`nlputils`](https://github.com/cod3licious/nlputils)

### Getting your hands dirty

First check out the Jupyter notebook [`basic_examples_simec_with_keras.ipynb`](https://github.com/cod3licious/simec/blob/master/basic_examples_simec_with_keras.ipynb), to get an idea of how Similarity Encoders can be implemented with keras. Then have a look at [`basic_examples_compact.ipynb`](https://github.com/cod3licious/simec/blob/master/basic_examples_compact.ipynb), which uses the `SimilarityEncoder` class from `simec.py` to setup a basic SimEc model with less lines of code.

The other Jupyter notebooks contain further examples and experiments reported in the paper (see below).

If you're interested in the PyTorch implementation of SimEc, checkout the [`examples_torch.ipynb`](https://github.com/cod3licious/simec/blob/master/examples_torch.ipynb) notebook, which gives some examples of how to use the SimEc model implemented in `simec_torch.py`.

If you have any questions please don't hesitate to send me an [email](mailto:cod3licious@gmail.com) and of course if you should find any bugs or want to contribute other improvements, pull requests are very welcome!


#### Examples
- [`experiments_paper.ipynb`](https://github.com/cod3licious/simec/blob/master/experiments_paper.ipynb): All experimental results reported in the original paper.
- [`00_matrix_factorization.ipynb`](https://github.com/cod3licious/simec/blob/master/00_matrix_factorization.ipynb): Classical SVD and eigendecomposition of a random matrix R (m x n) and a square symmetric matrix S (m x m) with SimEc to show that NN can be used for this kind of computation as first described in 1992 by A. Cichocki.
- [`00_flowerpots.ipynb`](https://github.com/cod3licious/simec/blob/master/00_flowerpots.ipynb): A simple illustrative example to show that SimEc can learn the connection between feature vectors and an arbitrary similarity matrix, thereby being able to map new test samples into a similarity preserving embedding space, which kernel PCA is unable to do.
- [`01_embed_linear_nonlinear.ipynb`](https://github.com/cod3licious/simec/blob/master/01_embed_linear_nonlinear.ipynb): Show on the MNIST (image) and 20 newsgroups (text) datasets, that SimEc can achieve the same solution as kernel PCA for linear and non-linear similarity matrices.
- [`02_embed_nonmetric.ipynb`](https://github.com/cod3licious/simec/blob/master/02_embed_nonmetric.ipynb): Experiments to demonstrate that SimEc can predict non-metric similarities and multiple similarities at once.
- [`03_embed_classlabels.ipynb`](https://github.com/cod3licious/simec/blob/master/03_embed_classlabels.ipynb): Experiments to demonstrate that SimEc can learn embeddings based on human similarity judgments.
- [`04_noisy_data.ipynb`](https://github.com/cod3licious/simec/blob/master/04_noisy_data.ipynb): Show how SimEc deals with noise in the input data (random/correlated, either added to the data or as additional dimensions). While kPCA can only handle moderate amounts of noise, SimEc is capable of filtering out noise even if it is several times the standard deviation of the underlying data.
- [`05_manifold_s-curve.ipynb`](https://github.com/cod3licious/simec/blob/master/05_manifold_s-curve.ipynb): Experiments on classical manifold learning datasets like the S-curve. With the right target similarities and parameters, SimEc can get both a "global" solution like PCA or a "local" solution (i.e. "unrolling" the manifold) like isomap.
- [`06_link_prediction.ipynb`](https://github.com/cod3licious/simec/blob/master/06_link_prediction.ipynb): Shows how SimEc can be used to predict relations between two entities on popular link prediction datasets.
- [`07_recommender_systems.ipynb`](https://github.com/cod3licious/simec/blob/master/07_recommender_systems.ipynb): Gives an example how SimEc can be used in practice for improving recommender systems. In particular, we show that better recommendations can be generated for new items, which did not receive any user ratings yet, by learning the mapping between the items' feature vectors and the user preferences. Furthermore, we show how the predicted ratings can be interpreted (i.e., why a certain user prefers an item) and how SimEc embeddings can improve suggestions based on content based similarity scores.
