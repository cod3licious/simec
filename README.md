# Similarity Encoders (SimEc)

This repository contains the code for the Similarity Encoder (SimEc) neural network architecture based on the `keras` library. Several Jupyter notebooks with examples should give you an idea of how to use this code. 
For further details on the model and experiments please refer to the [paper](http://arxiv.org/abs/1702.01824) (updated version will come soon!).

This code is still work in progress and intended for research purposes. It was programmed for Python 2.7.

#### dependencies
- *main simec code:* `numpy`, `keras` (with `tensorflow` backend)
- *examples:* `scipy`, `sklearn`, `matplotlib`, [`nlputils`](https://github.com/cod3licious/nlputils)

### Getting your hands dirty

First check out the Jupyter notebook [`basic_examples_simec_with_keras.ipynb`](https://github.com/cod3licious/simec/blob/master/basic_examples_simec_with_keras.ipynb), to get an idea of how Similarity Encoders can be implemented with keras. Then have a look at [`basic_examples_compact.ipynb`](https://github.com/cod3licious/simec/blob/master/basic_examples_compact.ipynb), which uses the `SimilarityEncoder` class from `simec.py` to setup the basic SimEc model with less lines of code.

The other Jupyter notebooks contain further examples and experiments reported in the paper.

If you have any questions please don't hesitate to send me an [email](mailto:cod3licious@gmail.com) and of course if you should find any bugs or want to contribute other improvements, pull requests are very welcome!
