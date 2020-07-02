# README

README for collaborative work between L. Stephey, M. Marchevsky, and M. Mustafa
at Lawrence Berkeley National Lab.

The goal of this effort is to use unsupervised learning techniques to better
understand acoustic events during the superconducting magnet training
process. 

2019-2020

This repo contains:

# sample-data

About 20 sample spectrograms in .csv format. Kindly provided by M. Marchevsky.

# k-means

Some exploratory work into kmeans clustering with data contained in summary
files (not the raw spectrograms themselves). Summary files kindly provided by
M. Marchevsky.

# conv2d-autoencoder

Jupyter notebooks designed to build and train a 2D convolustional autoencoder
to learn features in our unlabled spectrograms. Several scripts to prepare the
data, several scripts to build and train the network, and several scripts to
plot and analyze the encoded data produced by the trained network.

To reproduce the workflow presented in a poster at the 2020 ai4science
workshop, you can use the notebooks in the following order:

```
process_march_data.ipynb

data_prep.ipynb

process_post_quench_data.ipynb

post_quench_data_prep.ipynb

build_conv2d_autoencoder.ipynb

train_conv2d_autoencoder.ipynb

save_conv2d_autoencoder.ipynb

plot_conv2d_pca.ipynb

plot_conv2d_spectrograms.ipynb

```



