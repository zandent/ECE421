import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import helper as hlp

# Loading data
#data = np.load('data100D.npy')
data = np.load('data2D.npy')
[num_pts, dim] = np.shape(data)

# For Validation set
if is_valid:
  valid_batch = int(num_pts / 3.0)
  np.random.seed(45689)
  rnd_idx = np.arange(num_pts)
  np.random.shuffle(rnd_idx)
  val_data = data[rnd_idx[:valid_batch]]
  data = data[rnd_idx[valid_batch:]]

# Distance function for GMM
def distanceFunc(X, MU):
    # Inputs
    # X: is an NxD matrix (N observations and D dimensions)
    # MU: is an KxD matrix (K means and D dimensions)
    # Outputs
    # pair_dist: is the pairwise distance matrix (NxK)
    # TODO
    newX = tf.expand_dims(X,0)
    newMU = tf.expand_dims(MU,1)
    diff = tf.squared_difference(newX,newMU)
    return tf.transpose(tf.reduce_sum(diff,2))

def log_GaussPDF(X, mu, sigma):
    # Inputs
    # X: N X D
    # mu: K X D
    # sigma: K X 1

    # Outputs:
    # log Gaussian PDF N X K

    # TODO
    dist = distanceFunc(X, mu)
    condition_p = tf.exp(tf.multiply(-1/2,tf.divide(dist,tf.transpose(sigma)) ) )
    return tf.log(condition_p)

def log_posterior(log_PDF, log_pi):
    # Input
    # log_PDF: log Gaussian PDF N X K
    # log_pi: K X 1

    # Outputs
    # log_post: N X K

    # TODO
    up = tf.add(tf.transpose(log_pi),log_PDF)
    down = hlp.reduce_logsumexp(tf.add(tf.transpose(log_pi),log_PDF),keep_dims=True)
    return up - down


