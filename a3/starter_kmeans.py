import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import helper as hlp

# Distance function for K-means
def distanceFunc(X, MU):
    # Inputs
    # X: is an NxD matrix (N observations and D dimensions)
    # MU: is an KxD matrix (K means and D dimensions)
    # Outputs
    # pair_dist: is the squared pairwise distance matrix (NxK)
    # TODO
    
    newX = tf.expand_dims(X,0)
    newMU = tf.expand_dims(MU,1)
    diff = tf.squared_difference(newX,newMU)
    return tf.transpose(tf.reduce_sum(diff,2))

is_valid = 0
K = 5
epochs = 1000
# Loading data
data = np.load('data2D.npy')
#data = np.load('data100D.npy')
[num_pts, dim] = np.shape(data)

# For Validation set
if is_valid:
  valid_batch = int(num_pts / 3.0)
  np.random.seed(45689)
  rnd_idx = np.arange(num_pts)
  np.random.shuffle(rnd_idx)
  val_data = data[rnd_idx[:valid_batch]]
  data = data[rnd_idx[valid_batch:]]

MU = tf.get_variable(name = 'MU', shape=[K,dim], initializer=tf.random_normal_initializer())
X = tf.placeholder(dtype=tf.float32, shape = [None,dim], name = "X")
loss = tf.reduce_sum(tf.reduce_min(distanceFunc(X,MU),1))
optimizer = tf.train.AdamOptimizer(learning_rate=0.1, beta1=0.9, beta2=0.99,epsilon=1e-5).minimize(loss)
clu_index = tf.argmin(distanceFunc(X, MU), 1)

train_loss = []
valid_loss = []
init_op= tf.global_variables_initializer()
with tf.Session() as ses:
  ses.run(init_op)
  for step in range(0,epochs):
    #print("iter: ",step)
    _,err= ses.run([optimizer, loss],feed_dict={X: data})
    train_loss.append(err)
  clu_index_,MU_arr= ses.run([clu_index,MU],feed_dict={X: data})
  percentages = np.zeros(K)
  for i in range(K):
    percentages[i] = np.sum(np.equal(i, clu_index_))/num_pts
    print("Cluster:", i, "percentage:", percentages[i])
  print("MU:", MU.eval())
  print('Train Error:', train_loss[len(train_loss)-1])
  plt.figure()
  plt.subplot(211)
  plt.scatter(data[:, 0], data[:, 1], c=clu_index_)
  plt.plot(MU_arr[:, 0], MU_arr[:, 1], c="black", markersize=15, marker="*")
  plt.xlabel('X')
  plt.ylabel('Y')
  plt.subplot(212)
  PlotTrLoss,=plt.plot(train_loss, 'r', label="TrainLoss")
  plt.ylabel('Error')
  plt.xlabel('Epochs')
  plt.legend(handles=[PlotTrLoss])
  plt.show()


