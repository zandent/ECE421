import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import helper as hlp

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
    condition_p = -0.5 * dist / tf.transpose(sigma)
    up = condition_p
    down = 0.5 * tf.log(2*(np.pi)*tf.transpose(sigma)) * tf.to_float(tf.shape(X)[1])
    return up - down

def log_posterior(log_PDF, log_pi):
    # Input
    # log_PDF: log Gaussian PDF N X K
    # log_pi: K X 1

    # Outputs
    # log_post: N X K

    # TODO
    up = tf.add(tf.transpose(log_pi),log_PDF)
    down = hlp.reduce_logsumexp(tf.transpose(log_pi)+log_PDF,keep_dims=True)
    return up - down


tf.random.seed(45689)
is_valid = 1
# Loading data
data = np.load('data100D.npy')
#data = np.load('data2D.npy')
[num_pts, dim] = np.shape(data)

# For Validation set
if is_valid:
  valid_batch = int(num_pts / 3.0)
  np.random.seed(45689)
  rnd_idx = np.arange(num_pts)
  np.random.shuffle(rnd_idx)
  val_data = data[rnd_idx[:valid_batch]]
  data = data[rnd_idx[valid_batch:]]
K = 30
epochs = 3000
#pk = hlp.logsoftmax(tf.transpose(tf.range(1, K + 1, 1)))
pk = tf.get_variable(name = 'pk', shape=[K,1], initializer=tf.random_normal_initializer())
log_pi_ = hlp.logsoftmax(pk)
X = tf.placeholder(dtype=tf.float32, shape = [None,dim], name = "X")
sigma_ = tf.exp(tf.get_variable(name = 'sigma', shape=[K,1], initializer=tf.random_normal_initializer()))
MU = tf.get_variable(name = 'MU', shape=[K,dim], initializer=tf.random_normal_initializer())
log_PDF_ = log_GaussPDF(X,MU,sigma_)
loss = - tf.reduce_sum(hlp.reduce_logsumexp(tf.add(log_PDF_ , tf.transpose(log_pi_) ), 1, keep_dims=True))
optimizer = tf.train.AdamOptimizer(learning_rate=0.003, beta1=0.9, beta2=0.99,epsilon=1e-5).minimize(loss)
clu_index = tf.argmax(tf.nn.softmax(log_posterior(log_PDF_,log_pi_)),1)

train_loss = []
valid_loss = []
init_op= tf.global_variables_initializer()
with tf.Session() as ses:
  ses.run(init_op)
  for step in range(0,epochs):
    if step%500 == 0:
      print("iter: ",step)
    #log_PDF__,log_pi__= ses.run([log_PDF_, log_pi_],feed_dict={X: data})
    #print("pdf shape", np.shape(log_PDF__))
    #print("pi shape", np.shape(log_pi__))
    _,err= ses.run([optimizer, loss],feed_dict={X: data})
    train_loss.append(err)
    valid_err= ses.run([loss],feed_dict={X: val_data})
    valid_loss.append(valid_err)
  clu_index_,MU_arr= ses.run([clu_index,MU],feed_dict={X: data})
  percentages = np.zeros(K)
  for i in range(K):
    percentages[i] = np.sum(np.equal(i, clu_index_))/num_pts
    print("Cluster:", i, "percentage:", percentages[i])
  print("MU:", MU.eval())
  print('Train Error:', train_loss[len(train_loss)-1])
  print('Valid error:',valid_loss[len(valid_loss)-1])
  # plt.figure()
  # plt.subplot(211)
  # plt.scatter(data[:, 0], data[:, 1], c=clu_index_)
  # plt.plot(MU_arr[:, 0], MU_arr[:, 1], c="black", markersize=15, marker="*")
  # plt.xlabel('X')
  # plt.ylabel('Y')
  # plt.subplot(212)
  # PlotTrLoss,=plt.plot(train_loss, 'r', label="TrainLoss")
  # plt.ylabel('Error')
  # plt.xlabel('Epochs')
  # plt.legend(handles=[PlotTrLoss])
  # plt.show()
  plt.figure()
  # plt.subplot(211)
  # if is_valid:
  #   plt.scatter(data[:, 0], data[:, 1], c=clu_index_)
  # else:
  #   plt.scatter(val_data[:, 0], val_data[:, 1], c=clu_index_)
  # plt.plot(MU_arr[:, 0], MU_arr[:, 1], c="black", markersize=15, marker="*")
  # plt.xlabel('X')
  # plt.ylabel('Y')
  # plt.subplot(212)
  if is_valid==0 :
    PlotTrLoss,=plt.plot(train_loss, 'r', label="Loss")
  else:
    PlotTrLoss,=plt.plot(valid_loss, 'r', label="Loss")
  plt.ylabel('Error')
  plt.xlabel('Epochs')
  plt.legend(handles=[PlotTrLoss])
  plt.show()