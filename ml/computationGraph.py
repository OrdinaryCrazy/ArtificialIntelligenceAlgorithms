#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf


# In[2]:


tf.__version__


# In[3]:


# computation graph
x = tf.Variable(3, name="x")
y = tf.Variable(4, name="y")
f = x*x*y + y + 2


# In[4]:


sess = tf.Session()
sess.run(x.initializer)
sess.run(y.initializer)
result = sess.run(f)
result


# In[5]:


with tf.Session() as sess :
    x.initializer.run()
    y.initializer.run()
    result = f.eval()
    print(result)


# In[6]:


init = tf.global_variables_initializer()

with tf.Session() as sess :
    init.run()
    result = f.eval()
    print(result)


# In[7]:


sess = tf.InteractiveSession()
init.run()
result = f.eval()
print(result)
sess.close()


# In[8]:


# managing graphs
x.graph is tf.get_default_graph()


# In[9]:


graph = tf.Graph()
with graph.as_default():
    x2 = tf.Variable(2)
x2.graph is graph


# In[10]:


x2.graph is tf.get_default_graph()


# In[11]:


# clean to default graph
tf.reset_default_graph()


# In[12]:


import numpy as np
from sklearn.datasets import fetch_california_housing

housing = fetch_california_housing()
m, n = housing.data.shape
housingDataPlusBias = np.c_[np.ones((m,1)), housing.data]

X = tf.constant(housingDataPlusBias, dtype=tf.float32, name="X")
y = tf.constant(housing.target.reshape(-1,1), dtype=tf.float32, name="y")
XT = tf.transpose(X)
theta = tf.matmul(tf.matmul(tf.matrix_inverse(tf.matmul(XT, X)), XT), y)

with tf.Session() as sess :
    theta_value = theta.eval()


# In[13]:


theta_value


# In[17]:


tf.reset_default_graph()

n_epochs = 1000
learning_rate = 0.01

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
housingDataPlusBiasScaled = np.c_[np.ones((m,1)), scaler.fit_transform(housing.data)]
X = tf.constant(housingDataPlusBiasScaled,    dtype=tf.float32, name="X")
y = tf.constant(housing.target.reshape(-1,1), dtype=tf.float32, name="y")
theta = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0), name="theta")
y_pred = tf.matmul(X, theta, name="prediction")
error = y_pred - y
mse = tf.reduce_mean(tf.square(error), name="mse")
# manually compute
# gradients = 2/m * tf.matmul(tf.transpose(X), error)
# autodiff
# gradients = tf.gradients(mse, [theta])[0]
# create a node that will assign a new value to a variable
# training_op = tf.assign(theta, theta - learning_rate * gradients)

# optimizer
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9)
training_op = optimizer.minimize(mse)

init = tf.global_variables_initializer()

saver = tf.train.Saver()
    
with tf.Session() as sess :
    init.run()
    for epoch in range(n_epochs) :
        if epoch % 100 == 0:
            print("Epoch", epoch, "MSE = ", mse.eval())
        sess.run(training_op)
    best_theta = theta.eval()
    save_path = saver.save(sess, "test_save.ckpt")


# In[18]:


# placeholder and data feeding
A = tf.placeholder(tf.float32, shape=(None, 3))
B = A + 5

with tf.Session() as sess :
    B_val_1 = B.eval(feed_dict={A:[[1,2,3]]})
    B_val_2 = B.eval(feed_dict={A:[[4,5,6], [7,8,9]]})


# In[19]:


B_val_1


# In[20]:


B_val_2


# In[23]:


from datetime import datetime

now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
root_logdir = "tf_logs"
logdir = "{}/run-{}".format(root_logdir, now)

tf.reset_default_graph()

n_epochs = 100
learning_rate = 0.01

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
housingDataPlusBiasScaled = np.c_[np.ones((m,1)), scaler.fit_transform(housing.data)]

X = tf.constant(housingDataPlusBiasScaled,    dtype=tf.float32, name="X")
y = tf.constant(housing.target.reshape(-1,1), dtype=tf.float32, name="y")
theta = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0), name="theta")
y_pred = tf.matmul(X, theta, name="prediction")
error = y_pred - y
mse = tf.reduce_mean(tf.square(error), name="mse")
optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9)
training_op = optimizer.minimize(mse)

init = tf.global_variables_initializer()

mse_summary = tf.summary.scalar('MSE', mse)
file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())

with tf.Session() as sess :
    init.run()
    for epoch in range(n_epochs) :
        if epoch % 10 == 0:
            summary_str = mse_summary.eval()
            step = epoch
            file_writer.add_summary(summary_str, step)
            print("Epoch", epoch, "MSE = ", mse.eval())
        sess.run(training_op)
    best_theta = theta.eval()
file_writer.close()


# In[25]:


# name scope
with tf.name_scope("loss") as scope :
    error = y_pred - y
    mse = tf.reduce_mean(tf.square(error), name="mse")


# In[26]:


error.op.name


# In[27]:


mse.op.name


# In[29]:


n_features = 3
X = tf.placeholder(tf.float32, shape=(None, n_features), name="X")

w1 = tf.Variable(tf.random_normal((n_features, 1)), name="weights1")
w2 = tf.Variable(tf.random_normal((n_features, 1)), name="weights2")
b1 = tf.Variable(0.0, name="bias1")
b2 = tf.Variable(0.0, name="bias2")

z1 = tf.add(tf.matmul(X, w1), b1, name="z1")
z2 = tf.add(tf.matmul(X, w2), b2, name="z2")

relu1 = tf.maximum(z1, 0.0, name="relu1")
relu2 = tf.maximum(z2, 0.0, name="relu2")

output = tf.add(relu1, relu2, name="output")


# In[31]:


def relu(X):
    with tf.variable_scope("relu", reuse=True):
        threshold = tf.get_variable("threshold")
        w_shape = (int(X.get_shape()[1]), 1)
        w = tf.Variable(tf.random_normal(w_shape), name="weights")
        b = tf.Variable(0.0, name="bias")
        z = tf.add(tf.matmul(X,w), b, name="z")
#       return tf.maximum(z,0.0,name="relu")   
        return tf.maximum(z,threshold,name="relu")


n_features = 3
X = tf.placeholder(tf.float32, shape=(None, n_features), name="X")
with tf.variable_scope("relu"):
    threshold = tf.get_variable("threshold", shape=(), 
                                initializer=tf.constant_initializer(0.0))
relus = [relu(X) for i in range(5)]
output = tf.add_n(relus, name="output")


# In[32]:


def relu(X):
    threshold = tf.get_variable("threshold", shape=(), 
                                initializer=tf.constant_initializer(0.0))
    w_shape = (int(X.get_shape()[1]), 1)
    w = tf.Variable(tf.random_normal(w_shape), name="weights")
    b = tf.Variable(0.0, name="bias")
    z = tf.add(tf.matmul(X,w), b, name="z")
#   return tf.maximum(z,0.0,name="relu")   
    return tf.maximum(z,threshold,name="relu")


n_features = 3
X = tf.placeholder(tf.float32, shape=(None, n_features), name="X")
relus = []
for i in range(5):
    with tf.variable_scope("relu", reuse=(i >= 1)) as scope:
        relus.append(relu(X))
output = tf.add_n(relus, name="output")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




