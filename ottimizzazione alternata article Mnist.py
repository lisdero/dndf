import tensorflow as tf
import numpy as np
import tensorflow.examples.tutorials.mnist.input_data as input_data

DEPTH   = 3                 # Depth of a tree
N_LEAF  = 2 ** (DEPTH + 1)  # Number of leaf node
N_LABEL = 10                # Number of classes
N_TREE  = 5                 # Number of trees (ensemble)
N_BATCH = 128               # Number of data points per mini-batch
ALL_BATCH = 5000
import random
random.seed(1)
def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))


def init_prob_weights(shape, minval=-5, maxval=5):
    return tf.Variable(tf.random_uniform(shape, minval, maxval))


def model(X, w, w2, w3, w4_e, w_d_e, w_l_e, p_keep_conv, p_keep_hidden):
    """
    Create a forest and return the neural decision forest outputs:
        decision_p_e: decision node routing probability for all ensemble
            If we number all nodes in the tree sequentially from top to bottom,
            left to right, decision_p contains
            [d(0), d(1), d(2), ..., d(2^n - 2)] where d(1) is the probability
            of going left at the root node, d(2) is that of the left child of
            the root node.
            decision_p_e is the concatenation of all tree decision_p's
        leaf_p_e: terminal node probability distributions for all ensemble. The
            indexing is the same as that of decision_p_e.
    """
    assert(len(w4_e) == len(w_d_e))
    assert(len(w4_e) == len(w_l_e))
    l1a = tf.nn.relu(tf.nn.conv2d(X, w, [1, 1, 1, 1], 'SAME'))
    l1 = tf.nn.max_pool(l1a, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')
    l1 = tf.nn.dropout(l1, p_keep_conv)
    l2a = tf.nn.relu(tf.nn.conv2d(l1, w2, [1, 1, 1, 1], 'SAME'))
    l2 = tf.nn.max_pool(l2a, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')
    l2 = tf.nn.dropout(l2, p_keep_conv)
    l3a = tf.nn.relu(tf.nn.conv2d(l2, w3, [1, 1, 1, 1], 'SAME'))
    l3 = tf.nn.max_pool(l3a, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')
    l3 = tf.reshape(l3, [-1, w4_e[0].get_shape().as_list()[0]])
    l3 = tf.nn.dropout(l3, p_keep_conv)
    decision_p_e = []
    leaf_p_e = []
    for w4, w_d, w_l in zip(w4_e, w_d_e, w_l_e):
        l4 = tf.nn.relu(tf.matmul(l3, w4))
        l4 = tf.nn.dropout(l4, p_keep_hidden)
        decision_p = tf.nn.sigmoid(tf.matmul(l4, w_d))
        leaf_p = tf.nn.softmax(w_l)
        decision_p_e.append(decision_p)
        leaf_p_e.append(leaf_p)
    return decision_p_e, leaf_p_e





##################################################
# Load dataset
##################################################
mnist = input_data.read_data_sets("MNIST/", one_hot=True)
trX, trY = mnist.train.images, mnist.train.labels
teX, teY = mnist.test.images, mnist.test.labels
trX = trX.reshape(-1, 28, 28, 1)
teX = teX.reshape(-1, 28, 28, 1)

# Input X, output Y
X = tf.placeholder("float", [N_BATCH, 28, 28, 1])
Y = tf.placeholder("float", [N_BATCH, N_LABEL])

X_all = tf.placeholder("float", [ALL_BATCH, 28, 28, 1])
Y_all = tf.placeholder("float", [ALL_BATCH, N_LABEL])
##################################################
# Initialize network weights
##################################################
w = init_weights([3, 3, 1, 32])
w2 = init_weights([3, 3, 32, 64])
w3 = init_weights([3, 3, 64, 128])

w4_ensemble = []
w_d_ensemble = []
w_l_ensemble = []
for i in range(N_TREE):
    w4_ensemble.append(init_weights([128 * 4 * 4, 625]))
    w_d_ensemble.append(init_prob_weights([625, N_LEAF], -1, 1))
    w_l_ensemble.append(init_prob_weights([N_LEAF, N_LABEL], -2, 2))

p_keep_conv = tf.placeholder("float")
p_keep_hidden = tf.placeholder("float")

##################################################
# Define a fully differentiable deep-ndf
##################################################
# With the probability decision_p, route a sample to the right branch
decision_p_e, leaf_p_e = model(X, w, w2, w3, w4_ensemble, w_d_ensemble,
                               w_l_ensemble, p_keep_conv, p_keep_hidden)

decision_p_e_all, leaf_p_e_all = model(X_all, w, w2, w3, w4_ensemble, w_d_ensemble,
                               w_l_ensemble, p_keep_conv, p_keep_hidden)

flat_decision_p_e = []

# iterate over each tree
for decision_p in decision_p_e:
    # Compute the complement of d, which is 1 - d
    # where d is the sigmoid of fully connected output
    decision_p_comp = tf.subtract(tf.ones_like(decision_p), decision_p)
    # Concatenate both d, 1-d
    decision_p_pack = tf.stack([decision_p, decision_p_comp])
    # Flatten/vectorize the decision probabilities for efficient indexing
    flat_decision_p = tf.reshape(decision_p_pack, [-1])
    flat_decision_p_e.append(flat_decision_p)

flat_decision_p_e_all = []

# iterate over each tree
for decision_p_all in decision_p_e_all:
    # Compute the complement of d, which is 1 - d
    # where d is the sigmoid of fully connected output
    decision_p_comp_all = tf.subtract(tf.ones_like(decision_p_all), decision_p_all)
    # Concatenate both d, 1-d
    decision_p_pack_all = tf.stack([decision_p_all, decision_p_comp_all])
    # Flatten/vectorize the decision probabilities for efficient indexing
    flat_decision_p_all = tf.reshape(decision_p_pack_all, [-1])
    flat_decision_p_e_all.append(flat_decision_p_all)


# 0 index of each data instance in a mini-batch
batch_0_indices = \
    tf.tile(tf.expand_dims(tf.range(0, N_BATCH * N_LEAF, N_LEAF), 1),
            [1, N_LEAF])

###############################################################################
# The routing probability computation
#
# We will create a routing probability matrix \mu. First, we will initialize
# \mu using the root node d, 1-d. To efficiently implement this routing, we
# will create a giant vector (matrix) that contains all d and 1-d from all
# decision nodes. The matrix version of that is decision_p_pack and vectorized
# version is flat_decision_p.
#
# The suffix `_e` indicates an ensemble. i.e. concatenation of all responsens
# from trees.
#
# For depth = 2 tree, the routing probability for each leaf node can be easily
# compute by multiplying the following vectors elementwise.
# \mu =       [d_0,   d_0,   d_0,   d_0, 1-d_0, 1-d_0, 1-d_0, 1-d_0]
# \mu = \mu * [d_1,   d_1, 1-d_1, 1-d_1,   d_2,   d_2, 1-d_2, 1-d_2]
# \mu = \mu * [d_3, 1-d_3,   d_4, 1-d_4,   d_5, 1-d_5,   d_6, 1-d_6]
#
# Tree indexing
#      0
#    1   2
#   3 4 5 6
##############################################################################
in_repeat = int(N_LEAF / 2)
out_repeat = int(N_BATCH)

# Let N_BATCH * N_LEAF be N_D. flat_decision_p[N_D] will return 1-d of the
# first root node in the first tree.
batch_complement_indices = \
    np.array([[0] * in_repeat, [N_BATCH * N_LEAF] * in_repeat]
             * out_repeat).reshape(N_BATCH, N_LEAF)

# First define the routing probabilities d for root nodes
mu_e = []

# iterate over each tree
for i, flat_decision_p in enumerate(flat_decision_p_e):
    mu = tf.gather(flat_decision_p,
                   tf.add(batch_0_indices, batch_complement_indices))
    mu_e.append(mu)

# from the second layer to the last layer, we make the decision nodes
for d in range(1, DEPTH + 1):
    indices = tf.range(2 ** d, 2 ** (d + 1)) - 1
    tile_indices = tf.reshape(tf.tile(tf.expand_dims(indices, 1),
                                      [1, 2 ** (DEPTH - d + 1)]), [1, -1])
    batch_indices = tf.add(batch_0_indices, tf.tile(tile_indices, [N_BATCH, 1]))
    in_repeat = int(in_repeat / 2)
    out_repeat = int(out_repeat * 2)
    # Again define the indices that picks d and 1-d for the node
    batch_complement_indices = \
        np.array([[0] * in_repeat, [N_BATCH * N_LEAF] * in_repeat]
                 * out_repeat).reshape(N_BATCH, N_LEAF)
    mu_e_update = []
    for mu, flat_decision_p in zip(mu_e, flat_decision_p_e):
        mu = tf.multiply(mu, tf.gather(flat_decision_p,
                                  tf.add(batch_indices, batch_complement_indices)))
        mu_e_update.append(mu)
    mu_e = mu_e_update

batch_0_indices_all = \
    tf.tile(tf.expand_dims(tf.range(0, ALL_BATCH * N_LEAF, N_LEAF), 1),
            [1, N_LEAF])

in_repeat_all = int(N_LEAF / 2)
out_repeat_all = int(ALL_BATCH)

batch_complement_indices_all = \
    np.array([[0] * in_repeat_all, [ALL_BATCH * N_LEAF] * in_repeat_all]
             * out_repeat_all).reshape(ALL_BATCH, N_LEAF)


mu_e_all = []

# iterate over each tree
for i, flat_decision_p_all in enumerate(flat_decision_p_e_all):
    mu_all = tf.gather(flat_decision_p_all,
                   tf.add(batch_0_indices_all, batch_complement_indices_all))
    mu_e_all.append(mu_all)

# from the second layer to the last layer, we make the decision nodes
for d in range(1, DEPTH + 1):
    indices_all = tf.range(2 ** d, 2 ** (d + 1)) - 1
    tile_indices_all = tf.reshape(tf.tile(tf.expand_dims(indices_all, 1),
                                      [1, 2 ** (DEPTH - d + 1)]), [1, -1])
    batch_indices_all = tf.add(batch_0_indices_all, tf.tile(tile_indices_all, [ALL_BATCH, 1]))
    in_repeat_all = int(in_repeat_all / 2)
    out_repeat_all = int(out_repeat_all * 2)
    # Again define the indices that picks d and 1-d for the node
    batch_complement_indices_all = \
        np.array([[0] * in_repeat_all, [ALL_BATCH * N_LEAF] * in_repeat_all]
                 * out_repeat_all).reshape(ALL_BATCH, N_LEAF)
    mu_e_update_all = []
    for mu_all, flat_decision_p_all in zip(mu_e_all, flat_decision_p_e_all):
        mu_all = tf.multiply(mu_all, tf.gather(flat_decision_p_all,
                                  tf.add(batch_indices_all, batch_complement_indices_all)))
        mu_e_update_all.append(mu_all)
    mu_e_all = mu_e_update_all

##################################################
# Define p(y|x)
##################################################
py_x_e = []
for mu, leaf_p in zip(mu_e, leaf_p_e):
    # average all the leaf p
    py_x_tree = tf.reduce_mean(
        tf.multiply(tf.tile(tf.expand_dims(mu, 2), [1, 1, N_LABEL]),
               tf.tile(tf.expand_dims(leaf_p, 0), [N_BATCH, 1, 1])), 1)
    py_x_e.append(py_x_tree)

py_x_e = tf.stack(py_x_e)
py_x = tf.reduce_mean(py_x_e, 0)



py_x_e_all = []
for mu_all, leaf_p_all in zip(mu_e_all, leaf_p_e_all):
    # average all the leaf p
    py_x_tree_all = tf.reduce_mean(
        tf.multiply(tf.tile(tf.expand_dims(mu_all, 2), [1, 1, N_LABEL]),
               tf.tile(tf.expand_dims(leaf_p_all, 0), [ALL_BATCH, 1, 1])), 1)
    py_x_e_all.append(py_x_tree_all)

py_x_e_all = tf.stack(py_x_e_all)
py_x_all = tf.reduce_mean(py_x_e_all, 0)


##################################################
# Define cost and optimization method
##################################################

# cross entropy loss
cost = tf.reduce_mean(-tf.multiply(tf.log(py_x), Y))

cost2 = tf.reduce_mean(-tf.multiply(tf.log(py_x_all), Y_all))

# cost = tf.reduce_mean(tf.nn.cross_entropy_with_logits(py_x, Y))

train_step_nodes = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cost,var_list=[w,w2,w3,w4_ensemble,w_d_ensemble])

train_step_leaves = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cost2,var_list=[w_l_ensemble])

predict = tf.argmax(py_x, 1)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
acc_alt=np.zeros([2,30])
for i in range(30):
    # One epoch
    for start, end in zip(range(0, len(trX), ALL_BATCH), range(ALL_BATCH, len(trX), ALL_BATCH)):
        sess.run(train_step_leaves, feed_dict={X_all: trX[start:end], Y_all: trY[start:end],
                                        p_keep_conv: 0.8, p_keep_hidden: 0.5})
    
    
    for start, end in zip(range(0, len(trX), N_BATCH), range(N_BATCH, len(trX), N_BATCH)):
        sess.run(train_step_nodes, feed_dict={X: trX[start:end], Y: trY[start:end],
                                        p_keep_conv: 0.8, p_keep_hidden: 0.5})
    
    
    
    # Result on the test set
    results = []
    for start, end in zip(range(0, len(teX), N_BATCH), range(N_BATCH, len(teX), N_BATCH)):
        results.extend(np.argmax(teY[start:end], axis=1) ==
            sess.run(predict, feed_dict={X: teX[start:end], p_keep_conv: 1.0,
                                         p_keep_hidden: 1.0}))
    acc_alt[0,i]=np.mean(results)
    acc_alt[1,i]=i
    print('Epoch: %d, Test Accuracy: %f' % (i + 1, np.mean(results)))
