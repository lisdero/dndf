ALL_BATCH = 55000

def model_all(X, w, w2, w3, w4_e, w_d_e, w_l_e, p_keep_conv, p_keep_hidden):
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

X_all = tf.placeholder("float", [ALL_BATCH, 28, 28, 1])
Y_all = tf.placeholder("float", [ALL_BATCH, N_LABEL])


decision_p_e_all, leaf_p_e_all = model_all(X_all, w, w2, w3, w4_ensemble, w_d_ensemble,
                               w_l_ensemble, p_keep_conv, p_keep_hidden)

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


py_x_e_all = []
for mu_all, leaf_p_all in zip(mu_e_all, leaf_p_e_all):
    # average all the leaf p
    py_x_tree_all = tf.reduce_mean(
        tf.multiply(tf.tile(tf.expand_dims(mu_all, 2), [1, 1, N_LABEL]),
               tf.tile(tf.expand_dims(leaf_p_all, 0), [ALL_BATCH, 1, 1])), 1)
    py_x_e_all.append(py_x_tree_all)

py_x_e_all = tf.stack(py_x_e_all)
py_x_all = tf.reduce_mean(py_x_e_all, 0)

cost2 = tf.reduce_mean(-tf.multiply(tf.log(py_x_all), Y_all))
