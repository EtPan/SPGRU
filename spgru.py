from __future__ import print_function
import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn

import spgru_indices

# Training Parameters
learning_rate = 0.0001            
training_steps = 20000           
batch_size = 64             
display_step = 1000             

# Network Parameters
timesteps = 1                         
num_hidden1 = 256  
num_hidden2 = 128

num_input = spgru_indices.INPUT_DIMENSION
num_spa =spgru_indices.num_spa 
num_classes = spgru_indices.nb_classes   

data = spgru_indices.read_data_sets()
X = tf.placeholder("float", [None, timesteps, num_input], name='X')
Y = tf.placeholder("float", [None, num_classes], name='Y')
H_in = tf.placeholder("float",[None,num_spa], name='H_in')


# Define weights
weights = {
    'in': tf.Variable(tf.random_normal([num_input,num_hidden1])),
    'out': tf.Variable(tf.random_normal([num_hidden2,num_classes]))
}
biases = {
    'in':tf.Variable(tf.constant(0.1,shape = [num_hidden1,])),
    'out':tf.Variable(tf.constant(0.1,shape = [num_classes,]))
}

def SPGRU(x,h,weights, biases):

    x = tf.reshape(x,[-1,num_input])

    x_in = tf.matmul(x,weights['in']) + biases['in']
    x_in = tf.nn.relu(x_in)
    x_in = tf.reshape(x_in,[-1,timesteps,num_hidden1])

    H_in = tf.layers.dense(h,units=num_hidden1,activation='relu')
    gru_cell = rnn.GRUCell(num_hidden1,name="sgru")    
    outputs, states = tf.nn.dynamic_rnn(gru_cell, x_in, dtype=tf.float32, initial_state=H_in)

    tf.summary.histogram("states",states)

    states = tf.layers.dense(states,units=num_hidden2,activation='relu')
    results = tf.matmul(states, weights['out']) + biases['out']

    return results
  

logits = SPGRU(X,H_in,weights, biases)
tf.add_to_collection('pre_prob', logits)
prediction = tf.nn.softmax(logits)
tf.add_to_collection('rnn_pred_label', prediction)

# Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=Y))
tf.summary.scalar('loss_op', loss_op)
train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss_op)

# Evaluate model (with test logits, for dropout to be disabled)
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
tf.summary.scalar('batch_accuracy', accuracy)

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()
istate = np.zeros([batch_size, num_hidden1])

# Start training
saver = tf.train.Saver(max_to_keep=50)
gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.333)

from datetime import datetime
TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())
train_log_dir = "./logs/train/"+TIMESTAMP
with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    # Run the initializer
    sess.run(init)
    starttime = datetime.now()
    merged = tf.summary.merge_all()
    train_summary_writer = tf.summary.FileWriter(train_log_dir, sess.graph)

    for step in range(1, training_steps+1):
        batch_x, batch_y , istate = data.train.next_batch(batch_size)
        # Reshape data to get 28 seq of 28 elements
        batch_x = batch_x.reshape((batch_size, timesteps, num_input))
        istate = istate.reshape(batch_size,num_spa)
        # Run optimization op (backprop)
        sess.run(train_op, feed_dict={X: batch_x, Y: batch_y, H_in:istate})
        if step % display_step == 0 or step == 1:
            # Calculate batch loss and accuracy
            summary, loss, acc = sess.run([merged, loss_op, accuracy], feed_dict={X: batch_x,
                                                                 Y: batch_y,H_in:istate})
            print("Step " + str(step) + ", Minibatch Loss= " + \
                  "{:.4f}".format(loss) + ", Training Accuracy= " + \
                  "{:.3f}".format(acc))
            train_summary_writer.add_summary(summary, step)
            filename = ('SPGRU.ckpt')
            filename = os.path.join('./model/',filename)
            saver.save(sess, filename)
            #print("best valid accuracy = " + "{:.3f}".format(best))
    print("Optimization Finished!")
    endtime = datetime.now()
    print (endtime - starttime)