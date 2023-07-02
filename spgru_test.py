import tensorflow as tf
import numpy as np
import spgru_indices


batch_size = 100
timesteps = 1
num_input = spgru_indices.INPUT_DIMENSION
num_spa = spgru_indices.num_spa
num_classes = spgru_indices.nb_classes


data = spgru_indices.read_data_sets()
saver = tf.train.import_meta_graph('./model/''SPGRU.ckpt.meta')

prediction = np.zeros((1, num_classes), dtype=np.int32)
true_label = np.zeros((1, num_classes), dtype=np.int32)
with tf.Session() as sess:
    saver.restore(sess, './model/''SPGRU.ckpt')
    y = sess.graph.get_tensor_by_name('Softmax:0')
    X = sess.graph.get_operation_by_name('X').outputs[0]
    Hin = sess.graph.get_operation_by_name('H_in').outputs[0]
    proba = sess.graph.get_tensor_by_name('add:0')
    for index in range((data.test._num_examples // batch_size)+1):
        batch,Y,istate = data.test.next_batch_test(batch_size)
        if index == (data.test._num_examples // batch_size):
            batch = batch.reshape(((data.test._num_examples % batch_size), timesteps, num_input))
            istate = istate.reshape((data.test._num_examples % batch_size),num_spa)
        else:
            batch = batch.reshape((batch_size, timesteps, num_input))
            istate = istate.reshape(batch_size,num_spa)
        pre_pro = sess.run(y, feed_dict={X: batch, Hin:istate})
        prediction = np.concatenate((prediction, pre_pro), axis=0)
        true_label = np.concatenate((true_label, Y), axis=0)
predict_label = np.argmax(prediction[1:], 1) +1
true_label = np.argmax(true_label[1:], 1) + 1
every_class, confusion_mat = spgru_indices.Cal_accuracy(true_label, predict_label, 9)
print(every_class[:])

