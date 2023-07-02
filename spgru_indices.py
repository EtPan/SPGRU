from __future__ import print_function
import numpy as np
import scipy.io as sio
from sklearn.decomposition import PCA

mat_gt = sio.loadmat("./data/PaviaU_gt.mat")
label = mat_gt['paviaU_gt']
GT = label.reshape(np.prod(label.shape[:2]),)
data_mat=sio.loadmat("./data/PaviaU.mat")
data_IN = data_mat['paviaU']

INPUT_DIMENSION = data_IN.shape[2]
nb_classes = len(np.unique(GT))-1

windowSize= 9
numComponents=4
num_spa = windowSize*windowSize*numComponents


normdata = np.zeros((data_IN.shape[0], data_IN.shape[1], data_IN.shape[2]), dtype=np.float32)
for dim in range(data_IN.shape[2]):
    normdata[:, :, dim] = (data_IN[:, :, dim] - np.amin(data_IN[:, :, dim])) / \
                          float((np.amax(data_IN[:, :, dim]) - np.amin(data_IN[:, :, dim])))

labeled_sets = np.load('./data/labeled_index.npy')
test_sets    = np.load('./data/test_index.npy')
all_sets     = np.load("./data/all_index.npy")
en_sets     = np.load("./data/en_index.npy")

def applyPCA(X, numComponents=75):
    newX = np.reshape(X, (-1, X.shape[2]))
    pca = PCA(n_components=numComponents, whiten=True)
    newX = pca.fit_transform(newX)
    newX = np.reshape(newX, (X.shape[0],X.shape[1], numComponents))
    return newX, pca
def padWithZeros(X, margin=2):
    newX = np.zeros((X.shape[0] + 2 * margin, X.shape[1] + 2* margin, X.shape[2]))
    x_offset = margin
    y_offset = margin
    newX[x_offset:X.shape[0] + x_offset, y_offset:X.shape[1] + y_offset, :] = X
    return newX
def dense_to_one_hot(labels_dense, num_classes=nb_classes):
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()-1] = 1
    return labels_one_hot


data_pca,pca = applyPCA(data_IN,numComponents=numComponents)
normpca = np.zeros((data_pca.shape[0], data_pca.shape[1],data_pca.shape[2]), dtype=np.float32)
for dim in range(data_pca.shape[2]):
    normpca[:, :, dim] = (data_pca[:, :, dim] - np.amin(data_pca[:, :, dim])) / \
                          float((np.amax(data_pca[:, :, dim]) - np.amin(data_pca[:, :, dim])))
margin = int((windowSize - 1) / 2)
padded_data=padWithZeros(normpca,margin=margin)

class DataSet(object):
  def __init__(self, images):
    self._num_examples = images.shape[0]
    self._images = images
    self._epochs_completed = 0
    self._index_in_epoch = 0
  @property
  def images(self):
    return self._images
  @property
  def num_examples(self):
    return self._num_examples
  @property
  def epochs_completed(self):
    return self._epochs_completed

  def next_batch(self, batch_size):
    """Return the next `batch_size` examples from this data set."""
    start = self._index_in_epoch
    self._index_in_epoch += batch_size
    if self._index_in_epoch > self._num_examples:
      # Finished epoch
      self._epochs_completed += 1
      # Shuffle the data
      perm = np.arange(self._num_examples)
      np.random.shuffle(perm)
      self._images = self._images[perm]
      # Start next epoch
      start = 0
      self._index_in_epoch = batch_size
      assert batch_size <= self._num_examples
    end = self._index_in_epoch
    hsi_batch_pca = np.zeros((batch_size, windowSize, windowSize, numComponents), dtype=np.float32)
    col_pca = data_pca.shape[1]
    hsi_batch_patch = np.zeros((batch_size, INPUT_DIMENSION), dtype=np.float32)
    col = data_IN.shape[1]
    for q1 in range(batch_size):
      hsi_batch_patch[q1] = normdata[(self._images[start + q1] // col), (self._images[start + q1] % col), :]
      hsi_batch_pca[q1] = padded_data[(self._images[start + q1] // col_pca):
                                          ((self._images[start + q1] // col_pca) + windowSize),
                              (self._images[start + q1] % col_pca):
                              ((self._images[start + q1] % col_pca) + windowSize), :]
      #hsi_batch_patch[q1]=tf.concat[(hsi_batch_patch(q1),hsi_batch_pca(q1))]
    block = self._images[start:end]
    hsi_batch_label = GT[block]
    hsi_batch_label = dense_to_one_hot(hsi_batch_label, num_classes=nb_classes)
    return hsi_batch_patch, hsi_batch_label,hsi_batch_pca

  def next_batch_spe(self, batch_size):
    """Return the next `batch_size` examples from this data set."""
    start = self._index_in_epoch
    self._index_in_epoch += batch_size
    if self._index_in_epoch > self._num_examples:
      # Finished epoch
      self._epochs_completed += 1
      # Shuffle the data
      perm = np.arange(self._num_examples)
      np.random.shuffle(perm)
      self._images = self._images[perm]
      # Start next epoch
      start = 0
      self._index_in_epoch = batch_size
      assert batch_size <= self._num_examples
    end = self._index_in_epoch
    hsi_batch_patch = np.zeros((batch_size, INPUT_DIMENSION), dtype=np.float32)
    col = data_IN.shape[1]
    for q1 in range(batch_size):
      hsi_batch_patch[q1] = normdata[(self._images[start + q1] // col), (self._images[start + q1] % col), :]
    block = self._images[start:end]
    hsi_batch_label = GT[block]
    hsi_batch_label = dense_to_one_hot(hsi_batch_label, num_classes=nb_classes)
    return hsi_batch_patch, hsi_batch_label

  def next_batch_test(self, batch_size):
    start = self._index_in_epoch
    self._index_in_epoch += batch_size
    if self._index_in_epoch > self._num_examples:
        self._index_in_epoch = self._num_examples
    end = self._index_in_epoch
    hsi_batch_pca = np.zeros((end-start, windowSize, windowSize, numComponents), dtype=np.float32)
    col_pca = data_pca.shape[1]
    hsi_batch_patch = np.zeros((end-start, INPUT_DIMENSION), dtype=np.float32)
    col = data_IN.shape[1]
    for q1 in range(end-start):
      hsi_batch_patch[q1] = normdata[(self._images[start + q1] // col),(self._images[start + q1] % col),:]
      hsi_batch_pca[q1] = padded_data[(self._images[start + q1] // col_pca):
                                          ((self._images[start + q1] // col_pca) + windowSize),
                              (self._images[start + q1] % col_pca):
                              ((self._images[start + q1] % col_pca) + windowSize), :]
    block = self._images[start:end]
    hsi_batch_label = GT[block]
    hsi_batch_label = dense_to_one_hot(hsi_batch_label, num_classes=nb_classes)
    return hsi_batch_patch,hsi_batch_label,hsi_batch_pca


def read_data_sets():
    class DataSets(object):
        pass
    data_sets = DataSets()
    data_sets.train = DataSet(labeled_sets)
    data_sets.en = DataSet(en_sets)
    data_sets.test = DataSet(test_sets)
    data_sets.all = DataSet(all_sets)
    return data_sets

from sklearn.metrics import confusion_matrix
def Cal_accuracy(true_label,pred_label,class_num):
    M = 0
    C = np.zeros((class_num+1,class_num+1))
    c1 = confusion_matrix(true_label, pred_label)
    C[0:class_num,0:class_num] = c1
    C[0:class_num,class_num] = np.sum(c1,axis=1)
    C[class_num,0:class_num] = np.sum(c1,axis=0)
    N = np.sum(np.sum(c1,axis=1))
    C[class_num,class_num] = N   # all of the pixel number
    OA = np.trace(C[0:class_num,0:class_num])/N
    every_class = np.zeros((class_num+3,))
    for i in range(class_num):
        acc = C[i,i]/C[i,class_num]
        M = M + C[class_num,i] * C[i,class_num]
        every_class[i] = acc
    kappa = (N * np.trace(C[0:class_num,0:class_num]) - M)/(N*N-M)
    AA = np.sum(every_class,axis=0)/class_num
    every_class[class_num] = OA
    every_class[class_num+1] = AA
    every_class[class_num+2] = kappa
    return every_class, C

