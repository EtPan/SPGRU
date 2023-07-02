import os 
import numpy as np
import scipy.io as sio

#divide dataset into train and test datasets
def sampling(groundTruth,TRAIN_FRAC,pca_size):              
    m = max(groundTruth)
    labeled,test,all,en,pca ={},{}, {},{},{}
    labeled_indices,test_indices,all_indices,en_indices,pca_indices =[],[], [],[],[]
    for i in range(m+1):
        indices = [j for j, x in enumerate(groundTruth.ravel().tolist()) if x == i]
        print(len(indices))
        class_num = len(indices)
        np.random.shuffle(indices)
        en[i] = indices
        en_indices += en[i]
        if i != 0:
            np.random.shuffle(indices)
            all[i] = indices
            all_indices += all[i]
            np.random.shuffle(indices)
            train_split_size = int(class_num*TRAIN_FRAC)
            labeled[i] = indices[:train_split_size]
            labeled_indices += labeled[i]
            test[i] = indices[train_split_size:]
            test_indices += test[i] 
            pca[i] = indices[train_split_size:train_split_size+pca_size]
            pca_indices+=pca[i]
            train_num.append(train_split_size)   
            test_num.append(len(test[i]))        
    np.random.shuffle(labeled_indices)
    np.random.shuffle(test_indices)
    np.random.shuffle(pca_indices)
    return labeled_indices, test_indices, all_indices,en_indices,pca_indices

mat_gt = sio.loadmat("./data/PaviaU_gt.mat")
gt_IN = mat_gt['paviaU_gt']
gt = gt_IN.reshape(np.prod(gt_IN.shape[:2]),)
pca_size=100
TRAIN_FRAC = 0.2
train_num, test_num=[],[]
labeled_indices, test_indices, all_indices,en_indices,pca_indices= sampling(gt,TRAIN_FRAC,pca_size)

print("train_each_class:",train_num)
print("train_total_num:",len(labeled_indices))
print("test_each_calss:",test_num)
print("test_total_num:",len(test_indices))
print("labeled_total_num:",len(all_indices))

os.makedirs("./data", exist_ok = True)
np.save('./data/labeled_index.npy', labeled_indices)
np.save('./data/test_index.npy', test_indices)
np.save('./data/all_index.npy', all_indices)
np.save('./data/en_index.npy', en_indices)
np.save('./data/pca_index.npy', pca_indices)