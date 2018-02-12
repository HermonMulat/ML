from read_mnist import load_data, pretty_print
import numpy as np
from PIL import Image
import random,pickle
import matplotlib.pyplot as plt

FEATURE = 0
LABEL = 1

feature_vals = [set() for i in xrange(784)]
feature_sums = [0 for i in xrange(784)]

def main():
    train_set, test_set = load_data()
    train_set_size = len(train_set[FEATURE])

    heat_map = dict((i,[0 for i in xrange(784)])for i in xrange(10))
    count_map = list(range(10))
    for feature,lable in zip(train_set[FEATURE],train_set[LABEL]):
        for i in xrange(784):
            heat_map[lable][i] += feature[i]
            count_map[lable]+=1

    #average
    for num,psum in heat_map.items():
        heat_map[num] = [k/count_map[num] for k in psum]
    afile = open(r'heatMap.pkl', 'wb')
    pickle.dump(heat_map, afile)
    afile.close()
    # feature_count = [(k,len(feature_vals[k])) for k in xrange(784)]
    # feature_count.sort(key = lambda x:x[0])
    # avg_img = [feature_sums[i]/train_set_size for i in xrange(784)]
    #feature_sum.sort(key = lambda x:x[1])
    #
    # count_map = np.array([len(k) for k in feature_vals]).reshape(28,28)
    # avg = np.array(avg_img).reshape(28,28)
    #
    # plt.imshow(count_map, cmap='hot')
    # plt.colorbar()
    #
    # plt.show()

if __name__ == '__main__':
    main()
