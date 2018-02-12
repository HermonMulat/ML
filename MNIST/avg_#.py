import pickle,sys
import numpy as np
import matplotlib.pyplot as plt

def main():
    req = int(sys.argv[1])

    #reload heatMap from object
    file1 = open(r'heatMap.pkl', 'rb')
    data = pickle.load(file1)
    file1.close()

    num = np.array(data[req]).reshape(28,28)

    plt.imshow(num, cmap='hot')
    plt.colorbar()

    plt.show()

if __name__ == '__main__':
    main()
