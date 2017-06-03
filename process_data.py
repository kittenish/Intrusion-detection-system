import numpy as np

def znormalization(d):
    d=(d-d.mean(axis=0))/d.std(axis=0)
    
    return d

if __name__ == '__main__':
    data = np.load('./feature/tcp_finger.npz')

    x_train = data['x_train'][:,0:7]
    y_train = data['y_train'].ravel()
    x_test = data['x_test'][:,0:7]
    y_test = data['y_test'].ravel()
    
    X_train = znormalization(x_train)
    X_test = znormalization(x_test)
    
    print X_test