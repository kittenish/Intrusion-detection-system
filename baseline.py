import numpy as np
from sklearn import svm  
from process_data import znormalization
import time

data = np.load('./feature/tcp_finger.npz')

x_train = data['x_train'][:,0:7]
y_train = data['y_train'].ravel()
x_test = data['x_test'][:,0:7]
y_test = data['y_test'].ravel()

X_train = znormalization(x_train)
X_test = znormalization(x_test)
X_train = np.nan_to_num(X_train)
X_test = np.nan_to_num(X_test)
print X_train
clf = svm.SVC()
clf.fit(X_train, y_train)

time1 = time.time()
result = np.round(clf.predict(X_test))
time2 = time.time()
print float(time2 - time1) / float(len(y_test))
print float(sum(result==y_test)) / float(len(y_test))