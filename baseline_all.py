import numpy as np
from sklearn import svm  
from process_data import znormalization
import time

train = np.load('./feature/train.npz')
test = np.load('./feature/test.npz')
opt = 22

x_train = train['feature'][:,0:opt]
y_train = train['label_2'].ravel()
x_test = test['feature'][:,0:opt]
y_test = test['label_2'].ravel()

X_train = znormalization(x_train)
X_test = znormalization(x_test)
X_train = np.nan_to_num(X_train)
X_test = np.nan_to_num(X_test)

clf = svm.SVC()
clf.fit(X_train, y_train)

time1 = time.time()
result = np.round(clf.predict(X_test))
time2 = time.time()
print float(time2 - time1) / float(len(y_test))
#print float(sum(result==y_test)) / float(len(y_test))

fp = 0
for i in range(len(y_test)):
	if y_test[i] == 0 and result[i] == 1:
		fp = fp + 1

print float(fp) / float(sum(y_test==0))
