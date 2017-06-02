import numpy as np
from sklearn import svm  

train = np.load('./feature/train.npz')
test = np.load('./feature/test.npz')

feature = train['feature']
label_2 = train['label_2']
label_5 = train['label_5']
label_23 = train['label_23']
label_2 = label_2.ravel()
test_f = test['feature'][:,0:9]
test_l = test['label_2']
test_l = test_l.ravel()

clf = svm.SVC()
clf.fit(feature[:,0:9], label_2)

result = np.round(clf.predict(test_f))
print float(sum(result==test_l)) / float(len(test_l))