import numpy as np
import collections
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from matplotlib import pyplot as plt
import joblib

data = np.load('data.npy')
target = np.load('target.npy')

print(data.shape)
print(target.shape)
print(collections.Counter(target))

result = train_test_split(data, target, test_size=0.2)
train_data = result[0]
test_data = result[1]
train_target = result[2]
test_target = result[3]

'''pca = PCA()  # THIS CODE IS JUST FOR DECIDE THE n_compo that how many features should be needed
pca.fit(data)

plt.plot(np.cumsum(pca.explained_variance_ratio_))

plt.xlabel('NO of comp')
plt.ylabel('Comm varriance')
plt.show()'''
pca = PCA(n_components=7, whiten=True, random_state=42)
svc = SVC()
model = make_pipeline(pca, svc)  # creating pipeline

model.fit(train_data, train_target)

predict_target = model.predict(test_data)

acc = accuracy_score(test_target, predict_target)
print("ACCURACY:-", acc)

print(classification_report(test_target, predict_target))

joblib.dump(model, "SVM-Face-Recognition.sav")
print("FILE SAVED!!")