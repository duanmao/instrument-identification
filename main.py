import numpy as np
import csv
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn import decomposition
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn import metrics

import preprocessing

# load dataset

# feature extraction
# training
# testing
# evaluation

# if __name__ == '__main__':
features = []
labels = []
with open('data.csv', newline='') as f:
    reader = csv.reader(f)
    for row in reader:
        features.append([float(num) for num in row[0].split()])
        labels.append(int(row[1]))
labelNames = preprocessing.label_names()

def train():
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, shuffle=True)
    print(np.shape(X_train))
    print(np.shape(X_test))
    
    print("Linear SVC: ")
    svc = LinearSVC()
    svc.fit(X_train, y_train)
    
    y_pred_svc = svc.predict(X_test)
    print(metrics.classification_report(y_test, y_pred_svc,
        target_names=labelNames))

    print("KNN: ")
    knn = KNeighborsClassifier()
    knn.fit(X_train, y_train)
    y_pred_knn = knn.predict(X_test)
    print(metrics.classification_report(y_test, y_pred_knn,
        target_names=labelNames))

print(labelNames)
print(np.shape(features))
print(np.shape(labels))
# print(features[:10])
# print(labels[:10])
assert(len(features) == len(labels))
train()