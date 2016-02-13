#!/usr/bin/python

import matplotlib.pyplot as plt
from prep_terrain_data import makeTerrainData
from class_vis import prettyPicture
from time import time

features_train, labels_train, features_test, labels_test = makeTerrainData()


### the training data (features_train, labels_train) have both "fast" and "slow"
### points mixed together--separate them so we can give them different colors
### in the scatterplot and identify them visually
grade_fast = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==0]
bumpy_fast = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==0]
grade_slow = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==1]
bumpy_slow = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==1]


#### initial visualization
plt.xlim(0.0, 1.0)
plt.ylim(0.0, 1.0)
plt.scatter(bumpy_fast, grade_fast, color = "b", label="fast")
plt.scatter(grade_slow, bumpy_slow, color = "r", label="slow")
plt.legend()
plt.xlabel("bumpiness")
plt.ylabel("grade")
plt.show()
################################################################################


### your code here!  name your classifier object clf if you want the 
### visualization code (prettyPicture) to show you the decision boundary


# from sklearn.neighbors import NearestNeighbors
# from sklearn.metrics import accuracy_score
# from time import time
# clf = NearestNeighbors(n_neighbors=2, algorithm='ball_tree')
# t0 = time()
# clf.fit(features_train, labels_train)
# print "training time:" , round(time() - t0, 3) , "s"

# print clf.kneighbors(features_train)


### K Nearest Neighbors Algorithm

# from sklearn.neighbors import KNeighborsClassifier

# clf = KNeighborsClassifier(n_neighbors=8, algorithm='ball_tree')
# t0 = time()
# clf.fit(features_train, labels_train)
# print "training time:" , round(time() - t0, 3) , "s"
# print clf.score(features_test, labels_test)*100
# prettyPicture(clf, features_test, labels_test)

### AdaBoost Algorithm
# from sklearn.ensemble import AdaBoostClassifier
# clf = AdaBoostClassifier(n_estimators = 100, learning_rate = 1.0)
# t0 = time()
# clf.fit(features_train, labels_train) 
# print "training time:" , round(time() - t0, 3) , "s"
# print clf.score(features_test, labels_test)*100

### Random Forest Algorithm
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(min_samples_split=50, n_estimators=100)
clf.fit(features_train, labels_train)
t0 = time()
print round(time()-t0, 3)
print clf.score(features_test, labels_test)

try:
    prettyPicture(clf, features_test, labels_test)
except NameError:
    pass
