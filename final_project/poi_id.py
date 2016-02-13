#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi','salary'] # You will need to use more features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

print len(data_dict.keys())
#print data_dict.keys()
print data_dict["MCMAHON JEFFREY"]
print "\n"
#print data_dict["MCMAHON JEFFREY"].values()
data_dict.pop("TOTAL" , 0)
### Task 2: Remove outliers
cleaned_data = []

for key in data_dict:
	if(data_dict[key]["salary"]!="NaN"):
		cleaned_data.append((key, data_dict[key]["salary"]))

cleaned_data.sort(key=lambda x:x[1], reverse=True)
print cleaned_data[:4]
print "\n"
### Task 3: Create new feature(s)
def computeFraction(poi_messages, all_messages):
	fraction = 0.

	if(poi_messages == "NaN" or all_messages == "NaN"):
		fraction = 0.
	else:
		fraction = float(poi_messages)/float(all_messages)

	return fraction

fraction_from_poi_to_this_person = []
fraction_from_this_person_to_poi = []

for user in data_dict:
	fraction_from_poi_to_this_person.append(computeFraction(data_dict[user]["from_poi_to_this_person"], data_dict[user]["to_messages"]))
	fraction_from_this_person_to_poi.append(computeFraction(data_dict[user]["from_this_person_to_poi"], data_dict[user]["from_messages"]))

print fraction_from_poi_to_this_person
print "\n"
print fraction_from_this_person_to_poi
print "\n"
### Store to my_dataset for easy export below.
count = 0
for user in data_dict:
	data_dict[user]["fraction_from_poi_to_this_person"] = fraction_from_poi_to_this_person[count]
	data_dict[user]["fraction_from_this_person_to_poi"] = fraction_from_this_person_to_poi[count]
	count +=1

my_dataset = data_dict

print my_dataset
print my_dataset["GLISAN JR BEN F"]

features_list = ["poi", "fraction_from_this_person_to_poi","fraction_from_poi_to_this_person", "salary"]
### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

from sklearn.cross_validation import KFold
kf = KFold(len(labels), 3)
for train, test in kf:
	features_train = [features[i] for i in train]
	features_test = [features[i] for i in test]
	labels_train = [labels[i] for i in train]
	labels_test = [labels[i] for i in test]
### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
from time import time
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
t0 = time()
clf.fit(features_train, labels_train)
print "training time:", round(time()-t0, 3),"s"
print clf.score(features_test, labels_test)
t0 = time()
pred = clf.predict(features_test)
print "prediction time:", round(time()-t0, 3), "s"

from sklearn.metrics import *
print precision_score(labels_test, pred)
print recall_score(labels_test, pred)
print "\n\n"
### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

### SVM Testing

from sklearn.svm import SVC
clf = SVC()
t0 = time()
clf.fit(features_train, labels_train)
print "training time:", round(time()-t0, 3),"s"
print clf.score(features_test, labels_test)
t0 = time()
pred = clf.predict(features_test)
print "prediction time:", round(time()-t0, 3), "s"

from sklearn.metrics import *
print precision_score(labels_test, pred)
print recall_score(labels_test, pred)

print "\n\n"
### DecisionTree Testing

from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier()
t0 = time()
clf.fit(features_train, labels_train)
print "training time:", round(time()-t0, 3),"s"
print clf.score(features_test, labels_test)
t0 = time()
pred = clf.predict(features_test)
print "prediction time:", round(time()-t0, 3), "s"

from sklearn.metrics import *
print precision_score(labels_test, pred)
print recall_score(labels_test, pred)


print "on different distribution of test and train data\n"
# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier()
t0 = time()
clf.fit(features_train, labels_train)
print "training time:", round(time()-t0, 3),"s"
print clf.score(features_test, labels_test)
t0 = time()
pred = clf.predict(features_test)
print "prediction time:", round(time()-t0, 3), "s"

from sklearn.metrics import *
print precision_score(labels_test, pred)
print recall_score(labels_test, pred)
### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)