#!/usr/bin/python

import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from scipy import sparse
from sklearn import preprocessing

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi','salary','deferral_payments', 'total_payments',
                 'loan_advances', 'bonus', 'restricted_stock_deferred',
                 'deferred_income', 'total_stock_value', 'expenses',
                 'exercised_stock_options', 'long_term_incentive',
                 'restricted_stock', 'director_fees','from_poi_to_this_person',
                 'from_this_person_to_poi','to_messages','from_messages'] 

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Creates summary of data using a pandas dataframe
summary = pd.DataFrame.from_dict(data_dict)

### Data exploration -- counts number of selected features and total number of data points.
print "Data Exploration"
print "###########################################"

print "\nNumber of selected features: ", len(features_list)
print "Total number of data points: ", len(data_dict)

### Loops through the data_dict and counts number of POIs. The rest are non-POIs
is_poi = 0

for person in data_dict.values():
    if person["poi"]:
         is_poi = is_poi + 1
         
print "Total number of POI's: ", is_poi
print "Total number of Non-POIs: ", len(data_dict) - is_poi
print "\n###########################################"

### Task 2: Remove outliers
print "\nLet's print the list of people in the data set to check for outliers: "

### Loops through data_dict and adds people to a list titled names
names = []
for person in data_dict.keys():
    names.append(person)
    print person
    
print "\nLooking at the data there are 2 'people' that do not make sense: Total and The Travel Agency in the Park. Let's remove them from the dataset, then show the updated dataset statistics:"

### Removes TOTAL and THE TRAVEL AGENCY IN THE PARK (outliers) from the dataset.
data_dict.pop("TOTAL")
data_dict.pop("THE TRAVEL AGENCY IN THE PARK")

print "\nTotal number of data points: ", len(data_dict)
print "\nYou can see that the name outliers have been removed."
print "\n###########################################"

### Task 3: Create new feature(s)

print "\nCreating New Features"
my_dataset = data_dict

print '\nA reasonable place to start for creating new features is to find out which people working at Enron had the most communication with known POIs'
print "\nFirst we will create a feature to hold the ratio of total messages sent that were to POIs,"

### Loops through values in my_dataset and creates a new feature 'ratio_of_messages_to_poi' using original features
for person in my_dataset.values():
    person['ratio_of_messages_to_poi'] = 0.0
    if float(person['from_messages']) > 0.0:
        person['ratio_of_messages_to_poi'] = float(person['from_this_person_to_poi']) / float(person['from_messages'])       
        x = person['ratio_of_messages_to_poi']
        
print "then we will create a feature to hold the ratio of total messages sent that were from POIs."

### Loops through values in my_dataset and creates a new feature 'ratio_of_messages_from_poi' using original features
for person in my_dataset.values():
    person['ratio_of_messages_from_poi'] = 0.0
    if float(person['to_messages']) > 0.0:
        person['ratio_of_messages_from_poi'] = float(person['from_poi_to_this_person']) / float(person['to_messages'])
        y = person['ratio_of_messages_from_poi']

### Adding 2 new features to the features_list list
features_list.extend(['ratio_of_messages_to_poi', 'ratio_of_messages_from_poi'])

print "We can see below that 2 new features have been added to features_list by using the len() function."
print "\nNumber of features: ", len(features_list)
print "\n###########################################"
print "\nIntelligently Select Features"
print "\nNow we will use a K-Best test to select optimal features."

### Pulling labels and features from data set
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Defines returnElement function for use in sorted_scores below
def returnElement(elem):
    return elem[1]
    
from sklearn.feature_selection import SelectKBest, f_classif

k_best = SelectKBest(f_classif, k = 10)
k_best.fit(features, labels)
sorted_scores = sorted(zip(features_list[1:], k_best.scores_), key = returnElement, reverse = True)

### Prints sorted scores
print 'K-Best sorted scores: '
for elem in sorted_scores:
    print elem

print "\nUnsurprisingly, Exercised Stock Options, Stock Value, Bonus, and Salary are features that seem to carry the most weight. Interestingly, our created feature Ratio of Messages to POI had the 5th highest K-Value."
print "\n###########################################"

### Task 4: Try a varity of classifiers

print "\nExploring Classifiers"
print "\nMetrics using Gaussian Naive Bayes: "

### Splitting Features and Labels into Training and Testing Data
from sklearn.cross_validation import train_test_split
from sklearn import cross_validation

features_train, features_test, labels_train, labels_test = \
cross_validation.train_test_split(features, labels, test_size = .25, random_state = 9)

### Gaussian Naive Bayes Classifier
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn.metrics import accuracy_score, recall_score, precision_score
from time import time

clf = GaussianNB()
clf.fit(features_train, labels_train)
predictor = clf.predict(features_test)

NB_accuracy = accuracy_score(labels_test, predictor)
NB_recall = recall_score(labels_test, predictor)
NB_precision = precision_score(labels_test, predictor)

print "\nNaive Bayes accuracy: ", round(NB_accuracy, 2)
print "Naive Bayes recall: ", round (NB_recall, 2)
print "Naive Bayes precision: ", round(NB_precision, 2)

### Decision Tree Classifier
print "\nMetrics using Decision Tree: "

clf = tree.DecisionTreeClassifier()
clf.fit(features_train, labels_train)
predictor = clf.predict(features_test)

DT_accuracy = accuracy_score(labels_test, predictor)
DT_recall = recall_score(labels_test, predictor)
DT_precision = precision_score(labels_test, predictor)

print "\nDecision Tree accuracy: ", round(DT_accuracy, 2)
print "Decision Tree recall: ", round(DT_recall, 2)
print "Decision Tree precision: ", round(DT_precision, 2)

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. 


from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size = None, random_state = None)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)