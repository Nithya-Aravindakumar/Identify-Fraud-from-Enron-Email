
# coding: utf-8

# In[11]:

import matplotlib.pyplot 
import sys
import numpy
import pickle
from time import time
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn import cross_validation
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn import grid_search

sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit
import tester

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list =  ['poi','salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus', 
                  'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses', 
                  'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock', 
                  'director_fees', 'to_messages', 'from_poi_to_this_person', 'from_messages', 
                  'from_this_person_to_poi', 'shared_receipt_with_poi', 'email_address'] 


# Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)
        
# Finding total number of data points
print ("No. of data points:")
print(len(data_dict))

# Finding number of features
for key in data_dict.iterkeys():
    print key
    print data_dict[key]
    print "--------------"

# Number of POI and non-POI in the data set
poi_count = 0
non_poi_count = 0
for employee in data_dict:
    for feature,feature_value in data_dict[employee].iteritems():
        if (feature=="poi") & (feature_value==1):
            poi_count+=1
        if (feature=="poi") & (feature_value==0):
            non_poi_count+=1
print "poi_count:", poi_count
print "non_poi_count:",non_poi_count

# percentage of NaN in each feature
def count_nan(feature_name):
    """Prints the percentage of NaN in each feature"""
    count=0
    for employee in data_dict:
        for feature,feature_value in data_dict[employee].iteritems():
            if (feature == feature_name) & (feature_value=='NaN'):
                count += 1
    print feature_name            
    print ((count/146.0)*100.0)

print("\nNumber of NaN in each feature")
for elem in features_list:
    count_nan(elem)

# Find people with all values as NAN
for employee in data_dict:
    count=0
    for feature,feature_value in data_dict[employee].iteritems():
        if feature_value=='NaN':
            count+=1
    if count >=20:     #there are 20 features excluding poi
        print employee

# Remove email_address from feature list 
features_list.remove('email_address') 


# In[12]:

### Task 2: Remove outliers
def scatter_plot(feature1, feature2): 
    """creates scatterplot between 2 features"""
    data = featureFormat(data_dict, [feature1, feature2])
    for point in data:
        feature1 = point[0]
        feature2 = point[1]
        matplotlib.pyplot.scatter( feature1, feature2 )

    matplotlib.pyplot.xlabel(feature1)
    matplotlib.pyplot.ylabel(feature2)
    matplotlib.pyplot.show()
    
scatter_plot("salary","bonus")
data_dict.pop( "TOTAL", 0 ) # removes the outlier 'TOTAL' from the dictionary
data_dict.pop("THE TRAVEL AGENCY IN THE PARK") # remove non-employee detail
data_dict.pop("LOCKHART EUGENE E") # remove person with no non-NAN values
scatter_plot("salary","bonus")


# In[13]:

### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict

# Creating poi_interaction feature
def interactions_with_poi(poi_messages, all_messages):
    """calculates interactions with poi by finding
    percentage of ratio of poi messages to all messages.
    """
    percent = 0
    percent = (poi_messages/float(all_messages))*100
    return percent


for employee in my_dataset:   
    # Defining poi_messages variable as the sum of from_poi_to_this_person and from_this_person_to_poi
    from_poi_to_this_person = my_dataset[employee]['from_poi_to_this_person']
    from_this_person_to_poi = my_dataset[employee]['from_this_person_to_poi']
    
    if from_poi_to_this_person != 'NaN' and from_this_person_to_poi != 'NaN':
        poi_messages = my_dataset[employee]['from_poi_to_this_person'] + my_dataset[employee]['from_this_person_to_poi']
    else:
        poi_messages = 999999 # Done to prevent addition of strings i.e.'NaN'. 

    # Defining all_messages variable as the sum of to_messages and from_messages
    to_messages = my_dataset[employee]['to_messages']
    from_messages = my_dataset[employee]['from_messages']
   
    if to_messages != 'NaN' and from_messages != 'NaN':
        all_messages = to_messages + from_messages
    else:
        all_messages = 1  # Done to prevent addition of strings i.e.'NaN'. 
        
    poi_interaction = interactions_with_poi(poi_messages, all_messages)
    my_dataset[employee]['poi_interaction'] = poi_interaction

# Remove values that were created to avoid addition of 'NaN' strings
for employee in my_dataset:
    if my_dataset[employee]['poi_interaction'] > 100:
        my_dataset[employee]['poi_interaction'] = 0

# creating salary_bonus feature as the sum of salary and bonus
for employee in my_dataset:
    salary = my_dataset[employee]['salary']
    bonus = my_dataset[employee]['bonus'] 
    if salary != 'NaN' and bonus != 'NaN':
        salary_bonus = salary + bonus
    my_dataset[employee]['salary_bonus'] = salary_bonus
    
features_list_new = features_list
features_list_new =  features_list_new + ['poi_interaction'] + ['salary_bonus']
print '\nFeatures List with 2 new features: ',features_list_new
        
### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list_new, sort_keys = True)
labels, features = targetFeatureSplit(data)

#univariate feature selection 

def select_second_element(element):
    """ To return the second element"""
    return element[1]

scaler = preprocessing.MinMaxScaler()
features = scaler.fit_transform(features)

selector = SelectKBest(f_classif, k = 5)
selector.fit(features, labels)
scores = zip(features_list_new[1:], selector.scores_)  #poi has been excluded while creating tuples
scores_sorted = sorted(scores, key = select_second_element, reverse = True) #choosing 2nd element of tuple as key for sorting
print '\nFeatures with scores: ', scores_sorted
kBest_features = [(j[0]) for j in scores_sorted[0:5]]
print '\nKBest Features:', kBest_features
kbest_with_poi =  ['poi'] + kBest_features

print '\nkbest_with_poi:',kbest_with_poi

data = featureFormat(my_dataset, kbest_with_poi, sort_keys = True)
labels, features = targetFeatureSplit(data)


# In[14]:

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Try a variety of classifiers.

features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(features, labels, test_size=0.3, random_state=42)

def train_predict(clf):
    """ Trains and predicts according to given classifier.
    Cross-validation is done based on StratifiedShuffleSplit.
    Training time and Predicting time are also determined.
    """
    t0 = time()
    clf.fit(features_train, labels_train)
    print "Training time:", round(time()-t0, 3), "s"
    t1 = time()
    pred = clf.predict(features_test)
    print "Predicting time:", round(time()-t1, 3), "s"
    tester.test_classifier(clf, my_dataset, kbest_with_poi)

# Naive Bayes 
print "Naive Bayes"
clf1 = GaussianNB()
train_predict(clf1)

# Decision Tree
print "\nDecision Tree"
clf3 = tree.DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,
            max_features=None, max_leaf_nodes=None,
            min_impurity_split=1e-07, min_samples_leaf=1,
            min_samples_split=2, min_weight_fraction_leaf=0.0,
            presort=False, random_state=None, splitter='random')
train_predict(clf3)

# KNeighbors
print "\nKNeighbors"
clf4 = KNeighborsClassifier(n_neighbors = 3, metric = 'manhattan',  weights='uniform')
train_predict(clf4)

#AdaBoost
print "\nAdaBoost"
clf5 = AdaBoostClassifier(algorithm='SAMME.R', base_estimator=None,
          learning_rate=1.0, n_estimators=50, random_state=42)
train_predict(clf5)

#Random Forest
print "\nRandomForest"
clf6 = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=None, max_features='auto', max_leaf_nodes=None, 
            min_samples_leaf=1, min_samples_split=2, min_weight_fraction_leaf=0.0,
            n_estimators=10, n_jobs=1, oob_score=False, random_state=42,
            verbose=0, warm_start=False)
train_predict(clf6)


# In[15]:

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script.
    
from sklearn.grid_search import GridSearchCV 
from sklearn.cross_validation import StratifiedShuffleSplit

def optimal_parameters(classifier, parameters):
    """ Determines optimal parameters for classifiers 
    using StratifiedShuffleSplit and GridSearchCV
    """
    sss=StratifiedShuffleSplit(labels, 100, test_size=0.3, random_state=42) 
    grid = GridSearchCV(classifier,parameters,scoring='f1', cv = sss) 
    grid.fit(features, labels) 
    clf = grid.best_estimator_ 
    return clf

        
# Decision Tree
dt_parameters = {
    'criterion':('gini', 'entropy'),
    'splitter':('best','random')
}
dt = tree.DecisionTreeClassifier()
print ("\nDecision Tree") 
clf3 = optimal_parameters(dt, dt_parameters)
print clf3

# KNeighbors
kn_parameters = {
    'n_neighbors': [1, 3, 5, 10], 
    'weights':('uniform', 'distance'), 
    'algorithm':('auto', 'ball_tree', 'kd_tree', 'brute'),
    'metric': ['manhattan', 'euclidean', 'chebyshev']
}
kn = KNeighborsClassifier()
print ("\nK Neighbors Classifier")
clf4 = optimal_parameters(kn, kn_parameters)
print clf4

#AdaBoostClassifier 
ab_parameters = {
    'n_estimators': [1, 10, 50], 
}
ab = AdaBoostClassifier()
print ("\nAdaBoost Classifier")
clf5 = optimal_parameters(ab, ab_parameters)
print clf5

#RandomForestClassifier
parameters = {
    'criterion':('gini', 'entropy'),
    'n_estimators':[1, 10, 100]
}
rf =RandomForestClassifier()
print ("\nRandom Forest")
clf6 = optimal_parameters(rf, parameters)
print clf6


# In[16]:

from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA

# GaussianNB with PCA
pipe = Pipeline([('pca', PCA()),
                 ('nb', GaussianNB())])

pipe.fit(features_train, labels_train)

pred = pipe.predict(features_test)

tester.test_classifier(pipe, my_dataset, kbest_with_poi)


# In[17]:

clf = clf1
features_list = kbest_with_poi


# In[18]:

tester.dump_classifier_and_data(clf, my_dataset, features_list)


# In[19]:

tester.main()


# In[ ]:



