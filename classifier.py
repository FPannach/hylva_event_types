#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 14:44:06 2023

@author: fpannach
"""
import os
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from joblib import dump, load

#%%
path = "enter path to data here"

data = pd.read_csv(path+"Hylemes.csv", header=0)
data_complete = data.copy(deep=True)
data_complete = data_complete[(data_complete["Final"] != "unklar")]

#%%

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

text_clf = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', MultinomialNB()),
    ])

from sklearn.model_selection import GridSearchCV   ##Cross Validation and GridSearch
parameters = {
         'vect__ngram_range': [(1, 1), (1, 3)],
         'tfidf__use_idf': (True, False),
         'clf__alpha': (1e-2, 1e-3),
    }

gs_clf = GridSearchCV(text_clf, parameters, cv=5, n_jobs=-1)  #5 fold cross validation

#%%
y = data_complete["Final"].astype(str)

X_train, X_test, y_train, y_test = train_test_split(list(data_complete["hySpo"].astype(str)), list(data_complete["Final"].astype(str)), random_state=0, train_size = .75)

y_train_bin = ["durativ" if x != "punktuell" else x for x in y_train]
y_test_bin = ["durativ" if x != "punktuell" else x for x in y_test]

#%%
gs_clf = gs_clf.fit(X_train, y_train_bin)

#%%
predicted = gs_clf.predict(X_test)
print("Classification Report binärer Classifer")
print(classification_report(y_test_bin, predicted, target_names=["durativ", "punktuell"]))
dump(gs_clf, 'binary_classifier.joblib') 
#%%
gs_clf_orig = gs_clf.fit(X_train, y_train)
predicted = gs_clf_orig.predict(X_test)
print("Classification Report fine-grained Classifer")
print(classification_report(y_test, predicted))
dump(gs_clf, 'classifier_all_classes.joblib')
#%%
data_durativ = data.copy(deep=True)
data_durativ = data_durativ[(data_durativ["Final"] != "unklar")]
data_durativ = data_durativ[(data_durativ["Final"] != "punktuell")]

#%%
X_train, X_test, y_train, y_test = train_test_split(list(data_durativ["hySpo"].astype(str)), list(data_durativ["Final"].astype(str)), random_state=0, train_size = .75)
gs_clf_durativ = gs_clf.fit(X_train, y_train)

predicted_dur = gs_clf_durativ.predict(X_test)
print("Classification Report durativer Classifier")
print(classification_report(y_test, predicted_dur, target_names=["durativ-initial", "durativ-konstant", "durativ-resultativ"]))
dump(gs_clf, 'classifier_durative.joblib')

