import numpy as np
import pandas as pd

from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OrdinalEncoder
from sklearn.neural_network import MLPClassifier

import os
path = os.path.dirname(os.path.abspath(__file__))
print(path)

datasetFull = pd.read_csv(f"{path}/bank-additional-full.csv", delimiter=';')
dataset = pd.read_csv(f"{path}/bank-additional.csv", delimiter=';')
datasetFull.head()

(datasetTrain, datasetTrain_y) = datasetFull[datasetFull.columns[0:20]], datasetFull[datasetFull.columns[-1]]
(datasetTest, datasetTest_y) = dataset[dataset.columns[0:20]], dataset[dataset.columns[-1]]

datasetTrainScalled_p = pd.DataFrame(StandardScaler().fit_transform(OrdinalEncoder().fit_transform(datasetTrain)), columns=[datasetTrain.columns])
datasetTestScalled_p = pd.DataFrame(StandardScaler().fit_transform(OrdinalEncoder().fit_transform(datasetTest)), columns=[datasetTest.columns])

print(datasetTrainScalled_p.describe())

parameters = {'solver': ['adam', 'sgd'], 'max_iter': [1000], \
              'hidden_layer_sizes':[(800,400,200), (900,300,100)], \
              'activation': ['logistic','relu'], 'random_state': [1]}
grid_search = GridSearchCV(MLPClassifier(), parameters, n_jobs=-1, verbose=5, cv=4)
grid_search.fit(datasetTrainScalled_p.values,datasetTrain_y.values)

print(grid_search.score(datasetTestScalled_p.values, datasetTest_y.values))
print(grid_search.best_params_)
