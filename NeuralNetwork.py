import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter("ignore")

import numpy as np
import pandas as pd
import seaborn as sb
import seaborn as sns;
import matplotlib.pyplot as plt

from sklearn import preprocessing
from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OrdinalEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression

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

datasetTrainScalled_p.describe()

#%matplotlib inline
#sb.pairplot(datasetTrain, diag_kind="kde")

linear_model = LogisticRegression(verbose=1, max_iter=10, solver="sag")
linear_model.fit(datasetTrainScalled_p, datasetTrain_y)

conf_matrix = confusion_matrix(datasetTest_y, linear_model.predict(datasetTestScalled_p))

print("Confusion_matrix:")
print(conf_matrix)

acc = accuracy_score(datasetTest_y, linear_model.predict(datasetTestScalled_p))
print("Linear regression accuracy is {0:0.2f}".format(acc))

neural_network = MLPClassifier(hidden_layer_sizes=(800,400,200), random_state=1, activation="logistic")
neural_network.fit(datasetTrainScalled_p, datasetTrain_y)
conf_matrix_neural_network = confusion_matrix(datasetTest_y, neural_network.predict(datasetTestScalled_p))

print("Confusion_matrix:")
print(conf_matrix_neural_network)

sns.heatmap(conf_matrix_neural_network)

acc = accuracy_score(datasetTest_y, neural_network.predict(datasetTestScalled_p))
print("Neural network model accuracy is {0:0.2f}".format(acc))
