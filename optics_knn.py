#!/usr/bin/env python
# coding: utf-8

# Neural network for optics design
# =============
# 
# The goal of this notebook is to use deep neural networks in optics design.


# These are all the modules we'll be using later. Make sure you can import them before proceeding further.
from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.utils import shuffle
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, explained_variance_score

get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'svg'")

def plot_diff(test_pred, test_labels):
  plt.figure()
  plt.plot(np.arange(len(test_pred)),test_labels,'go-',label='true value')
  plt.plot(np.arange(len(test_pred)),test_pred,'ro-',label='predict value')
  plt.title('score: %f'%score)
  plt.legend(fontsize=15)
  plt.rcParams['xtick.labelsize']=15
  plt.rcParams['ytick.labelsize']=15
  plt.show()

# load data
mat = io.loadmat('optics_design2.mat')
# print(type(mat))

dataset = mat.get('dataset')
labels = mat.get('labels')

print(dataset.shape)
print(labels.shape)
dataset, labels = shuffle(dataset, labels, random_state=42)

# scale data
dataset = StandardScaler().fit_transform(dataset)
labels = StandardScaler().fit_transform(labels)

# split data and labels. Only the first 7 items (totally 21) in the label are fitted.
labels2 = labels[:,0:7]
train_dataset, test_dataset, train_labels, test_labels = train_test_split(dataset, labels2, test_size=1/15, random_state=42)
print('Imported dataset')
print('Training set', train_dataset.shape, train_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)

# neural network regressor
clf = MLPRegressor(hidden_layer_sizes=(500,300,200,100,60,10), activation='relu', solver='adam', alpha=0.0001, batch_size='auto',
                   learning_rate='constant', learning_rate_init=0.001, power_t=0.5, max_iter=5000, shuffle=True,
                   random_state=1, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True,
                   early_stopping=False, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

clf.fit(train_dataset, train_labels)
test_pred = clf.predict(test_dataset)
score = clf.score(test_dataset, test_labels)

print("MAE: %.4f" % mean_absolute_error(test_pred, test_labels))
print("MSE: %.4f" % mean_squared_error(test_pred, test_labels))
print("R2: %.4f" % r2_score(test_pred, test_labels))
print("EV: %.4f" % explained_variance_score(test_pred, test_labels))

plot_diff(test_pred, test_labels)

# save loss
plt.plot(clf.loss_curve_)
plt.xlabel("Training iteration", Fontsize=15)
plt.ylabel("Loss", Fontsize=15)
plt.savefig('loss.jpg', dpi=300, bbox_inches='tight', pad_inches=0)

# save prediction
index = 0
plot_diff(test_pred[:,index], test_labels[:,index])
plt.savefig('test1.jpg', dpi=300, bbox_inches='tight', pad_inches=0)
plt.show()