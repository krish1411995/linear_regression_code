# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
from sklearn import linear_model, datasets, tree
import matplotlib.pyplot as plt
number_of_samples = 100
x = np.linspace(-np.pi, np.pi, number_of_samples)
# get the random values for y here np.random.random(x.shape) is a random Guassian noise to be added to have a values other than on the line  
y = 0.5*x+np.sin(x)+np.random.random(x.shape)
#just to shuffle the x and y
random_indices = np.random.permutation(number_of_samples)
#to train the data
x_train = x[random_indices[:75]]
y_train = y[random_indices[:75]]
#data for validation
x_val = x[random_indices[75:85]]
y_val = y[random_indices[75:85]]
#Test setd
x_test = x[random_indices[85:]]
y_test = y[random_indices[85:]]
model = linear_model.LinearRegression()
 #Create a least squared error linear regression object

#sklearn takes the inputs as matrices. Hence we reshpae the arrays into column matrices
x_train_for_line_fitting = np.matrix(x_train.reshape(len(x_train),1))
y_train_for_line_fitting = np.matrix(y_train.reshape(len(y_train),1))

#Fit the line to the training data
model.fit(x_train_for_line_fitting, y_train_for_line_fitting)

#Plot the line
plt.scatter(x_train, y_train, color='purple')
plt.plot(x.reshape((len(x),1)),model.predict(x.reshape((len(x),1))),color='blue')
plt.xlabel('x-input data')
plt.ylabel('y-answer')
plt.show()
mean_val_error = np.mean( (y_val - model.predict(x_val.reshape(len(x_val),1)))**2 )
mean_test_error = np.mean( (y_test - model.predict(x_test.reshape(len(x_test),1)))**2 )

print 'Validation MSE: ', mean_val_error, '\nTest MSE: ', mean_test_error