# -*- coding: utf-8 -*-
"""
Created on Sun Apr 30 21:26:55 2017

@author: pc
"""
# Raw Lab practice problem 1
import tensorflow as tf
import numpy as np

# tf Graph Input
X = np.array([2, 3, 5, 6, 8, 9, 15, 17, 19])
Y = np.array([6.4, 9.6, 16.0, 19.2, 25.6, 28.8, 48.0, 54.4, 60.8])

# Set Wrong model Weights
W = tf.Variable(-4.0)

# Linear Model
hypothesis = """erase this and enter your code"""

# cost / loss function
cost = """erase this and enter your code"""

# Minimize : Gradient Descent
optimizer = """erase this and enter your code"""
train = """erase this and enter your code"""
# Launch the graph in a session
sess = """erase this and enter your code"""

# Initializes global variables in the graph
sess."""erase this and enter your code"""


for step in range(51):
    print(step, sess.run(W))
    sess.run(train)