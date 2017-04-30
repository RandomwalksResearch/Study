# -*- coding: utf-8 -*-
"""
Created on Sun Apr 30 21:26:55 2017

@author: pc
"""
# Raw Lab practice problem 1
import tensorflow as tf
import numpy as np

# tf Graph Input
X = np.array([3, 8, 19, 21, 40])
Y = np.array([9.6, 25.6, 60.8, 67.2, 128.])

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


for step in range(201):
    print(step, sess.run(W))
    sess.run(train)