# perceptron.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 10/27/2018

import numpy as np
import random
import time
"""
This is the main entry point for MP6. You should only modify code
within this file -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""

def classify(train_set, train_labels, dev_set, learning_rate,max_iter):
    """
    train_set - A Numpy array of 32x32x3 images of shape [7500, 3072].
                This can be thought of as a list of 7500 vectors that are each
                3072 dimensional.  We have 3072 dimensions because there are
                each image is 32x32 and we have 3 color channels.
                So 32*32*3 = 3072
    train_labels - List of labels corresponding with images in train_set
    example: Suppose I had two images [X1,X2] where X1 and X2 are 3072 dimensional vectors
             and X1 is a picture of a dog and X2 is a picture of an airplane.
             Then train_labels := [1,0] because X1 contains a picture of an animal
             and X2 contains no animals in the picture.

    dev_set - A Numpy array of 32x32x3 images of shape [2500, 3072].
              It is the same format as train_set
    """
    # TODO: Write your code here
    # return predicted labels of development set
    start = time.time()
    w = np.zeros(len(train_set[0]))

    for iter in range(max_iter):
        for i in range(len(train_set)):
            func = np.matmul(train_set[i],w) + 1
            ynot = False if np.sign(func) == -1 else True
            if ynot != train_labels[i]:
                if train_labels[i] == False:
                    curr_y = -1
                else:
                    curr_y = 1
                w = w + learning_rate * curr_y * train_set[i]

    dev_labels = []
    for image in dev_set:
        func = np.matmul(image,w)+1
        y_label = 0 if np.sign(func) == -1 else 1
        dev_labels.append(y_label)

    end = time.time()
    print(end-start,'secs')
    return dev_labels

def classifyEC(train_set, train_labels, dev_set,learning_rate,max_iter):
    # Write your code here if you would like to attempt the extra credit
    transformed_labels = np.where(train_labels == 0, -1, train_labels)

    regularization_parameter = [1e-3]

    for reg_para in regularization_parameter:
        a = np.zeros(3072)
        b = 0

        for i in range(50):
            epoch_valSet = []
            stepsize = 1/(0.01 * i + 50)
            for j in range(len(train_set)):
                # Randomly choosing one data item from the training set
                selected_data = train_set[j]
                selected_label = transformed_labels[j]
                answer = (np.dot(a, selected_data) + b) * selected_label
                #print(answer)
                if(answer >= 1):
                    a = a - stepsize * reg_para * a
                else:
                    a = a - stepsize * reg_para * a + stepsize * selected_label * selected_data
                    b = b + stepsize * selected_label
    #print(answer)
    dev_labels = []
    for image in dev_set:
        result = np.dot(a, image) + b
        if(result > 0):
            prediction = 1
        if(result < 0):
            prediction = 0
        dev_labels.append(prediction)

    return dev_labels
