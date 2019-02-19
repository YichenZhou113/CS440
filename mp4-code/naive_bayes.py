# naive_bayes.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 09/28/2018
import numpy as np
import math
import nltk

"""
This is the main entry point for MP4. You should only modify code
within this file -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""

def naiveBayes(train_set, train_labels, dev_set, smoothing_parameter):
    """
    train_set - List of list of words corresponding with each email
    example: suppose I had two emails 'i like pie' and 'i like cake' in my training set
    Then train_set := [['i','like','pie'], ['i','like','cake']]

    train_labels - List of labels corresponding with train_set
    example: Suppose I had two emails, first one was spam and second one was ham.
    Then train_labels := [0,1]

    dev_set - List of list of words corresponding with each email that we are testing on
              It follows the same format as train_set

    smoothing_parameter - The smoothing parameter you provided with --laplace (1.0 by default)
    """
    # TODO: Write your code here
    # return predicted labels of development set
    #vectorizer = CountVectorizer()
    #totals = Counter(i for i in list(itertools.chain.from_iterable(train_set)))
    my_dict = {}
    for sublist in train_set:
        for item in sublist:
            if item not in my_dict:
                my_dict[item] = 0
            my_dict[item] += 1

    train_spam_set = []
    train_ham_set = []
    for i in range(len(train_labels)):
        if train_labels[i] == 0:
            train_spam_set.append(train_set[i])
        else:
            train_ham_set.append(train_set[i])

    spam_dict = {}
    for sublist in train_spam_set:
        for item in sublist:
            if item not in spam_dict:
                spam_dict[item] = 0
            spam_dict[item] += 1

    #print(spam_prob_dict)
    ham_dict = {}
    for sublist in train_ham_set:
        for item in sublist:
            if item not in ham_dict:
                ham_dict[item] = 0
            ham_dict[item] += 1

    spam_bidict = {}
    ham_bidict = {}

    for sublist in train_spam_set:
        for i in range(len(sublist)-1):
            item = sublist[i] + sublist[i+1]
            if item not in spam_bidict:
                spam_bidict[item] = 0
            spam_bidict[item] += 1

    for sublist in train_ham_set:
        for i in range(len(sublist)-1):
            item = sublist[i] + sublist[i+1]
            if item not in ham_bidict:
                ham_bidict[item] = 0
            ham_bidict[item] += 1

    spam_prob_bidict = {}
    sum_bispam = sum(spam_bidict.values())
    for item in spam_bidict:
        spam_prob_bidict[item] = (spam_bidict[item]+smoothing_parameter)/(sum_bispam+smoothing_parameter*(len(spam_bidict)+1))

    ham_prob_bidict = {}
    sum_biham = sum(ham_bidict.values())
    for item in ham_bidict:
        ham_prob_bidict[item] = (ham_bidict[item]+smoothing_parameter)/(sum_biham+smoothing_parameter*(len(ham_bidict)+1))

    #print(ham_prob_bidict)

    spam_prob_dict = {}
    sum_spam = sum(spam_dict.values())
    for item in spam_dict:
        spam_prob_dict[item] = (spam_dict[item]+smoothing_parameter)/(sum_spam+smoothing_parameter*(len(spam_dict)+1))

    ham_prob_dict = {}
    sum_ham = sum(ham_dict.values())
    for item in ham_dict:
        ham_prob_dict[item] = (ham_dict[item]+smoothing_parameter)/(sum_ham+smoothing_parameter*(len(ham_dict)+1))


    dev_labels = []
    lamb = 0                     #lambda equals 0 is the unigram model
    for email in dev_set:
        log_spam_sum = 0
        log_ham_sum = 0
        log_spam_bisum = 0
        log_ham_bisum = 0
        for word in email:
            if word not in spam_dict:
                log_spam_sum += math.log(smoothing_parameter/(sum_spam + smoothing_parameter * (len(spam_dict)+1)))
            else:
                log_spam_sum += math.log(spam_prob_dict[word])
            if word not in ham_dict:
                log_ham_sum += math.log(smoothing_parameter/(sum_ham + smoothing_parameter * (len(ham_dict)+1)))
            else:
                log_ham_sum += math.log(ham_prob_dict[word])

        for i in range(len(email)-1):
            bi_word = email[i] + email[i+1]
            if bi_word not in spam_bidict:
                log_spam_bisum += math.log(smoothing_parameter/(sum_bispam + smoothing_parameter * (len(spam_bidict)+1)))
            else:
                log_spam_bisum += math.log(spam_prob_bidict[bi_word])
            if bi_word not in ham_bidict:
                log_ham_bisum += math.log(smoothing_parameter/(sum_biham + smoothing_parameter * (len(ham_bidict)+1)))
            else:
                log_ham_bisum += math.log(ham_prob_bidict[bi_word])

        mixed_spam_total = (1-lamb) * log_spam_sum + lamb * log_spam_bisum
        mixed_ham_total = (1-lamb) * log_ham_sum + lamb * log_ham_bisum
        if (mixed_ham_total>mixed_spam_total):
            dev_labels.append(1)
        else:
            dev_labels.append(0)


    return dev_labels
