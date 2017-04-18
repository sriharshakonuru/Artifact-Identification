#!/usr/bin/env python
import numpy as np
import misvm
from saxpy import SAX
import matplotlib.pyplot as plt
from collections import defaultdict
import operator
import pickle
import sys

# Load Data
bags_file = np.load("comp_artifact/bags.npz")
labels_file = np.load("comp_artifact/labels.npz")
bags = list(bags_file['arr_0'])
labels = list(labels_file['arr_0'])

# Parameters
# word_size = 8 # Entire time series in these many characters
# alphabet_size = 4 # Total number of characters
# num_subsequences = 80 #windowSize = int(len(x)/numSubsequences)
# overlapping_fraction = 0.9 #overlap = window_size*overlapping_fraction
# s = SAX(word_size, alphabet_size)
# max_features = 8000

word_size = int(sys.argv[1]) # Entire time series in these many characters
alphabet_size = int(sys.argv[2]) # Total number of characters
num_subsequences = int(sys.argv[3]) #windowSize = int(len(x)/numSubsequences)
overlapping_fraction = float(sys.argv[4]) #overlap = window_size*overlapping_fraction
s = SAX(word_size, alphabet_size)
max_features = int(sys.argv[5])

print("Word Size : ",word_size)
print("Alphabet Size : ",alphabet_size)
print("Number of subsequences : ",num_subsequences)
print("Overlapping Fraction : ",overlapping_fraction)
print("Max Features : ",max_features)

# Get feature list
vocab = dict()
count = 0
for bag in bags:
    for component in bag:
        (sax_str, sax_ind) = s.sliding_window(component, num_subsequences, overlapping_fraction)
        for word in sax_str:
            if word in vocab:
                vocab[word] = vocab[word] + 1
            else:
                vocab[word] = 1
    print("Bag Done : "+str(count))
    count = count + 1
sorted_vocab = sorted(vocab.items(), key=operator.itemgetter(1),reverse=True)
vocab_features = sorted_vocab[0:max_features]
pickle.dump(vocab_features,open("model/vocab_features.pkl","wb"))

print("-------------------------------Generating Features--------------------------------------")
# Generate features
bags_features =[]
count = 0
for bag in bags:
    num_components = bag.shape[0]
    feature = np.ndarray((num_components,max_features))
    for i in range(len(bag)):
        component = bag[i]
        (sax_str, sax_ind) = s.sliding_window(component, num_subsequences, overlapping_fraction)
        for j in range(max_features):
            feature[i][j] = sax_str.count(vocab_features[j][0])
    bags_features.append(feature)
    print("Feature Generation Done : "+str(count))
    count = count + 1

np.savez("comp_artifact/bags_features",bags_features)
