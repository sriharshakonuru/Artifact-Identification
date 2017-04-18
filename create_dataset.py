from functions import *
import random
from scipy import signal
from sklearn.decomposition import FastICA, PCA
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import h5py
import gc

def generate_negative_bag(n_samples):
    pid = random.randint(1,18)
    while pid == 4 or pid == 16 :
        pid = random.randint(1,18)
    print(pid)
    eeg_artifacts = np.load("comp_artifact/"+str(pid)+".artifacts.npy").item()
    data = np.load("comp_artifact/"+str(pid)+".npz")['arr_0']
    signal1 = get_signal(data,eeg_artifacts, None, 17,n_samples,None)
    return signal1.copy()

def generate_positive_bag(n_samples):
    pid = random.randint(1,18)
    while pid == 4 or pid == 16 :
        pid = random.randint(1,18)
    print(pid)
    eeg_artifacts = np.load("comp_artifact/"+str(pid)+".artifacts.npy").item()
    data = np.load("comp_artifact/"+str(pid)+".npz")['arr_0']
    signal1 = get_signal(data,eeg_artifacts, None, 106,n_samples,None)
    return signal1.copy()

def generate_bag(bag_label,n_samples):
    if bag_label == 1: # Positive label
        return generate_positive_bag(n_samples)
    else:
        return generate_negative_bag(n_samples)

def generate_bags_and_labels(n_bags,n_samples):
    bags = []
    labels = []
    for i in range(n_bags):
        bag_label = random.choice([-1,1])
        bag = generate_bag(bag_label,n_samples)
        bags.append(bag)
        labels.append(bag_label)
    return bags,labels

random.seed(10)
n_samples = 1000
n_bags = 500

bags,labels = generate_bags_and_labels(n_bags,n_samples)
np.savez("comp_artifact/bags",bags)
np.savez("comp_artifact/labels",labels)
