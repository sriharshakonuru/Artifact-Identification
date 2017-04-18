from functions import *
import random
from scipy import signal
from sklearn.decomposition import FastICA, PCA
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import h5py

for i in range(1,19):
    components = np.load("comp_artifact/"+str(i)+".npz")
    data = components['arr_0']
    np.save(open("comp_artifact/"+str(i)+".npy","w"),data)
    components.close()

for i in range(1,19):
    if i == 4 or i == 16 :
        continue
    EEG_struct = load_eeg(i)
    eeg_artifacts = get_eeg_artifacts(EEG_struct)
    np.save("comp_artifact/"+str(i)+".artifacts",eeg_artifacts)
