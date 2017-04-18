from functions import *
import random
from scipy import signal
from sklearn.decomposition import FastICA, PCA
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import h5py

for pid in range(1,19):
    EEG_struct = load_eeg(pid)
    data = EEG_struct['EEG']['data'][0][0]

    ica = FastICA(n_components=64,max_iter=200)
    components = ica.fit_transform(data[0:64,:].T) # Reconstruct signals

    np.savez("comp_artifact/"+str(pid),components.T)
    print(str(pid)+" done")
