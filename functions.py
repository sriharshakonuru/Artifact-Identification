import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random

def get_cat_color(index):
    palette = sns.color_palette("hls", 10)
    return palette[index]

def get_eeg_artifacts(EEG_data_struct):
    eeg_artifacts = {}
    for artifact in EEG_data_struct['EEG']['orig_urevent'][0][0][0]:
        artifact_type = artifact[0][0][0]
        artifact_latency = artifact[1][0][0]
        eeg_artifacts[artifact_latency] = artifact_type
    return eeg_artifacts

def plot_eeg(data,eeg_artifacts,node,start,end_idx,length):
    cats = list(np.unique(eeg_artifacts.values()))
    sorted_artifacts = sorted(eeg_artifacts.keys())

    for i in range(len(sorted_artifacts)):
        if start < sorted_artifacts[i]:
            curr_artifact_index = i
            break

    curr_artifact = sorted_artifacts[curr_artifact_index]
    c_sync_pulse = get_cat_color(cats.index(17))
    fig, ax = plt.subplots()
    ax.plot(range(start,curr_artifact),data[node,start:curr_artifact],color=c_sync_pulse)
    for i in range(curr_artifact_index,len(sorted_artifacts)):
        curr_artifact = sorted_artifacts[i]
        next_artifact = sorted_artifacts[i+1]
        c = get_cat_color(cats.index(eeg_artifacts[curr_artifact]))
        ax.plot(range(curr_artifact,curr_artifact+length), data[node,curr_artifact:curr_artifact+length], color=c)
        ax.plot(range(curr_artifact+length,next_artifact), data[node,curr_artifact+length:next_artifact], color=c_sync_pulse)
        if end_idx < next_artifact:
            break
    plt.show()

def load_eeg(pid):
    file_path = 'comp_artifact/B_'+'%02d'%pid+'_ART.mat'
    EEG_struct = sio.loadmat(file_path)
    return EEG_struct

def get_signal(data,eeg_artifacts, node, art_type,length,artifact_occ):
    all_latencies = [latency for latency,artifact in eeg_artifacts.iteritems() if artifact == art_type]
    all_latencies_sorted = sorted(all_latencies)
    if artifact_occ == None:
        sel_latency = random.choice(all_latencies_sorted)
    else:
        sel_latency = all_latencies_sorted[artifact_occ]
    if (node==None):
        return data[0:64,sel_latency:sel_latency+length]
    else:
        return data[node,sel_latency:sel_latency+length]
