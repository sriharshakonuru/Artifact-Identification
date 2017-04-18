import numpy as np
import misvm
import pickle
from scipy import signal
from sklearn.decomposition import FastICA, PCA
from saxpy import SAX
from functions import *

pid = 1
data = np.load("comp_artifact/"+str(pid)+".npz")['arr_0']
eeg_artifacts = np.load("comp_artifact/"+str(pid)+".artifacts.npy").item()
#Load Vocabulary and model
vocab_features = pickle.load(open("model/vocab_features.pkl"))
classifiers = pickle.load(open("model/model11.pkl",'rb'))
count = 0

eegdat = {}

####################################################################
#### Store all the artifact data samples in a dictionary############

for i in range(len(eeg_artifacts)):
    if eeg_artifacts.values()[i] == 17:
        eegdat[i] = eeg_artifacts.keys()[i]

artifacts = {}

####################################################################
#### Store all the artifact data samples in a dictionary############
for i in range(len(eeg_artifacts)):
    if eeg_artifacts.values()[i] == 106:
        artifacts[i] = eeg_artifacts.keys()[i]


####################################################################    

for i in range(50):

    artifact1 = get_signal(data,eeg_artifacts, None, 106,1000,0)
    artifact2 = get_signal(data,eeg_artifacts, None, 106,1000,1)
    artifact3 = get_signal(data,eeg_artifacts, None, 106,1000,2)
    #artifact4 = get_signal(data,eeg_artifacts, None, 106,1000,artifacts.values()[3])
    # artifact5 = get_signal(data,eeg_artifacts, None, 106,1000,None)
    # artifact6 = get_signal(data,eeg_artifacts, None, 106,1000,None)
    # artifact7 = get_signal(data,eeg_artifacts, None, 106,1000,None)
    # artifact8 = get_signal(data,eeg_artifacts, None, 106,1000,None)
    # artifact9 = get_signal(data,eeg_artifacts, None, 106,1000,None)
    # artifact10 = get_signal(data,eeg_artifacts, None, 106,1000,None)
    # artifact11= get_signal(data,eeg_artifacts, None, 106,1000,None)
    # artifact12 = get_signal(data,eeg_artifacts, None, 106,1000,None)
    # artifact13 = get_signal(data,eeg_artifacts, None, 106,1000,None)
    # artifact14= get_signal(data,eeg_artifacts, None, 106,1000,None)
    # artifact15 = get_signal(data,eeg_artifacts, None, 106,1000,None)
    # artifact16 = get_signal(data,eeg_artifacts, None, 106,1000,None)
    # artifact17 = get_signal(data,eeg_artifacts, None, 106,1000,None)
    # artifact18 = get_signal(data,eeg_artifacts, None, 106,1000,None)


    regular_eeg1 = get_signal(data,eeg_artifacts, None, 17,1000,0)
    regular_eeg2 = get_signal(data,eeg_artifacts, None, 17,1000,1)
    regular_eeg3 = get_signal(data,eeg_artifacts, None, 17,1000,2)
    #regular_eeg4 = get_signal(data,eeg_artifacts, None, 17,1000,eegdat.values()[3])
    # regular_eeg5 = get_signal(data,eeg_artifacts, None, 17,1000,None)
    # regular_eeg6 = get_signal(data,eeg_artifacts, None, 17,1000,None)
    # regular_eeg7 = get_signal(data,eeg_artifacts, None, 17,1000,None)
    # regular_eeg8 = get_signal(data,eeg_artifacts, None, 17,1000,None)
    # regular_eeg9 = get_signal(data,eeg_artifacts, None, 17,1000,None)
    # regular_eeg10 = get_signal(data,eeg_artifacts, None, 17,1000,None)
    # regular_eeg11 = get_signal(data,eeg_artifacts, None, 17,1000,None)
    # regular_eeg12 = get_signal(data,eeg_artifacts, None, 17,1000,None)
    # regular_eeg13 = get_signal(data,eeg_artifacts, None, 17,1000,None)
    # regular_eeg14 = get_signal(data,eeg_artifacts, None, 17,1000,None)
    # regular_eeg15 = get_signal(data,eeg_artifacts, None, 17,1000,None)
    # regular_eeg16 = get_signal(data,eeg_artifacts, None, 17,1000,None)
    # regular_eeg17 = get_signal(data,eeg_artifacts, None, 17,1000,None)
    # regular_eeg18 = get_signal(data,eeg_artifacts, None, 17,1000,None)

    ICA_signal = np.zeros((64,6000))
    ICA_signal[:,0:1000] = regular_eeg1
    ICA_signal[:,1000:2000] = regular_eeg2
    ICA_signal[:,2000:3000] = regular_eeg3
    ICA_signal[:,3000:4000] = artifact1
    ICA_signal[:,4000:5000] = artifact2
    ICA_signal[:,5000:6000] = artifact3
    # ICA_signal[:,6000:7000] = regular_eeg4
    # ICA_signal[:,7000:8000] = regular_eeg5
    # ICA_signal[:,8000:9000] = regular_eeg6
    # ICA_signal[:,9000:10000] = artifact4
    # ICA_signal[:,10000:11000] = artifact5
    # ICA_signal[:,11000:12000] = artifact6
    # ICA_signal[:,12000:13000] = regular_eeg7
    # ICA_signal[:,13000:14000] = regular_eeg8
    # ICA_signal[:,14000:15000] = regular_eeg9
    # ICA_signal[:,15000:16000] = artifact7
    # ICA_signal[:,16000:17000] = artifact8
    # ICA_signal[:,17000:18000] = artifact9
    # ICA_signal[:,18000:19000] = regular_eeg10
    # ICA_signal[:,19000:20000] = regular_eeg11
    # ICA_signal[:,20000:21000] = regular_eeg12
    # ICA_signal[:,21000:22000] = artifact10
    # ICA_signal[:,22000:23000] = artifact11
    # ICA_signal[:,23000:24000] = artifact12
    # ICA_signal[:,12000:13000] = regular_eeg13
    # ICA_signal[:,13000:14000] = regular_eeg14
    # ICA_signal[:,14000:15000] = regular_eeg15
    # ICA_signal[:,15000:16000] = artifact13
    # ICA_signal[:,16000:17000] = artifact14
    # ICA_signal[:,17000:18000] = artifact15
    # ICA_signal[:,18000:19000] = regular_eeg16
    # ICA_signal[:,19000:20000] = regular_eeg17
    # ICA_signal[:,20000:21000] = regular_eeg18
    # ICA_signal[:,21000:22000] = artifact16
    # ICA_signal[:,22000:23000] = artifact17
    # ICA_signal[:,23000:24000] = artifact18 
    # ICA
    ica = FastICA(n_components=64,max_iter=200)
    # Create Data for ICA
    components = ica.fit_transform(ICA_signal[0:64,:].T)
    bag = components.T
    #print bag.shape
    # Slice Small section of EEG data
    # 1000 Samples only
    bag = bag[:,3000:4000]
   


    # Featurize all components
    # Parameters
    word_size = 5 # Entire time series in these many characters
    alphabet_size = 4 # Total number of characters
    num_subsequences = 10 #windowSize = int(len(x)/numSubsequences)
    overlapping_fraction = 0.8 #overlap = window_size*overlapping_fraction
    s = SAX(word_size, alphabet_size)
    max_features = 200

    num_components = bag.shape[0]
    feature = np.ndarray((num_components,max_features))
    for i in range(len(bag)):
        component = bag[i]
        (sax_str, sax_ind) = s.sliding_window(component, num_subsequences, overlapping_fraction)
        for j in range(max_features):
            feature[i][j] = sax_str.count(vocab_features[j][0])

    # Classify the bag
   
    # for algorithm, classifier in classifiers.items():
    #     #classifier.fit(train_bags, train_labels)
    #     predictions = classifier.predict([feature])
    #     accuracies[algorithm] = np.average(-1 == np.sign(predictions))
    #     classifier._bags = None
    #     classifier._y = None
    #     classifier._alphas = None
    #     classifier._bag_predictions = None
    #     classifier._objective = None
    # print predictions.shape

    predictions = {}
    # Classify the bag
    for algorithm, classifier in classifiers.items():
        prediction = classifier.predict([feature])
        predictions[i] = prediction

    
    if np.sign(predictions[i]) == 1:
        count = count +1

   
#for algorithm, accuracy in accuracies.items():
#    print '\n%s Accuracy: %.1f%%' % (algorithm, 100 * accuracy)


    print np.sign(predictions[i])
    print count


