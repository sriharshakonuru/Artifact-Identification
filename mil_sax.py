#!/usr/bin/env python
import numpy as np
import misvm
import pickle
import sys

def main():
    bags_file = np.load("comp_artifact/bags_features.npz")
    labels_file = np.load("comp_artifact/labels.npz")
    bags = list(bags_file['arr_0'])
    labels = list(labels_file['arr_0'])

    # Spilt dataset arbitrarily to train/test sets
    train_bags = bags[250:]
    train_labels = labels[250:]
    test_bags = bags[:250]
    test_labels = labels[:250]

    # Construct classifiers
    classifiers = {}
    classifiers['NSK'] = misvm.NSK(kernel='linear', C=1.0,sv_cutoff=1e-04)
    # classifiers['MISVMlinear'] = misvm.MISVM(kernel='linear', C=1.0, max_iters=20)
    # classifiers['miSVMlinear'] = misvm.miSVM(kernel='linear', C=1.0, max_iters=20)
    # classifiers['MISVMqudratic'] = misvm.MISVM(kernel='quadratic', C=1.0, max_iters=20)
    # classifiers['miSVMqudratic'] = misvm.miSVM(kernel='quadratic', C=1.0, max_iters=20)
    #classifiers['MissSVM'] = misvm.MissSVM(kernel='linear', C=1.0, max_iters=20)
    #classifiers['sbMIL'] = misvm.sbMIL(kernel='linear', eta=0.1, C=1.0)
    #classifiers['SIL'] = misvm.SIL(kernel='linear', C=1.0)

    # Train/Evaluate classifiers
    accuracies = {}
    for algorithm, classifier in classifiers.items():
        classifier.fit(train_bags, train_labels)
        predictions = classifier.predict(test_bags)
        accuracies[algorithm] = np.average(test_labels == np.sign(predictions))
        classifier._bags = None
        classifier._y = None
        classifier._alphas = None
        classifier._bag_predictions = None
        classifier._objective = None


    pickle.dump(classifiers,open('model/model'+sys.argv[1]+'.pkl', 'wb'))

    for algorithm, accuracy in accuracies.items():
        print '\n%s Accuracy: %.1f%%' % (algorithm, 100 * accuracy)


if __name__ == '__main__':
    main()
