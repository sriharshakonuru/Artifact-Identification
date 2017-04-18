#!/usr/bin/env python
import numpy as np
import misvm
import matplotlib.pyplot as plt

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
    classifiers['miSVM'] = misvm.MISVM(kernel='linear', C=1.0, max_iters=20)

    # Train/Evaluate classifiers
    accuracies = {}
    for algorithm, classifier in classifiers.items():
        classifier.fit(train_bags, train_labels)
        predictions,instance_pred = classifier.predict(test_bags,instancePrediction= True)
        accuracies[algorithm] = np.average(test_labels == np.sign(predictions))

    for algorithm, accuracy in accuracies.items():
        print '\n%s Accuracy: %.1f%%' % (algorithm, 100 * accuracy)

    bags_original_file = np.load("comp_artifact/bags.npz")
    bags_original = list(bags_original_file['arr_0'])
    test_bags_original = bags_original[:10]

    count = 0
    for bag in test_bags_original:
        for j in range(bag.shape[0]):
            if instance_pred[count] < 0:
                plt.figure()
                plt.plot(bag[j,:])
                plt.savefig("comp_artifact_results/negative_labels/"+str(count)+".jpg")
            else:
                plt.figure()
                plt.plot(bag[j,:])
                plt.savefig("comp_artifact_results/positive_labels/"+str(count)+".jpg")
            count +=1

if __name__ == '__main__':
    main()
