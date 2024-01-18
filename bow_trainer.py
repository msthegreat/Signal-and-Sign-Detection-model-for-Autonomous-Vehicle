# Reference: Learning_OpenCV_3_Computer_Vision_with_Python_Second_Edition/Chapter-27

import os

import cv2
import numpy as np
from sklearn.svm import LinearSVC
import pickle


def bow_features(image, sift, extract_bow):
    features = sift.detect(image)
    return extract_bow.compute(image, features)

def init_trainer():
    sift = cv2.SIFT_create()
    params = dict(algorithm=1, trees=5)
    matcher = cv2.FlannBasedMatcher(params, {})
    bow_kmeans_trainer = cv2.BOWKMeansTrainer(128)
    extract_bow = cv2.BOWImgDescriptorExtractor(sift, matcher)
    return sift, bow_kmeans_trainer, extract_bow


def erd_trainer(path, classes, classifer_file_name):
    # path = "./Resources/Signs"
    # classes = ["pedestrianCrossing", "signalAhead", "stop"]
    sift, bow_kmeans_trainer, extract_bow = init_trainer()
    for index, sign in enumerate(classes):
        full_path = path + "/" + sign
        for file in os.listdir(full_path):
            image = cv2.imread(full_path + "/" + file)
            _, descriptors = sift.detectAndCompute(image, None)
            if descriptors is not None:
                bow_kmeans_trainer.add(descriptors)

    sign_vocab = bow_kmeans_trainer.cluster()
    np.save(classifer_file_name+"_vocab.npy", sign_vocab)
    sign_vocab = np.load(classifer_file_name+"_vocab.npy")
    extract_bow.setVocabulary(sign_vocab)

    X, labels = [], []

    for index, sign in enumerate(classes):
        full_path = path + "/" + sign
        for file in os.listdir(full_path):
            image = cv2.imread(full_path + "/" + file)
            result = bow_features(image, sift, extract_bow)
            if result is not None:
                X.extend(result)
                labels.append(index + 1)

    X = np.array(X)
    labels = np.array(labels)

    svc = LinearSVC(random_state=0, tol=1e-05)
    svc.fit(X, labels)

    with open(classifer_file_name+".pkl", 'wb') as model:
        pickle.dump(svc, model)


# This is classifier-only-test method. Not used in main model.
def test_sign_predict(path, classes, classifier_file_name):
    sift, _, extract_bow = init_trainer()
    sign_vocab = np.load(classifier_file_name + "_vocab.npy")
    extract_bow.setVocabulary(sign_vocab)

    with open(classifier_file_name+'.pkl', 'rb') as model:
        svc = pickle.load(model)

    # Predict
    score = [0, 0, 0, 0]
    total_count = [0, 0, 0, 0]
    unknowns = 0
    for index, sign in enumerate(classes):
        full_path = path + "/" + sign
        for file in os.listdir(full_path):
            image = cv2.imread(full_path + "/" + file)
            features = bow_features(image, sift, extract_bow)
            if features is None:
                result = 0
                unknowns +=1
            else:
                result = int(svc.predict(features))
            if result == index + 1:
                score[result] += 1
            total_count[index+1] += 1
    score = np.array(score[1:], dtype=np.float)
    total_count = np.array(total_count[1:], dtype=np.float)
    print(score, total_count, unknowns, np.sum(score)+unknowns, np.sum(total_count))
    score /= total_count
    return score

# ! DO NOT INVOKE THIS ! This is just for classifier training and testing prediction
def main():
    '''
    Training images are uploaded to box here : https://gatech.box.com/s/v2260x3i35owmzht1j5cia1pigdge912
    If training needs to be verified, please download and extract the CLassifier_Training_Images.zip
    to the paths mentioned below in code before execution
    '''

    path = "./Resources/Signs"
    classes = ["pedestrianCrossing", "signalAhead", "stop"]
    # path = "./Resources/Lights"
    # classes = ["go", "stop"]

    # Training
    erd_trainer(path, classes, "sign_classifier")

    # Prediction
    prediction_score = test_sign_predict(path, classes, "sign_classifier")
    print(prediction_score)

main()