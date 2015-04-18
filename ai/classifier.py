from nltk import word_tokenize, WordNetLemmatizer, classify
from nltk.classify.scikitlearn import SklearnClassifier

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import KFold

from sklearn import metrics
from sklearn.svm import SVC
from sklearn.cross_validation import train_test_split

import numpy
import data_parser
import stopwords
import re
import random

from plotter import plot


def feature_extractor(d):
    features = {}
    words = {}
    lematizer = WordNetLemmatizer()

    # get individual words from text
    words = [lematizer.lemmatize(word.lower()) for word in word_tokenize(d)]

    for word in words:
        word = word.encode('utf-8', 'ignore')
        if len(word) > 1:
            # check if word in not a stop word
            if word not in stopwords.stop_words:
                # check if the word is not a url or @person
                if not re.match('http://.*|@.*', word):
                    if word in features:
                        features[word] += 1
                    else:
                        features[word] = 1
    return features


def parse_data():
    data_pos, data_neg = data_parser.build_data()
    random.shuffle(data_pos)
    random.shuffle(data_neg)
    data_test = data_pos[int(len(data_pos) * .8):] + data_neg[int(len(data_neg) * .8):]
    data_train = data_pos[:int(len(data_pos) * .8)] + data_neg[:int(len(data_neg) * .8)]
    print "Size: ", len(data_train) + len(data_test)
    training_set = [(feature_extractor(d['text']), d['class'])
                    for d in data_train]
    random.shuffle(training_set)
    test_set = [(feature_extractor(d['text']), d['class']) for d in data_test]
    random.shuffle(test_set)
    return training_set, test_set, training_set + test_set


def cross_validation(data_set, n_folds=8):
    kf = KFold(len(data_set), n_folds=n_folds)
    best_accuracy = -1
    training_accuracy = 0
    for train, cv in kf:
        print cv
        classifier = SklearnClassifier(
            Pipeline([('tfidf', TfidfTransformer()),
                      ('nb', LinearSVC(C=1, tol=0.000001))]))
        training_data = data_set[0:cv[0]] + data_set[cv[-1]:]
        cv_data = data_set[cv[0]:cv[-1]+1]
        classifier.train(training_data)
        accuracy = classify.accuracy(classifier, cv_data)
        if accuracy > best_accuracy:
            best_classifier = classifier
            best_accuracy = accuracy
            training_accuracy = classify.accuracy(classifier, training_data)
    return best_classifier, training_accuracy, best_accuracy


def diagnose(classifier, data_set):
    feats = []
    classes = []
    for data in data_set:
        feats.append(data[0])
        classes.append(data[1])
    pred = classifier.classify_many(feats)
    print(metrics.classification_report(classes, pred))


training_set, test_set, total = parse_data()
print 'Training set size: ' + str(len(training_set))
print 'Test set size: ' + str(len(test_set))

i = 1000
sizes = []
training_acc = []
cv_acc = []
while i < len(training_set):
    t = training_set[:i]
    c, training, cv = cross_validation(t)
    sizes.append(i)
    training_acc.append(training)
    cv_acc.append(cv)
    i += 1000
    # print i
plot(sizes, ys=[training_acc, cv_acc], legs=['Training', 'CV'])
# Get accuracy
lr_classifier = cross_validation(training_set)[0]
lr_accuracy = classify.accuracy(lr_classifier, test_set)
# print "Classifier accuracy on test: " + str(lr_accuracy)
lr_accuracy_training = classify.accuracy(lr_classifier, training_set)
# print "Classifier accuracy on training: " + str(lr_accuracy_training)

diagnose(lr_classifier, test_set)

sentence = 'a'
while True:
    sentence = raw_input('Test me: ')
    if sentence == '':
        break
    features = feature_extractor(sentence)
    classification = lr_classifier.classify(features)
    if classification == 1:
        print 'Subjective'
    else:
        print 'Objective'
