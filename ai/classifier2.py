from nltk import word_tokenize, WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.svm import LinearSVC
from sklearn.cross_validation import KFold
from sklearn import metrics
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction import DictVectorizer

import data_parser
import stopwords
import random
import numpy

from plotter import plot


class SubjectivityClassier(object):

    def __init__(self, data_file_pos='data_1.pos', data_file_neg='data_2.neg'):
        print 'initializing'
        # initialize helper classes
        self._vectorizer = DictVectorizer(dtype=float, sparse=True)
        self._lematizer = WordNetLemmatizer()
        self.initialize_data_sets(data_file_pos, data_file_neg)

    def initialize_data_sets(self, data_file_pos='data_1.pos',
                             data_file_neg='data_2.neg'):
        training_set, test_set = self.__parse(
            data_file_pos, data_file_neg)
        print "Transformin feature vectors"
        X_training = []
        Y_training = []
        for x in training_set:
            X_training.append(x[0])
            Y_training.append(x[1])
        X_test = []
        Y_test = []
        for x in test_set:
            X_test.append(x[0])
            Y_test.append(x[1])
        self.X_training = self._vectorizer.fit_transform(X_training).toarray()
        self.Y_training = Y_training
        self.X_test = self._vectorizer.transform(X_test).toarray()
        self.Y_test = Y_test

    def train_classifier(self, n_folds=10, learning_curve=True,
                         start_size=4000, inc=1000):
        print "training"
        size = len(self.X_training)
        train_accs = []
        cv_accs = []
        sizes = []
        if learning_curve:
            size = start_size
        while size <= len(self.X_training):
            print size
            sizes.append(size)
            X = self.X_training[:size]
            Y = self.Y_training[:size]
            classifier, train_acc, cv_acc = self.cross_validation(X, Y)
            train_accs.append(train_acc)
            cv_accs.append(cv_acc)
            size += inc
        print "here"
        self.classifier = classifier
        if learning_curve:
            plot(sizes, ys=[train_accs, cv_accs], legs=['Training', 'CV'])

    def test(self):
        print "testing"
        print "Accouracy on test set: {0}".format(self.__accuracy(
            self.classifier, self.X_test, self.Y_test))
        y_pred = self.classifier.predict(self.X_test)
        print metrics.classification_report(self.Y_test, y_pred)

    def predict(self, sentence):
        word_dict = self.__word_dict(sentence)
        features = self._vectorizer.transform(word_dict).toarray()
        return self.classifier.predict(features)

    def cross_validation(self, X, Y, n_folds=8):
        kf = KFold(len(X), n_folds=n_folds)
        best_accuracy = -1
        training_accuracy = 0
        for train, cv in kf:
            classifier = LinearSVC(C=1, tol=0.000001)
            X_train = []
            X_train.extend(X[0:cv[0]])
            X_train.extend(X[cv[-1]:])
            Y_train = []
            Y_train.extend(Y[0:cv[0]])
            Y_train.extend(Y[cv[-1]:])
            # c = 0
            # for index in train:
            #     X_train.append(X[index])
            #     Y_train.append(Y[index])
            # X_train = X[cv[-1]:]
            # Y_train = Y[cv[-1]:]
            # print len(X_train[0])
            # if cv[0] != 0:
            #     print len(X[0:cv[0]][0])
            #     X_train += X[0:cv[0]]
            #     Y_train += Y[0:cv[0]]
            X_cv = X[cv[0]:cv[-1]+1]
            Y_cv = Y[cv[0]:cv[-1]+1]
            print "Fitting..."
            classifier.fit(X_train, Y_train)
            print "Accouracy..."
            accuracy = self.__accuracy(classifier, X_cv, Y_cv)
            if accuracy > best_accuracy:
                best_classifier = classifier
                best_accuracy = accuracy
                training_accuracy = self.__accuracy(
                    classifier, X_train, Y_train)
        return best_classifier, training_accuracy, best_accuracy

    def __accuracy(self, classifier, X, Y):
        y_pred = classifier.predict(X)
        count = 0
        i = 0
        size = len(Y)
        while i < size:
            if y_pred[i] == Y[i]:
                count += 1
            i += 1
        return float(count) / size

    def __parse(self, data_file_pos='data_1.pos', data_file_neg='data_2.neg'):
        print "Reading files..."
        data_pos, data_neg = data_parser.build_data(positive=data_file_pos,
                                                    negative=data_file_neg)
        data_test = data_pos[int(len(data_pos) * .8):] + \
            data_neg[int(len(data_neg) * .8):]
        random.shuffle(data_test)
        data_train = data_pos[:int(len(data_pos) * .8)] + \
            data_neg[:int(len(data_neg) * .8)]
        random.shuffle(data_train)
        print "Size: ", len(data_train) + len(data_test)
        training_set = [(self.__word_dict(d['text']), d['class'])
                        for d in data_train]
        test_set = [(self.__word_dict(d['text']), d['class'])
                    for d in data_test]
        return training_set, test_set

    def __word_dict(self, sentence):
        features = {}
        words = {}
        lematizer = WordNetLemmatizer()

        # get individual words from text
        words = [lematizer.lemmatize(word.lower()) for word in \
                 word_tokenize(sentence)]

        for word in words:
            word = word.encode('utf-8', 'ignore')
            if len(word) > 1:
                # check if word in not a stop word
                if word not in stopwords.stop_words:
                    if word in features:
                        features[word] += 1
                    else:
                        features[word] = 1
        return features

    def train(self, n_folds=10):
        classifier = LinearSVC()


a = SubjectivityClassier()
a.train_classifier(learning_curve=False)
a.test()

sentence = 'a'
while True:
    sentence = raw_input('Test me: ')
    if sentence == '':
        break
    classification = a.predict(sentence)
    if classification == 1:
        print 'Subjective'
    else:
        print 'Objective'
