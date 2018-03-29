import os
import re
import json
from collections import defaultdict
from functools import reduce
from math import log
from random import sample
from numpy import std, exp
from sklearn.metrics import precision_recall_curve

import matplotlib.pyplot as plt

class SpamFilter(object):
    def __init__(self, data_dir='.', TRAINSET_PORTION=0.8, init = False, dump=False, sample_rate = 1, laplace = 1e-20, use_addr = False, use_date = False):
        '''
        Init the classifier. Set related parameters.
        '''
        self.spam_wdic_dir = 'spam_wdic.json'
        self.ham_wdic_dir = 'ham_wdic.json'
        self.spam_eadic_dir = 'spam_eadic.json'
        self.ham_eadic_dir = 'ham_eadic.json'

        self.data_dir = data_dir
        self.FOLDER_SIZE = 300
        # self.TOTAL_FOLDER = 216
        self.TOTAL_SIZE = 64620
        self.TRAINSET_SIZE = int(self.TOTAL_SIZE * TRAINSET_PORTION)
        self.use_addr = use_addr
        self.use_date = use_date
        self.init = init
        self.dump = dump

        # sampling the train set. Laplace is for laplace smoothing and self.laplace is the smoothing factor
        self.sample_rate = sample_rate
        self.laplace = laplace

        self.labels = []
        self.sample_list = []

        #dict for spam and ham. key: word and value: word count
        self.spam_wdic = defaultdict(int)
        self.ham_wdic = defaultdict(int)

        # number of total words for spam and ham
        self.spam_wn = 0
        self.ham_wn = 0

        # number of total different words for spam and ham
        self.spam_len = 0
        self.ham_len = 0

        # dict for email address
        self.spam_eadic = defaultdict(int)
        self.ham_eadic = defaultdict(int)

        #dict for date
        self.spam_ddic = defaultdict(int)
        self.ham_ddic = defaultdict(int)

        self.spam_ean = 0
        self.ham_ean = 0

        self.tot_train_spam = 0
        self.tot_train_ham = 0

        # for calc precision, recall and F1
        self.p = self.r = 0
        self.tot_predict_true = 0
        self.TP= 0

        # for plot PR curve, show the possibility of spam
        self.score = []

    def parse_mail(self, mail_dir, spam_label, predict):
        # P(y = spam) & P(y = ham)
        p_spam = 0
        p_ham = 0

        zhPattern = re.compile(u'[\u4e00-\u9fa5]+')
        with open(mail_dir, 'r') as mo:
            mail = mo.readlines()
            i = 0
            flag = False
            for line in mail:
                i += 1
                if not flag and line != '\n': # find the body of the mail
                    continue
                flag = True
                words = line.split()
                if not predict: # set up the dict for spam and ham
                    for word in words:
                        if zhPattern.search(word):
                            if spam_label:
                                self.spam_wdic[word] += 1
                            else:
                                self.ham_wdic[word] += 1
                else:   # naive bayes prediction
                    for word in words:
                        if zhPattern.search(word):
                            self.spam_wdic.setdefault(word, 0)
                            self.ham_wdic.setdefault(word, 0)
                            if self.spam_wdic[word] + self.ham_wdic[word] != 0:
                                p_spam += log((self.spam_wdic[word] + self.laplace) / (self.spam_wn + self.laplace*self.spam_len))
                                p_ham += log((self.ham_wdic[word] + self.laplace) / (self.ham_wn + self.laplace*self.ham_len))
        if predict:
            p_spam += log(self.tot_train_spam / (self.TRAINSET_SIZE * self.sample_rate))
            p_ham += log(1 - self.tot_train_spam / (self.TRAINSET_SIZE * self.sample_rate))
            spam_predict = (p_spam >= p_ham)
            self.tot_predict_true += spam_predict
            self.TP += (spam_predict and spam_label)

            # p_spam = (1 / (1 + exp(p_ham-p_spam)))
            return (p_ham/(p_ham+p_spam))


    def parse_head(self, mail_dir, spam_label, predict):
        # P(y = spam) & P(y = ham)
        p_spam = 0
        p_ham = 0

        mailPattern = re.compile(r"[-_\w\.]{0,64}@([-\w]{1,63}\.)*[-\w]{1,63}")
        with open(mail_dir, 'r') as mo:
            mail = mo.readlines()
            i = 0

            for line in mail:
                i += 1
                if len(line) > 0 and line[:4] == 'From':
                    if not mailPattern.search(line):
                        break
                    mailaddr = mailPattern.search(line).group(0)
                    words = re.split(r'[@\.]',mailaddr)
                    if not predict:
                        for word in words:
                            if spam_label:
                                self.spam_eadic[word] += 1
                            else:
                                self.ham_eadic[word] += 1
                    else:
                        for word in words:
                            self.spam_eadic.setdefault(word, 0)
                            self.ham_eadic.setdefault(word, 0)
                            if self.spam_eadic[word] * self.ham_eadic[word] != 0:
                                p_spam += log((self.spam_eadic[word])/(self.spam_ean))
                                p_ham += log((self.ham_eadic[word])/(self.ham_ean))
                    break
                else:
                    continue
        if predict:
            p_spam += log(self.tot_train_spam /
                          (self.TRAINSET_SIZE * self.sample_rate))
            p_ham += log(1 - self.tot_train_spam /
                         (self.TRAINSET_SIZE * self.sample_rate))

            return (p_ham/(p_ham+p_spam))

    def parse_date(self, mail_dir, spam_label, predict):
        # P(y = spam) & P(y = ham)
        p_spam = 0
        p_ham = 0

        with open(mail_dir, 'r') as mo:
            mail = mo.readlines()
            i = 0

            for line in mail:
                i += 1
                if len(line) > 0 and line[:4] == 'Date':
                    line = line[6:]
                    words = re.split(r'[@:,\s\.]', line)
                    if not predict:
                        for word in words:
                            if spam_label:
                                self.spam_ddic[word] += 1
                            else:
                                self.ham_ddic[word] += 1
                    else:
                        for word in words:
                            self.spam_ddic.setdefault(word, 0)
                            self.ham_ddic.setdefault(word, 0)
                            if self.spam_ddic[word] * self.ham_ddic[word] != 0:
                                p_spam += log((self.spam_ddic[word])/(self.spam_dn))
                                p_ham += log((self.ham_ddic[word])/(self.ham_dn))
                    break
        if predict:
            p_spam += log(self.tot_train_spam /
                          (self.TRAINSET_SIZE * self.sample_rate))
            p_ham += log(1 - self.tot_train_spam /
                         (self.TRAINSET_SIZE * self.sample_rate))

            spam_predict = (p_spam >= p_ham)

            self.tot_predict_true += spam_predict
            self.TP += (spam_predict and spam_label)
            return (p_ham/(p_ham+p_spam))


    def load_data(self):
        if self.init:
            self.sample_list = sample([i for i in range(self.TRAINSET_SIZE)], int(self.TRAINSET_SIZE*self.sample_rate))

            with open(os.path.join(self.data_dir, 'trec06c-utf8', 'label', 'index'), 'r') as ld:
                self.labels = list(map(lambda x: x.split()[0] == 'spam', ld.readlines()))
            self.labels = [self.labels[x] for x in self.sample_list]

            dcd = os.path.join(self.data_dir, 'trec06c-utf8', 'data_cut')
            
            cnt = 0
            for i in self.sample_list:
                folder = i // self.FOLDER_SIZE
                index = i % self.FOLDER_SIZE
                md = os.path.join(dcd, '%s' % (format(folder, '03')), '%s' % (format(index, '03')))
                self.parse_mail(md, self.labels[cnt], False)
                self.parse_head(md, self.labels[cnt], False)
                self.parse_date(md, self.labels[cnt], False)

                cnt += 1
                if cnt % 3000 == 0:
                    print('%.2f'%(cnt/(self.TRAINSET_SIZE*self.sample_rate)*100), '%')

            if self.dump:
                with open(self.spam_wdic_dir, 'w') as swf:
                    json.dump(self.spam_wdic, swf, ensure_ascii=False)
                with open(self.ham_wdic_dir, 'w') as hwf:
                    json.dump(self.ham_wdic, hwf, ensure_ascii=False)
                with open(self.spam_eadic_dir, 'w') as sf:
                    json.dump(self.spam_eadic, sf, ensure_ascii=False)
                with open(self.ham_eadic_dir, 'w') as hf:
                    json.dump(self.ham_eadic, hf, ensure_ascii=False)
        else:
            with open(os.path.join(self.data_dir, 'trec06c-utf8', 'label', 'index'), 'r') as ld:
                self.labels = list(map(lambda x: x.split()[0] == 'spam', ld.readlines()[:self.TRAINSET_SIZE]))
            with open(self.spam_wdic_dir, 'r') as swf:
                self.spam_wdic = json.load(swf)
            with open(self.ham_wdic_dir, 'r') as hwf:
                self.ham_wdic = json.load(hwf)
            with open(self.spam_eadic_dir, 'r') as sf:
                self.spam_eadic = json.load(sf)
            with open(self.ham_eadic_dir, 'r') as hf:
                self.ham_eadic = json.load(hf) 

        self.tot_train_spam = reduce(lambda a,b: a+b, self.labels)
        self.tot_train_ham = int(self.TRAINSET_SIZE*self.sample_rate) - self.tot_train_spam
        print(self.tot_train_spam, self.tot_train_ham)

        self.spam_len = len(self.spam_wdic)
        self.spam_wn = reduce(lambda a, b: a+b, self.spam_wdic.values())
        self.spam_len = len(self.ham_wdic)
        self.ham_wn = reduce(lambda a, b: a+b, self.ham_wdic.values())

        self.spam_ean = reduce(lambda a, b: a+b, self.spam_eadic.values())
        self.ham_ean = reduce(lambda a, b: a+b, self.ham_eadic.values())

        self.spam_dn = reduce(lambda a, b: a+b, self.spam_ddic.values())
        self.ham_dn = reduce(lambda a, b: a+b, self.ham_ddic.values())


    def plot(self):
        precision, recall, threshold = precision_recall_curve(self.labels[self.TRAINSET_SIZE:self.TOTAL_SIZE], self.score)

        plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
        plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
        plt.ylim(0.95, 1)
        plt.plot(recall, precision)
        plt.show()
        return precision, recall


    def predict(self):
        with open(os.path.join(self.data_dir, 'trec06c-utf8', 'label', 'index'), 'r') as ld:
            self.labels = list(map(lambda x: x.split()[0] == 'spam', ld.readlines()))
        dcd = os.path.join(self.data_dir, 'trec06c-utf8', 'data_cut')

        self.acc = 0

        for i in range(self.TRAINSET_SIZE, self.TOTAL_SIZE):
            folder = i // self.FOLDER_SIZE
            index = i % self.FOLDER_SIZE
            md=os.path.join(dcd, '%s' % (format(folder, '03')), '%s' % (format(index, '03')))
            if self.use_addr:
                prob = (self.parse_mail(md, self.labels[i], True) + self.parse_head(md, self.labels[i], True)) / 2
                self.acc += (( prob>= 0.5) == self.labels[i])
                self.score.append(prob)
            else:
                prob = 0.5
                if self.use_date:
                    prob = (self.parse_date(md, self.labels[i], True))
                else:
                    prob = (self.parse_mail(md, self.labels[i], True))
                self.acc += ((prob >= 0.5) == self.labels[i])
                self.score.append(prob)

        self.tot_test = self.TOTAL_SIZE - self.TRAINSET_SIZE
        self.p = self.TP / self.tot_predict_true
        total_true = reduce(lambda a,b: a+b, self.labels[self.TRAINSET_SIZE: self.TOTAL_SIZE])
        self.r = self.TP / total_true
        self.F1 = (2*self.p*self.r)/(self.p+self.r)

        self.acc /= self.tot_test
        print("Test Acc:", self.acc, "P:", self.p, "R:", self.r, "F1:", self.F1)

        return self.plot()

'''
# generate word dict
pt = SpamFilter(init=True)
pt.load_data()
'''


def test():
    # train and predict
    pt = SpamFilter(init = True, dump=False, sample_rate=1)
    pt.load_data()
    pt.predict()



# plot pr curve with different sample rate
def plot_diff_port():
    portion = [0.01, 0.05, 0.25, 0.5, 1]
    pres = []
    recs = []
    for i in range(5):
        pt5 = SpamFilter(init = True, dump=False, sample_rate=portion[i])
        print('iter', i)
        pt5.load_data()
        pre, rec = pt5.predict()
        pres.append(pre)
        recs.append(rec)

    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    plt.ylim(0.95, 1)
    plt.xlim(0.2, 1)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    for i in range(5):
        plt.plot(recs[i], pres[i])
    plt.legend(['0.01', '0.05', '0.25', '0.5', '1.0'], loc=0)
    plt.show()


# stat top 30 words in spam and ham
def top30():
    spam_wdic_dir = 'spam_wdic.json'
    ham_wdic_dir = 'ham_wdic.json'
    with open(spam_wdic_dir, 'r') as swf:
        spam_wdic = json.load(swf)
    with open(ham_wdic_dir, 'r') as hwf:
        ham_wdic = json.load(hwf)
    swd = list(spam_wdic.items())
    swd = sorted(swd, key = lambda x: x[1], reverse=True)
    print(swd[:30])
    print('\n')
    hwd = list(ham_wdic.items())
    hwd = sorted(hwd, key=lambda x: x[1], reverse=True)
    print(hwd[:30])

test()