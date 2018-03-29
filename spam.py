import os
import re
import json
from collections import defaultdict
from functools import reduce
from math import log
from random import sample
from numpy import std

class SpamFilter(object):
    def __init__(self, data_dir='.', TRAINSET_PORTION=0.8, init = False, dump=False, sample_rate = 1):
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
        self.init = init
        self.dump = dump

        # sampling the train set. Laplace is for laplace smoothing and self.laplace is the smoothing factor
        self.sample_rate = sample_rate
        self.laplace = 1e-10

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

        self.spam_eadic = defaultdict(int)
        self.ham_eadic = defaultdict(int)

        self.spam_ean = 0
        self.ham_ean = 0

        self.tot_train_spam = 0
        self.tot_train_ham = 0

        self.p = self.r = 0
        self.tot_predict_true = 0
        self.TP= 0

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
            return spam_predict == spam_label 


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
            spam_predict = (p_spam >= p_ham)
            self.tot_predict_true += spam_predict
            self.TP += (spam_predict and spam_label)
            return spam_predict == spam_label


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

                cnt += 1
                if cnt % 3000 == 0:
                    print('%.2f'%(cnt/(self.TRAINSET_SIZE*self.sample_rate)*100), '%')

            if self.dump:
                # with open(self.spam_wdic_dir, 'w') as swf:
                #     json.dump(self.spam_wdic, swf, ensure_ascii=False)
                # with open(self.ham_wdic_dir, 'w') as hwf:
                #     json.dump(self.ham_wdic, hwf, ensure_ascii=False)
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

        self.tot_train_spam = reduce(lambda a,b: a+b, self.labels)
        self.tot_train_ham = int(self.TRAINSET_SIZE*self.sample_rate) - self.tot_train_spam
        print(self.tot_train_spam, self.tot_train_ham)

        self.spam_len = len(self.spam_wdic)
        self.spam_wn = reduce(lambda a, b: a+b, self.spam_wdic.values())
        self.spam_len = len(self.ham_wdic)
        self.ham_wn = reduce(lambda a, b: a+b, self.ham_wdic.values())

        self.spam_ean = reduce(lambda a, b: a+b, self.spam_eadic.values())
        self.ham_ean = reduce(lambda a, b: a+b, self.ham_eadic.values())


    def predict(self):
        with open(os.path.join(self.data_dir, 'trec06c-utf8', 'label', 'index'), 'r') as ld:
            self.labels = list(map(lambda x: x.split()[0] == 'spam', ld.readlines()))
        dcd = os.path.join(self.data_dir, 'trec06c-utf8', 'data_cut')

        self.acc = 0

        for i in range(self.TRAINSET_SIZE, self.TOTAL_SIZE):
            folder = i // self.FOLDER_SIZE
            index = i % self.FOLDER_SIZE
            md=os.path.join(dcd, '%s' % (format(folder, '03')), '%s' % (format(index, '03')))
            # self.acc += self.parse_mail(md, self.labels[i], True)
            self.acc += self.parse_head(md, self.labels[i], True)

        self.tot_test = self.TOTAL_SIZE - self.TRAINSET_SIZE
        self.p = self.TP / self.tot_predict_true
        total_true = reduce(lambda a,b: a+b, self.labels[self.TRAINSET_SIZE: self.TOTAL_SIZE])
        self.r = self.TP / total_true
        self.F1 = (2*self.p*self.r)/(self.p+self.r)
        self.acc /= self.tot_test
        print("Test Acc:", self.acc, "P:", self.p, "R:", self.r, "F1:", self.F1)
        return self.acc

'''
# generate word dict
pt = SpamFilter(init=True)
pt.load_data()
'''

'''
pt = SpamFilter(init = True, dump=False, sample_rate=0.1)
pt.load_data()
pt.predict()
'''

res = []
for i in range(1):
    pt5 = SpamFilter(init = True, dump=True, sample_rate=1)
    print('iter', i)
    pt5.load_data()
    res.append(pt5.predict())
print(res)
print('Max:', max(res), 'Min:', min(res), 'Avg:', sum(res)/len(res), 'Std:', std(res))
