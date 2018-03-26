import os
import re
import json
from collections import defaultdict
from functools import reduce


class PreTreat(object):
    def __init__(self, data_dir='.', TRAINSET_SIZE=171, dump=False):
        self.spam_wdic_dir = 'spam_wdic.json'
        self.ham_wdic_dir = 'ham_wdic.json'

        self.laplace = 1e-10
        self.data_dir = data_dir
        self.FOLDER_SIZE = 300
        self.TOTAL_FOLDER = 216
        self.TRAINSET_SIZE = TRAINSET_SIZE
        self.dump = dump

        self.train_labels = []
        self.spam_wdic = defaultdict(int)
        self.ham_wdic = defaultdict(int)
        self.spam_len = 0
        self.spam_wn = 0
        self.ham_len = 0
        self.ham_wn = 0

    def parse_mail(self, mail_dir, spam_label, predict):
        # if predict:
        p_spam = self.tot_train_spam / (self.TRAINSET_SIZE*self.FOLDER_SIZE)
        p_ham = (1 - p_spam)

        zhPattern = re.compile(u'[\u4e00-\u9fa5]+')
        with open(mail_dir, 'r') as mo:
            mail = mo.readlines()
            i = 0
            flag = False
            for line in mail:
                i += 1
                if not flag and line != '\n':
                    continue
                flag = True
                words = line.split()
                if not predict:
                    for word in words:
                        if zhPattern.search(word):
                            if spam_label:
                                self.spam_wdic[word] += 1
                            else:
                                self.ham_wdic[word] += 1
                else:
                    for word in words:
                        if zhPattern.search(word):
                            self.spam_wdic.setdefault(word, 0)
                            self.ham_wdic.setdefault(word, 0)
                            p_spam *= (self.spam_wdic[word] + self.laplace) / (self.spam_wn + self.laplace*self.spam_len) * 1e4
                            p_ham *= (self.ham_wdic[word] + self.laplace) / (self.ham_wn + self.laplace*self.ham_len) * 1e4
                            # print(p_spam, p_ham)

        spam_predict = (p_spam >= p_ham)
        # exit()

        return spam_predict == spam_label 

    def load_data(self):
        if self.dump:
            with open(os.path.join(self.data_dir, 'trec06c-utf8', 'label', 'index'), 'r') as ld:
                self.train_labels = list(map(lambda x: x.split()[0] == 'spam', ld.readlines()[
                    :self.TRAINSET_SIZE * self.FOLDER_SIZE]))
            dcd = os.path.join(self.data_dir, 'trec06c-utf8', 'data_cut')
            for i in range(self.TRAINSET_SIZE):
                dcdd = os.path.join(dcd, '%s' % (format(i, '03')))
                for j in range(self.FOLDER_SIZE):
                    md = os.path.join(dcdd, '%s' % (format(j, '03')))
                    self.parse_mail(md, self.train_labels[i*self.FOLDER_SIZE + j], False)
                if (i % 10 == 0):
                    print(i/self.TRAINSET_SIZE*100, '%')
            with open(self.spam_wdic_dir, 'w') as swf:
                json.dump(self.spam_wdic, swf, ensure_ascii=False)
            with open(self.ham_wdic_dir, 'w') as hwf:
                json.dump(self.ham_wdic, hwf, ensure_ascii=False)
        else:
            with open(os.path.join(self.data_dir, 'trec06c-utf8', 'label', 'index'), 'r') as ld:
                self.train_labels = list(map(lambda x: x.split()[0] == 'spam', ld.readlines()))
            self.tot_train_spam = reduce(lambda a,b: a+b, self.train_labels[:self.TRAINSET_SIZE*self.FOLDER_SIZE])
            self.tot_train_ham = self.TRAINSET_SIZE*self.FOLDER_SIZE - self.tot_train_spam
            print(self.tot_train_spam, self.tot_train_ham)
            with open(self.spam_wdic_dir, 'r') as swf:
                self.spam_wdic = json.load(swf)
                self.spam_len = len(self.spam_wdic)
                self.spam_wn = reduce(lambda a, b: a+b, self.spam_wdic.values())
            with open(self.ham_wdic_dir, 'r') as hwf:
                self.ham_wdic = json.load(hwf)
                self.spam_len = len(self.ham_wdic)
                self.ham_wn = reduce(lambda a, b: a+b, self.ham_wdic.values())

    def predict(self):
        dcd = os.path.join(self.data_dir, 'trec06c-utf8', 'data_cut')
        self.acc = 0
        self.tot_test = 0
        for i in range(self.TRAINSET_SIZE, self.TOTAL_FOLDER):
        # for i in range(183, self.TOTAL_FOLDER):
            dcdd = os.path.join(dcd, '%s' % (format(i, '03')))
            if i < self.TOTAL_FOLDER - 1:
                for j in range(self.FOLDER_SIZE):
                    md = os.path.join(dcdd, '%s' % (format(j, '03')))
                    self.acc += self.parse_mail(md, self.train_labels[i*self.FOLDER_SIZE + j], True)
                self.tot_test += self.FOLDER_SIZE
            else:
                folder_size = len(os.listdir(dcdd))
                for j in range(folder_size):
                    md = os.path.join(dcdd, '%s' % (format(j, '03')))
                    self.acc += self.parse_mail(
                        md, self.train_labels[i*self.FOLDER_SIZE + j], True)
                self.tot_test += folder_size
        self.acc /= self.tot_test
        print("Test Acc:", self.acc)

'''
# generate word dict
pt = PreTreat(dump=True)
pt.load_data()
'''

pt = PreTreat(dump=False)
pt.load_data()
pt.predict()
