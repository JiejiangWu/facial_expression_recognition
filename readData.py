# -*- coding: utf-8 -*-
"""
Created on Wed Mar  7 13:41:55 2018

@author: wyj
"""

#!/usr/bin/env python

import csv
import pickle  as pk
from keras.utils import to_categorical

#facialData = open('fer2013.csv','rt')
#csvr = csv.reader(facialData)
#header = next(csvr)
#rows = [row for row in csvr]
#
#trn = [row[:-1] for row in rows if row[-1] == 'Training']
##csv.writer(open('train.csv', 'w+')).writerows([header[:-1]] + trn)
#print(len(trn))
#
#tst = [row[:-1] for row in rows if row[-1] == 'PublicTest']
##csv.writer(open('test.csv', 'w+')).writerows([header[:-1]] + tst)
#print(len(tst))
#
#tst2 = [row[:-1] for row in rows if row[-1] == 'PrivateTest']
##csv.writer(open('testprivate.csv', 'w+')).writerows([header[:-1]] + tst2)
#print(len(tst2))

trnLabel = [int(row[0]) for row in trn]
trnLabel = to_categorical(trnLabel)
trnData = [(row[1].split(' ')) for row in trn]
trnData = [list(map(int,row)) for row in trnData]

tstLabel = [int(row[0]) for row in tst]
tstLabel = to_categorical(tstLabel)
tstData = [(row[1].split(' ')) for row in tst]
tstData = [list(map(int,row)) for row in tstData]

dataFile = open("fer2013.dat", 'wb')
pk.dump((trnLabel,trnData,tstLabel,tstData), dataFile, pk.HIGHEST_PROTOCOL)
dataFile.close()