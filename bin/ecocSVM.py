# -*- coding: utf-8 -*-
#!/usr/bin/python
#
#

###  GLOBAIS ###

_CODES= 'E:/Dropbox/ProjectML/Codes'
_DATA='E:/Dropbox/ProjectML/Data'

_VARS=[ 'loudness', 'tempo',
       'time_signature', 'key', 'mode', 'duration', 'avg_timbre1',
       'avg_timbre2', 'avg_timbre3', 'avg_timbre4', 'avg_timbre5',
       'avg_timbre6', 'avg_timbre7', 'avg_timbre8', 'avg_timbre9',
       'avg_timbre10', 'avg_timbre11', 'avg_timbre12', 'var_timbre1',
       'var_timbre2', 'var_timbre3', 'var_timbre4', 'var_timbre5',
       'var_timbre6', 'var_timbre7', 'var_timbre8', 'var_timbre9',
       'var_timbre10', 'var_timbre11', 'var_timbre12']

_PERF=['genre']
_GENRES=[ 'folk',  'dance and electronica',
       'jazz and blues', 'soul and reggae', 'punk']

### IMPORTANDO MODULOS ###

import os
os.chdir(_CODES)

import numpy as np
import pandas as pd
from sklearn import ensemble
from sklearn import multiclass
from sklearn import svm

import helpers as h

### CODIGO ###

#base de modelagem sem normalizacao, com todas as classes
modDF=pd.read_csv(_DATA+'/modDF.txt')
testDF=pd.read_csv(_DATA+'/testDF.txt')

#local onde esta o code dataframe
codeDFPath='E:/Dropbox/ProjectML/Data/ecocSVM/codeMtx.xlsx'
codeDF=pd.read_excel(codeDFPath)

#filtrando as variaveis dependentes da base de modelagem
X=modDF.loc[:,_VARS].reset_index(drop=True)
Xtest=testDF.loc[:,_VARS].reset_index(drop=True)
#fitlrando a variavel independente da base de modelagem
y=modDF.loc[:,'genre'].reset_index(drop=True)
ytest=testDF.loc[:,'genre'].reset_index(drop=True)
#path para salvar os resultados
resultsPath='E:/Dropbox/ProjectML/Data/ecocSVM_Renan'

#RODANDO ecocSVM
h.ecocSVM(X,y,codeDF,resultsPath)
pred=h.combineEcocSVM(X,y,resultsPath,True)
oobScore=h.classifyRFC(pred)
h.classifyRFC(pred)

_RFC=ensemble.RandomForestClassifier(150,max_leaf_nodes=80,oob_score=True)
_RFC.fit(pred.drop('y',1),pred.y)
_RFC.score(pred.drop('y',1),pred.y)
_RFC.oob_score_
_RFC.feature_importances_
#predTest=h.combineEcocSVM(Xtest,ytest,resultsPath)

testDF=pd.read_csv(_DATA+'/testDF.txt')
Xtest=testDF.loc[:,_VARS]
ytest=testDF.loc[:,'genre']
predTest=h.combineEcocSVM(Xtest,ytest,resultsPath,True)
_RFC.score(predTest.drop('y',1),predTest.y)

Xconcat=pd.concat([X,pred],1)
Xtest=testDF.loc[:,_VARS].reset_index(drop=True)
ytest=testDF.loc[:,'genre'].reset_index(drop=True)
predTest=h.combineEcocSVM(Xtest,ytest,resultsPath,True)
testConcat=pd.concat([Xtest,predTest],1)

_RFC=ensemble.ExtraTreesClassifier(90,max_features=6,min_samples_leaf=3,oob_score=True,bootstrap=True,random_state=4)
_RFC.fit(Xconcat.drop('y',1),Xconcat.y)
_RFC.score(Xconcat.drop('y',1),Xconcat.y)
_RFC.oob_score_
_RFC.score(testConcat.drop('y',1),testConcat.y)
_RFC.feature_importances_


### Multi-class tests ###

_SVC=svm.SVC(class_weight='auto')

#One versus one
_OVO=multiclass.OneVsOneClassifier(_SVC,n_jobs=-1)
_OVO.fit(X.get_values(),y.get_values())
_OVO.score(Xtest.get_values(),ytest.get_values())

#One versus rest
_OVR=multiclass.OneVsRestClassifier(_SVC,n_jobs=-1)
_OVR.fit(X.get_values(),y.get_values())
_OVR.score(Xtest.get_values(),ytest.get_values())

#ECOC
for i in np.arange(1,10):
    _ECOC=multiclass.OutputCodeClassifier(_SVC,i,n_jobs=-1)
    _ECOC.fit(X.get_values(),y.get_values())
    _ECOC.score(Xtest.get_values(),ytest.get_values())

#Random Forest usando os valores iniciais sem SVM
_RFC=ensemble.RandomForestClassifier(450, oob_score=True, n_jobs=-1)
_RFC.fit(X.get_values(),y.get_values())
_RFC.score(X.get_values(),y.get_values())
_RFC.oob_score_
_RFC.score(Xtest.get_values(),ytest.get_values())
_RFC.feature_importances_

### Confusion Matrix ###

predTest=h.combineEcocSVM(Xtest,ytest,resultsPath,True)
testConcat=pd.concat([Xtest,predTest],1)
testDF.loc[:,'RFC']=_RFC.predict(testConcat.drop('y',1))

for i in np.unique(testDF.genre):
    for j in np.unique(testDF.genre):
        print 'genero: '+i+' RFC '+j
        np.sum(np.all([testDF.loc[:,'genre']==i,testDF.loc[:,'RFC']==j],0))
