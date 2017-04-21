# -*- coding: utf-8 -*-
#!/usr/bin/python
#
#

import numpy as np
import pandas as pd
import random
from sklearn import svm
from sklearn import ensemble
import cPickle

def ecocSVM(X,y,codeDF,resultsPath,nLabels=None,bootstrap=False):
    if nLabels==None:
        nLabels=np.size(np.unique(y))
        
    defaultParameters={
        '_C':1.0,
        '_kernel':'rbf',
        '_degree':3,
        '_gamma':0.0,
        '_coef0':0.0,
        '_probability':False,
        '_shrinking':True,
        '_random_state':None
    }

    resultados=pd.DataFrame(columns=['meanNorm','stdNorm','accRef','accVal','svmID','svmPath'])   

    #Criando modelos para cada uma das linhas da Code Matrix
    for i in np.arange(codeDF.shape[0]):
        resample=pd.DataFrame()
        par=defaultParameters
        for j in np.arange(nLabels):
            if codeDF.iloc[i,j]==1:
                aux=X.loc[y==str(codeDF.columns[j]),:].copy()
                aux.loc[:,'codeClass']=1
                resample=resample.append(aux)
            if codeDF.iloc[i,j]==-1:
                aux=X.loc[y==str(codeDF.columns[j]),:].copy()
                aux.loc[:,'codeClass']=-1
                resample=resample.append(aux)
        for j in np.arange(nLabels,codeDF.shape[1]):
            if not np.isnan(codeDF.iloc[i,j]):
                par[str(codeDF.columns[j])]=codeDF.iloc[i,j]
        
        #Extraindo amostra aleatoria (treinamento 80 / validação 20)
        idx=random.sample(resample.index,resample.shape[0])
        refX=resample.ix[idx[0:int(0.8*resample.shape[0])]]
        valX=resample.ix[idx[int(0.8*resample.shape[0]):resample.shape[0]]]
        
        #Normalizando dados e armazenando em "resultados" as medias e stds das normalizacoes
        meanNorm=[]
        stdNorm=[]
        for col in X.columns.values:
            meanNorm.append(np.mean(refX.loc[:,col],0))
            stdNorm.append(np.std(refX.loc[:,col],0))
            valX.loc[:,col]=(valX.loc[:,col]-np.mean(refX.loc[:,col],0))/np.std(refX.loc[:,col],0)  
            refX.loc[:,col]=(refX.loc[:,col]-np.mean(refX.loc[:,col],0))/np.std(refX.loc[:,col],0)
        resultados.loc[i,'meanNorm']=meanNorm
        resultados.loc[i,'stdNorm']=stdNorm
        
        _SVC=svm.SVC(par['_C'],par['_kernel'],par['_degree'],par['_gamma'],par['_coef0'],\
        par['_probability'],par['_shrinking'],random_state=par['_random_state'],class_weight='auto')
        #_SVC.predict()
        #Fazendo o fit e calculando a accuracy
        _SVC.fit(refX.drop('codeClass',1),refX.loc[:,'codeClass'])
        resultados.loc[i,'accRef']=_SVC.score(refX.drop('codeClass',1),refX.loc[:,'codeClass'])
        resultados.loc[i,'accVal']=_SVC.score(valX.drop('codeClass',1),valX.loc[:,'codeClass'])
        
        #Salvando modelo em resultsPath
        cPickle.dump(_SVC,open(resultsPath+'/_SVM_'+str(i)+'.p','wb'))
        resultados.loc[i,'svmID']='_SVM_'+str(i)
        resultados.loc[i,'svmPath']=resultsPath+'/_SVM_'+str(i)+'.p'
    
    finalResults=pd.concat([codeDF,resultados],1)
    cPickle.dump(finalResults,open(resultsPath+'/finalResults.p','wb'))
    finalResults.drop(['meanNorm','stdNorm'],1).to_excel(resultsPath+'/finalResults.xlsx')

def combineEcocSVM(X,y,resultsPath,probability=False):
    ecocDF=pd.read_pickle(resultsPath+'/finalResults.p')
    predictDF=pd.DataFrame()
    for i in np.arange(ecocDF.shape[0]):
        aux=X.copy()
        for c in np.arange(X.shape[1]):
            aux.iloc[:,c]=(aux.iloc[:,c]-ecocDF.loc[i,'meanNorm'][c])/ecocDF.loc[i,'stdNorm'][c]
        _SVC=cPickle.load(open(resultsPath+'/'+ecocDF.loc[i,'svmID']+'.p','rb'))
        if probability==True and ecocDF.loc[i,'_probability']==True:
            predictDF.loc[:,ecocDF.loc[i,'svmID']]=_SVC.predict_proba(aux)[:,0]
        else:
            predictDF.loc[:,ecocDF.loc[i,'svmID']]=_SVC.predict(aux)
    predictDF.loc[:,'y']=y
    return predictDF

def classifyRFC(predictDF):
    _RFC=ensemble.RandomForestClassifier(100,oob_score=True)
    _RFC.fit(predictDF.drop('y',1),predictDF.y)
    return _RFC.oob_score_
    
def classifyETC(predictDF, n_trees):
    _ETC=ensemble.ExtraTreesClassifier(n_trees,oob_score=True, bootstrap=True)
    _ETC.fit(predictDF.drop('y',1),predictDF.y)
    return _ETC.oob_score_    