# -*- coding: utf-8 -*-
#!/usr/bin/python
#
#

_CODES= 'E:\\Dropbox\\ProjectML\\Codes'
_DATA='E:\\Dropbox\\ProjectML\\Data'
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
       'jazz and blues', 'soul and reggae', 'punk',]

import os
os.chdir(_CODES)

import numpy as np
import pandas as pd
import random

rawDF=pd.read_csv(_DATA+'\\rawDF.txt')
rawDF.genre.value_counts()

#Filtrando apenas as observacoes cujos generos pertencem a _GENRES
rawDF=rawDF.loc[np.any([rawDF.loc[:,'genre']==i for i in _GENRES],0),:]

#Criando DataFrame de modelagem e de teste
modDF=pd.DataFrame()
testDF=pd.DataFrame()

#Random Sampling
for g in np.unique(rawDF['genre']):
    idx=random.sample(rawDF.loc[rawDF.loc[:,'genre']==g,:].index,3200)
    modDF=modDF.append(rawDF.ix[idx[0:2880]].copy())
    testDF=testDF.append(rawDF.ix[idx[2880:3200]].copy())

#Salvando DataFrames de modelagem e de teste
#modDF.to_csv(_DATA+'\\modDF.txt')
#testDF.to_csv(_DATA+'\\testDF.txt')
