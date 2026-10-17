import pandas as pd, numpy as np
from scipy.stats import spearmanr
F='data/external/flab/'
g=pd.read_csv('data/external/gdpa/GDPa1_v1.2_20250814.csv')
g['k']=g.vh_protein_sequence.str.strip()+'|'+g.vl_protein_sequence.str.strip()
pairs=[('aggregation/jain2017biophysical_ACSINS.csv','AC-SINS_pH7.4'),
('aggregation/jain2017biophyscial_HICRT.csv','HIC'),
('polyreactivity/jain2017biophysical_SMACRT.csv','SMAC'),
('polyreactivity/jain2017biophysical_PSR.csv','PR_CHO'),
('polyreactivity/jain2017biophysical_PSR.csv','PR_Ova'),
('polyreactivity/jain2017biophysical_ELISA.csv','PR_CHO'),
('polyreactivity/jain2017biophysical_ELISA.csv','PR_Ova')]
for f,c in pairs:
    j=pd.read_csv(F+f); j['k']=j.heavy.str.strip()+'|'+j.light.str.strip()
    m=j.merge(g,on='k')[[j.columns[2],c]].dropna()
    r,_=spearmanr(m.iloc[:,0],m.iloc[:,1])
    print(f"{f.split('/')[1][:32]:34s} vs {c:14s} n={len(m):3d} rho={r:+.3f}")
