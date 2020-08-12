# %%
import pandas as pd
import numpy as np
import glob
import csv
import os
csv_files = (glob.glob("../data/*.csv"))
# %%
dfs={}
for i in csv_files:
    file_name=i.replace('.csv',"").split('/')[-1]
    dfs[file_name]=pd.read_csv(i,error_bad_lines=False)
    dfs[file_name]=dfs[file_name].loc[:, ~dfs[file_name].columns.str.contains('^Unnamed')]
for i in dfs.keys():
    print(i)
    
    
# modify all_requirements.csv so that it is rated
print("\nAll Responses : ",dfs["all_requirements"].shape)
dfs["rated_requirements"]=dfs["all_requirements"][dfs["all_requirements"]["id"].isin(dfs["creativity-ratings"]["tid"].values)]
print("Rated Responses : ",dfs["rated_requirements"].shape)
df=dfs["all_requirements"].merge(dfs["creativity-ratings"],how="inner",left_on='id',right_on='tid')

# %%
import matplotlib.pyplot as plt
import seaborn as sns

fig, axs = plt.subplots(ncols=3)
fig.set_size_inches(12,3)

sns.distplot(df['novelty'], hist=True, kde=True, color = 'tomato',
             hist_kws={'edgecolor':'black'},ax=axs[0])


sns.distplot(df['usefulness'], hist=True, kde=True, color = 'turquoise',
             hist_kws={'edgecolor':'black'},ax=axs[1])


sns.distplot(df['detailedness'], hist=True, kde=True, color = 'darkblue',
             hist_kws={'edgecolor':'black'},ax=axs[2])
print("Correlation Matrix")
display(df.corr())
print("\n\nCovariance Matrix")
display(df.cov())

# %%
from sentence_transformers import SentenceTransformer
# model = SentenceTransformer('bert-large-nli-stsb-mean-tokens')
model = SentenceTransformer('roberta-large-nli-stsb-mean-tokens')



# %%
