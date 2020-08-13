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
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

def featurize_stsb(df,col):
    df[col+"_features"] = df[col].apply((lambda x: model.encode([x])[0]))
    return df

# %%

import numpy as np
df=featurize_stsb(df,"context")
df=featurize_stsb(df,"response")
df=featurize_stsb(df,"stimuli")

# %%
import io

df["combined_features"] = df.apply(lambda x:x["context_features"]+x["stimuli_features"]+x["response_features"] , axis=1)

out_v = io.open('vecs.tsv', 'w', encoding='utf-8')
out_m = io.open('meta.tsv', 'w', encoding='utf-8')
out_m.write("context"+"\t"+"stimuli"+"\t"+"response"+"\t"+"novelty"+"\t"+ "\n")

for index, row in df.iterrows():
    vec = row["combined_features"]
    out_m.write(row["context"]+"\t"+row["stimuli"]+"\t"+row["response"]+"\t"+str(row["novelty"])+"\t"+ "\n")
    out_v.write('\t'.join([str(x) for x in vec]) + "\n")

out_v.close()
out_m.close()


# %%
from sklearn.cluster import KMeans
from tqdm import tqdm

fig, axs = plt.subplots(ncols=1)
fig.set_size_inches(6,5)

for col_no,col in enumerate(["context","response","stimuli"]):
    X=df[col+"_features"]
    
    wcss = []
    for i in (range(1, 80)):
        
        kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
        kmeans.fit(list(X.values))
        wcss.append(kmeans.inertia_)

    sns.lineplot(x=range(1, 80),y=wcss,color=['tomato',"turquoise","darkblue"][col_no],ax=axs)
    df[col+"_cluster"]=list(kmeans.labels_)
plt.legend(labels=['context', 'response', 'stimuli'])


# %%
from sklearn.cluster import KMeans,DBSCAN
col = "stimuli"
corpus_embeddings = df[col+"_features"].values.tolist()
# clustering_model = DBSCAN(eps=12, min_samples=4)
clustering_model = KMeans(n_clusters=21, init='k-means++', max_iter=300, n_init=10, random_state=0)
clustering_model.fit(corpus_embeddings)
cluster_assignment = clustering_model.labels_
num_clusters = len(set(clustering_model.labels_))

clustered_sentences = [[] for i in range(num_clusters)]
for sentence_id, cluster_id in enumerate(cluster_assignment):
    clustered_sentences[cluster_id].append(df[col].values[sentence_id])

for i, cluster in enumerate(clustered_sentences):
    print("Cluster ", i+1)
    print(cluster)
    print("")


# %%
from nlptriples import triples,setup
rdf=triples.RDF_triple()
from tqdm import tqdm
def extract_triples(sents):
    try:
        sents=sents.split(".")
        triples=[rdf.extract(sent) for sent in sents]
        return triples
    except:
        return []

# %%
tqdm.pandas()
df["stimuli_triples"]=df["stimuli"].progress_apply(extract_triples)
df["response_triples"]=df["response"].progress_apply(extract_triples)
df["context_triples"]=df["context"].progress_apply(extract_triples)

# %%
df.to_csv("cleaned.csv")

# %%
