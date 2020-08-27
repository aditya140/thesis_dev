# %%
import pandas as pd
df = pd.read_pickle("../data/nela/data.pkl")
# df = df.head(20000)
from tqdm import tqdm
tqdm.pandas()
import numpy as np
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
    row_encoded = np.empty(shape=(0,1024))
    batch_size = 256
    for idx in tqdm(range(df.shape[0]//batch_size)):
        row = idx*batch_size
        x = df.iloc[row:row+batch_size][col].values
        row_encoded = np.append( row_encoded,model.encode(x),axis=0)
    row = row+batch_size
    x=df.iloc[row:]["title_clean"].values
    row_encoded = np.append( row_encoded,model.encode(x),axis=0)
    return row_encoded
# %%
embed=featurize_stsb(df,"title_clean")

 # %%
df["title_emb"] = pd.Series(embed.tolist())
# %%
df.to_pickle(path="../data/nela/embedded_df.pkl")
# %%
df = pd.read_pickle("../data/nela/embedded_df_sample.pkl")
# %%
from sklearn.cluster import KMeans,DBSCAN
col = "title_emb"
corpus_embeddings = df[col].values.tolist()
clustering_model = DBSCAN(eps=12, min_samples=4)
clustering_model.fit(corpus_embeddings)
cluster_assignment = clustering_model.labels_
num_clusters = len(set(clustering_model.labels_))

# %%
clustered_sentences = [[] for i in range(num_clusters)]
for sentence_id, cluster_id in enumerate(cluster_assignment):
    clustered_sentences[cluster_id].append(df["title_clean"].values[sentence_id])

for i, cluster in enumerate(clustered_sentences):
    print("Cluster ", i)
    print(cluster)
    print("")
# %%
