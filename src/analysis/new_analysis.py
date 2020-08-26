# %%
import pandas as pd
df = pd.read_pickle("../data/nela/data.pkl")
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
    batch_size = 64
    for row in tqdm(range(df.shape[0]//64)):
        x = df.iloc[row:row+64]["title_clean"].values
        print(x)
        row_encoded = np.append( row_encoded,model.encode(x),axis=0)
        print(row_encoded)
    print(len(row_encoded))
# %%
df=featurize_stsb(df,"title_clean")
# %%
 