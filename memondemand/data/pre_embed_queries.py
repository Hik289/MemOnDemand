import os as _os, sys as _sys
REPO_ROOT = _os.environ.get("MEMONDEMAND_REPO_ROOT", _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..")))
_sys.path.insert(0, REPO_ROOT)
"""Pre-compute query embeddings through the general embedding endpoint."""
import numpy as np, pandas as pd, time, json
from memondemand.data.general_embedder import GeneralEmbedder

src = 'manifests/erag_queries.parquet'
out_npy = 'manifests/erag_queries_emb_general.npy'
out_ids = 'manifests/erag_queries_emb_general_ids.json'

df = pd.read_parquet(src)
print(f'Loaded {len(df)} queries from {src}, cols={list(df.columns)}')
texts = df['query_text'].tolist()
ids = df['query_id'].tolist()

emb = GeneralEmbedder()
t0 = time.time()
vecs = emb.encode(texts, batch_size=128)
print(f'Embedded {vecs.shape} in {time.time()-t0:.1f}s')

np.save(out_npy, vecs)
with open(out_ids, 'w') as f:
    json.dump({'query_ids': ids, 'query_texts': texts, 'dim': int(vecs.shape[1])}, f)
print(f'Wrote {out_npy} and {out_ids}')
