from contextlib import contextmanager
import numpy as np
import faiss
import os
from ipdb import set_trace as bp
from tqdm import tqdm

faiss_index_dir = "/mnt/rangehow/in-context-pretraining/output/embed/fineweb_edu/tokenizer-BAAI_bge_large_en_v1.5/seq_len-5120/chunk_len-512/model-BAAI_bge_large_en_v1.5/faiss_index.bin"
numpy_mmap_file_dir =  "/mnt/rangehow/in-context-pretraining/output/embed/fineweb_edu/tokenizer-BAAI_bge_large_en_v1.5/seq_len-5120/chunk_len-512/model-BAAI_bge_large_en_v1.5/fineweb_edu_500.jsonl.npy"
EMBEDDING_DIM = 1024
BATCH_SIZE = 10000
TOP_K = 50

@contextmanager
def memmap(*args, **kwargs):
    reference = np.memmap(*args, **kwargs)
    yield reference
    del reference

