from contextlib import contextmanager
import numpy as np
from pymilvus import Collection, MilvusClient


EMBEDDING_DIM=1024
numpy_mmap_file_dir="/mnt/rangehow/in-context-pretraining/output/embed/fineweb_edu_500/tokenizer-BAAI_bge_large_en_v1.5/seq_len-5120/chunk_len-512/model-BAAI_bge_large_en_v1.5/fineweb_edu_500.jsonl.npy"
BATCH_SIZE=10000

client = MilvusClient("./ICLP.db")


index_params = client.prepare_index_params()
index_params.add_index(
    field_name="vector",
    metric_type="COSINE",
    index_type="HNSW",
    index_name="vector_index",
    # params={ "M": 32768,"efConstruction":256 }
)


client.create_collection(
    collection_name="ICLP",
    dimension=EMBEDDING_DIM , # The vectors we will use in this demo has 384 dimensions
    index_params=index_params,
)


@contextmanager
def memmap(*args, **kwargs):
    reference = np.memmap(*args, **kwargs)
    yield reference
    del reference


with memmap(numpy_mmap_file_dir, dtype=np.float32, mode='r') as embeds_flat:
    # 获取向量维度（假设是768，如果不是，请相应调整）
    vector_dim = EMBEDDING_DIM
    num_vectors = embeds_flat.shape[0] // vector_dim
    assert embeds_flat.shape[0]%EMBEDDING_DIM==0,"在这里设置的向量维度必须和转句向量所用的一致。"
    # 重塑数组以获得正确的向量形状
    embeds = embeds_flat.reshape(num_vectors, vector_dim)
    

    for start in range(0, num_vectors, BATCH_SIZE):
        end = min(start + BATCH_SIZE, num_vectors)

        data=[ {"id":i,"vector": embeds[i]} for i in range(start,end) ]
        
        res = client.insert(
            collection_name="ICLP",
            data=data
        )
        print(res)


res = client.list_indexes(
    collection_name="ICLP"
)

print(res)