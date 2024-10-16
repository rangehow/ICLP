from contextlib import contextmanager
import numpy as np
from pymilvus import Collection, MilvusClient

EMBEDDING_DIM = 1024

BATCH_SIZE = 10000
TOP_K = 50
collection_name = "ICLP"
# input
numpy_mmap_file_dir = "/mnt/rangehow/in-context-pretraining/output/embed/fineweb_edu_500/tokenizer-BAAI_bge_large_en_v1.5/seq_len-5120/chunk_len-512/model-BAAI_bge_large_en_v1.5/fineweb_edu_500.jsonl.npy"
# output
output_file = "nearest_neighbors_ids.npy"

client = MilvusClient("./ICLP.db")

@contextmanager
def memmap(*args, **kwargs):
    reference = np.memmap(*args, **kwargs)
    yield reference
    del reference

def search_nearest_neighbors(collection_name, vectors, top_k=TOP_K):

    results = []
    for i in range(0, len(vectors), BATCH_SIZE):
        batch = vectors[i:i+BATCH_SIZE]
        result = client.search(
            collection_name=collection_name,
            data=batch.tolist(),
            anns_field="vector",
            # param=search_params,
            limit=top_k,
            output_fields=["id"],
            consistency_level="Strong"
        )
        results.extend(result)

    return results



with memmap(numpy_mmap_file_dir, dtype=np.float32, mode='r') as embeds_flat:
    vector_dim = EMBEDDING_DIM
    num_vectors = embeds_flat.shape[0] // vector_dim
    assert embeds_flat.shape[0] % EMBEDDING_DIM == 0, "在这里设置的向量维度必须和转句向量所用的一致。"
    embeds = embeds_flat.reshape(num_vectors, vector_dim)

    # 搜索最近邻
    search_results = search_nearest_neighbors(collection_name, embeds)

    # 将结果转换为numpy数组
    
    neighbor_ids = np.array([[hit["id"] for hit in result] for result in search_results], dtype=np.int64)
    # 保存结果为npy文件

    np.save(output_file, neighbor_ids)

print(f"Results saved to {output_file}")