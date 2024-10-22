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

def search_nearest_neighbors(vectors, top_k=TOP_K):
    results = []
    total_batches = (len(vectors) + BATCH_SIZE - 1) // BATCH_SIZE
    with tqdm(total=total_batches, desc="Searching nearest neighbors") as pbar:
        for i in range(0, len(vectors), BATCH_SIZE):
            batch = vectors[i:i+BATCH_SIZE]
            distances, indices = index.search(batch, top_k)
            batch_results = [
                [{"id": int(idx), "distance": float(dist)} for idx, dist in zip(row_indices, row_distances)]
                for row_indices, row_distances in zip(indices, distances)
            ]
            
            results.extend(batch_results)
            pbar.update(1)
    return results

# 加载FAISS索引
index = faiss.read_index(faiss_index_dir)
# 如果GPU OOM可以注释这一行，我还没太多办法可以处理这个
index = faiss.index_cpu_to_all_gpus(index)
# index.nprobe = 500
# 设置输出文件名
output_file = os.path.join(os.path.dirname(numpy_mmap_file_dir), "nearest_neighbors.npy")

with memmap(numpy_mmap_file_dir, dtype=np.float32, mode='r') as embeds_flat:
    vector_dim = EMBEDDING_DIM
    num_vectors = embeds_flat.shape[0] // vector_dim
    assert embeds_flat.shape[0] % EMBEDDING_DIM == 0, "在这里设置的向量维度必须和转句向量所用的一致。"
    embeds = embeds_flat.reshape(num_vectors, vector_dim)

    # 搜索最近邻
    search_results = search_nearest_neighbors(embeds)
    
    # 将结果转换为numpy数组
    neighbor_ids = np.array([[hit["id"] for hit in result] for result in search_results], dtype=np.int64)

    print(f"搜索结果形状为{neighbor_ids.shape}")
    # 保存结果为npy文件
    np.save(output_file, neighbor_ids)
    
    
    
def check_completeness(vectors, results, tolerance=1e-5):
    for i, neighbors in enumerate(results):
        if neighbors[0]["id"] != i or neighbors[0]["distance"] > tolerance:
            bp()
            print(f"Warning: Vector at index {i} does not have itself as the nearest neighbor.")
            print(f"Nearest neighbor: id={neighbors[0]['id']}, distance={neighbors[0]['distance']}")
            # return False
    return True



# is_complete = check_completeness(embeds, search_results)
# print(f"Search results are complete: {is_complete}")
print(f"Results saved to {output_file}")