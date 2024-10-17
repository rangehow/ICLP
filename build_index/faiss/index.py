import numpy as np
import faiss
import os

# 设置参数
EMBEDDING_DIM = 1024
numpy_mmap_file_dir = "/mnt/rangehow/in-context-pretraining/output/embed/fineweb_edu_500/tokenizer-BAAI_bge_large_en_v1.5/seq_len-5120/chunk_len-512/model-BAAI_bge_large_en_v1.5/fineweb_edu_500.jsonl.npy"
BATCH_SIZE = 10000

# 创建内存映射
memmap = np.memmap(numpy_mmap_file_dir, dtype='float32', mode='r')

# 获取嵌入向量的总数
num_vectors = memmap.shape[0] // EMBEDDING_DIM

# 重塑内存映射以获得正确的嵌入向量形状
vectors = memmap.reshape(num_vectors, EMBEDDING_DIM)

index = faiss.index_factory(EMBEDDING_DIM, "IVF10_HNSW32768,Flat")
index.train(vectors)

index.add(vectors)


print(f"索引创建完成。总共索引了 {index.ntotal} 个向量。")

# 保存索引
output_dir = os.path.dirname(numpy_mmap_file_dir)
index_file = os.path.join(output_dir, "faiss_index.bin")
faiss.write_index(index, index_file)

print(f"索引已保存到: {index_file}")