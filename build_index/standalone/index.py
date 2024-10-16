
from contextlib import contextmanager
import numpy as np
from pymilvus import Collection, MilvusException, connections, db,utility
from pymilvus import MilvusClient, DataType
from pymilvus.bulk_writer import LocalBulkWriter, BulkFileType
from pymilvus import bulk_import
EMBEDDING_DIM = 1024
BATCH_SIZE = 100000



EMBEDDING_DIM = 1024
BATCH_SIZE = 100000


conn = connections.connect(host="127.0.0.1", port=19530)

try:
    db.create_database("ICLP")
except MilvusException as e:
    # 这个异常不需要处理，是database已经存在了，其他的还是照常抛出
    pass 

client = MilvusClient(
    uri="http://localhost:19530",
    db_name="ICLP"
)


# You need to work out a collection schema out of your dataset.
schema = client.create_schema(
    auto_id=True,
    enable_dynamic_field=True
)
schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM)
schema.verify()
client.create_collection(
    collection_name="ICLP",
    schema=schema
)

writer = LocalBulkWriter(
    schema=schema,
    local_path='.',
    segment_size=512 * 1024 * 1024, # Default value
    file_type=BulkFileType.NUMPY
)

numpy_mmap_file_dir="/mnt/rangehow/in-context-pretraining/output/embed/fineweb_edu_500/tokenizer-BAAI_bge_large_en_v1.5/seq_len-5120/chunk_len-512/model-BAAI_bge_large_en_v1.5/fineweb_edu_500.jsonl.npy"

@contextmanager
def memmap(*args, **kwargs):
    reference = np.memmap(*args, **kwargs)
    yield reference
    del reference

# 使用内存映射打开文件
with memmap(numpy_mmap_file_dir, dtype=np.float32, mode='r') as embeds_flat:
    # 获取向量维度（假设是768，如果不是，请相应调整）
    vector_dim = EMBEDDING_DIM
    from pdb import set_trace
    set_trace()
    num_vectors = embeds_flat.shape[0] // vector_dim
    assert embeds_flat.shape[0]%EMBEDDING_DIM==0,"在这里设置的向量维度必须和转句向量所用的一致。"
    # 重塑数组以获得正确的向量形状
    embeds = embeds_flat.reshape(num_vectors, vector_dim)
    

    for start in range(0, num_vectors, BATCH_SIZE):
        end = min(start + BATCH_SIZE, num_vectors)
        
        # 只读取一个批次的向量
        batch = embeds[start:end]
        
        for _, vector in enumerate(batch, start=start):
            writer.append_row({
                "vector": vector.tolist()
            })
        
        print(f"Processed vectors {start} to {end}")

# 提交写入操作
writer.commit()
print(writer.batch_files)

utility.do_bulk_insert(
    collection_name="test_collection",
    files=["/mnt/rangehow/in-context-pretraining/build_embeds/8d3a3105-0ce0-447d-860f-093c1f20b020/1/vector.npy"],
) 


index_params = {
    "metric_type": "L2",
    "index_type": "GPU_IVF_PQ", # Or 
    "params": {
        "nlist": 1024
    }
}
