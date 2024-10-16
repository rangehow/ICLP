
from contextlib import contextmanager
import numpy as np
from pymilvus import Collection, MilvusClient, connections, db,utility,


EMBEDDING_DIM = 1024
BATCH_SIZE = 100000

conn = connections.connect(
    host="127.0.0.1",
    port="19530",
    db_name="ICLP"
)
client = MilvusClient(
    uri="http://127.0.0.1:19530",
    db_name="ICLP"
)

res = client.describe_collection(
    collection_name="ICLP"
)

print(res)


task_id=utility.do_bulk_insert(
    collection_name="ICLP",
    files=["/mnt/rangehow/in-context-pretraining/build_embeds/8d3a3105-0ce0-447d-860f-093c1f20b020/1/$meta.npy","/mnt/rangehow/in-context-pretraining/build_embeds/8d3a3105-0ce0-447d-860f-093c1f20b020/1/vector.npy"],
) 
bulk_import(
    
)
task_state=utility.get_bulk_insert_state(
    task_id=task_id
)

# print("Before insertion:", before_info)
print(task_state)