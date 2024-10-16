from pymilvus import MilvusClient, connections, db
from pymilvus.exceptions import MilvusException



client.create_collection("ICLP",dimension=EMBEDDING_DIM,auto_id=True,enable_dynamic_field=True)

