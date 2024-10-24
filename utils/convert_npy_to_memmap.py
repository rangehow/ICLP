from contextlib import contextmanager
import numpy as np
from pathlib import Path

@contextmanager
def memmap(*args, **kwargs):
    reference = np.memmap(*args, **kwargs)
    yield reference
    del reference

def convert_npy_to_memmap(shard_filename):
    embeds_file_path=shard_filename
    # 读取npy文件
    embeds_shard = np.load(shard_filename)
    processed_chunks = len(embeds_shard)
    
    # 删除原始文件
    if isinstance(shard_filename, str):
        shard_filename = Path(shard_filename)
    shard_filename.unlink()
    
    # 创建并写入memmap文件
    with memmap(
        embeds_file_path,
        dtype=np.float32,
        mode="w+",
        shape=(processed_chunks, 1024)
    ) as embeds:
        embeds[:] = embeds_shard
    
    


convert_npy_to_memmap("input.npy")
