import sys

import numpy as np
from omegaconf import OmegaConf, DictConfig
from pathlib import Path, PosixPath
import logging
import pprint
import hydra

# import faiss
import os
from ipdb import set_trace as bp
from retro_z_utils import init_logging, log

from retro_z_data_xforms import (
    parallel_chunks_files_to_embeds_files,
    embeds_dir_plus_index_to_cached_knns,
    tokens_files_to_chunks_files,
    chunks_files_to_embeds_files,
    docs_files_to_tokens_files,
    embeds_dir_to_index,
    embeds_dir_to_index_swj,
    import_docs,
)

from transformers import AutoTokenizer

"""
['', '/private/home/hcir/.conda/envs/retro/lib/python3.8/site-packages/_pdbpp_path_hack', '/private/home/hcir/.conda/envs/retro/lib/python38.zip', '/private/home/hcir/.conda/envs/retro/lib/python3.8', '/private/home/hcir/.conda/envs/retro/lib/python3.8/lib-dynload', '/private/home/swj0419/.local/lib/python3.8/site-packages', '/private/home/hcir/.conda/envs/retro/lib/python3.8/site-packages', '/private/home/hcir/src2/richjames0/autofaiss', '/private/home/hcir/src/richjames0/RETRO-pytorch', '/private/home/hcir/src/sksq96/pytorch-summary', '/private/home/hcir/.conda/envs/retro/lib/python3.8/site-packages/locket-0.2.1-py3.8.egg', "/private/home/hcir/.conda/envs/retro/lib/python3.8/site-packages"]
"""
# TODO: pull these out
import sys

# sys.path.append("/private/home/swj0419/rlm_pretrain/hcir/retro-z/retro_z/RETRO-pytorch")
from retro_z_data_xforms import get_tokenizer

# BookCorpusFair_10.jsonl.lz4.npy
TFDS_DATASET_VERSION = "1.0.0"
# 2x number of CPUs on devfair
FAISS_NUM_THREADS = 160


def get_source_dir_path(cfg: DictConfig):
    return Path(cfg.source_path)


def get_documents_dir_path(cfg: DictConfig):
    return Path(cfg.base_path) / f"{str(cfg.dataset_name)}"


def get_normalized_tokenizer_name(cfg: DictConfig):
    # / in tokenizer name is not valid for a directory name
    return cfg.tokens.tokenizer.name.replace("/", "_").replace("-", "_")


def get_tokens_dir_path(cfg: DictConfig):
    return (
        get_documents_dir_path(cfg) / f"tokenizer-{get_normalized_tokenizer_name(cfg)}"
    )


def get_sequences_dir_path(cfg: DictConfig):
    return get_tokens_dir_path(cfg) / f"seq_len-{cfg.sequences.seq_len}"


# def get_chunks_dir_path(cfg: DictConfig):
#     return get_sequences_dir_path(cfg) / f"chunk_len-{cfg.chunks.chunk_len}"


def get_normalized_model_name(cfg: DictConfig):
    # / in model name is not valid for a directory name
    return cfg.embeddings.model.name.replace("/", "_").replace("-", "_")


def get_embeddings_dir_path(cfg: DictConfig):
    return get_tokens_dir_path(cfg) / f"model-{get_normalized_model_name(cfg)}"
 

"""
index: PosixPath('/checkpoint/swj0419/dataset_name-1t-0/tokenizer-facebook_contriever/seq_len-2048/chunk_len-512/model-facebook_contriever/index_string-flat')
knn_dir_path: PosixPath('/checkpoint/swj0419/dataset_name-1t-0/tokenizer-facebook_contriever/seq_len-2048/chunk_len-512/model-facebook_contriever/index_string-flat/k-50')
embeddings_file_path: PosixPath('/checkpoint/swj0419/dataset_name-1t-0/tokenizer-facebook_contriever/seq_len-2048/chunk_len-512/model-facebook_contriever/train01.jsonl.lz4.npy')
knns: query_id2docid: /checkpoint/swj0419/dataset_name-1t-0/tokenizer-facebook_contriever/seq_len-2048/chunk_len-512/model-facebook_contriever/index_string-flat/k-50/knns.npy
knns_dists: query_id2docid: /checkpoint/swj0419/dataset_name-1t-0/tokenizer-facebook_contriever/seq_len-2048/chunk_len-512/model-facebook_contriever/index_string-flat/k-50/knns.npy.dist

"""


def generate_embeddings(cfg: DictConfig, tokenizer_info):
    with log("Generating embeddings"):
        embeddings_dir_path = get_embeddings_dir_path(cfg)
        os.makedirs(embeddings_dir_path, exist_ok=True)
        print("get_tokens_dir_path(cfg): ", get_tokens_dir_path(cfg))

        # # # swj: to be deleted
        # chunks_files_to_embeds_files(get_chunks_dir_path(cfg), embeddings_dir_path, cfg.embeddings.model,
        #                                  cfg.chunks.chunk_len, cfg.embeddings.batch_size, cfg.embeddings.embed_dim)
        # 1/0
        # bp()
        # if cfg.embeddings.parallel.num_workers == 1:
        #     chunks_files_to_embeds_files(get_chunks_dir_path(cfg), embeddings_dir_path, cfg.embeddings.model,
        #                                  cfg.chunks.chunk_len, cfg.embeddings.batch_size,tokenizer_info)
        # else:
        #     # bp()
        tokens_dir_path = [get_tokens_dir_path(cfg)/f.name.rsplit('.')[0] for f in sorted(list(get_source_dir_path(cfg).glob("*.jsonl")))]
        

        parallel_chunks_files_to_embeds_files(
            tokens_dir_path,
            embeddings_dir_path,
            cfg.embeddings.batch_size,
            cfg.embeddings.model,
            cfg.chunks.chunk_len,
            cfg.embeddings.parallel,
            tokenizer_info,
        )


# def generate_chunks(cfg: DictConfig, tokenizer_info):
#     with log("Generating chunks"):
#         chunks_dir_path = get_chunks_dir_path(cfg)
#         os.makedirs(chunks_dir_path, exist_ok=True)

#         tokens_files_to_chunks_files(
#             get_tokens_dir_path(cfg),
#             chunks_dir_path,
#             cfg.chunks.chunk_len,
#             tokenizer_info,
#         )


def generate_tokens(cfg: DictConfig):
    with log("Generating tokens"):
        tokens_dir_path = get_tokens_dir_path(cfg)
        os.makedirs(tokens_dir_path, exist_ok=True)

        docs_files_to_tokens_files(
            PosixPath(cfg.source_path), tokens_dir_path, cfg.tokens.tokenizer
        )



def completeness_check(cfg, tokenizer_info):
    
    source_file_paths = sorted(list(get_source_dir_path(cfg).glob("*.jsonl")))
    embed_file_path = get_embeddings_dir_path(cfg)
    
    # 文件数量检查
    for source_file in source_file_paths:
        source_file_name = source_file.name.rsplit('.')[0]
        if not os.path.exists(embed_file_path/source_file_name):
            raise FileNotFoundError(f"原文件 {source_file} 对应的 embed 文件未找到")
    
    # 行数检查

    for source_file in source_file_paths:
        # 获取源文件的行数
        with open(source_file, "r", encoding="utf-8") as f:
            source_lines = sum(1 for _ in f)

        source_file_name = source_file.name.rsplit('.')[0]
        # 获取embeddings文件的行数
        embed_file_size = os.path.getsize(embed_file_path/source_file_name)
        embed_dim = tokenizer_info.embed_dim 
        embed_dtype = np.float32  # 假设数据类型为float32，如果不是请修改
        embed_item_size = np.dtype(embed_dtype).itemsize * embed_dim
        embed_lines = embed_file_size // embed_item_size

        # 检查行数是否匹配
        assert source_lines == embed_lines, (
            f"文件 {source_file.name} 的行数不匹配。"
            f"源文件行数：{source_lines}，embeddings行数：{embed_lines}"
        )

    print("所有文件的行数检查完成，匹配成功！")


def validate_config_interdependencies(cfg: DictConfig):
    assert (
        cfg.sequences.seq_len % cfg.chunks.chunk_len
    ) == 0, "Sequence length must be divisible by chunk size"


# from argparse import ArgumentParser
# def parse_args():
#     parser=ArgumentParser()
#     parser.add_argument("")
    
#     return parser.parse_args()


@hydra.main(config_path="configs", config_name="example_config", version_base="1.2")
def main(cfg: DictConfig):
    init_logging()
    logging.info(f"Executing with config:\n {pprint.pformat(OmegaConf.to_object(cfg))}")
    # logging.critical('Press enter to continue'); input()

    # validate_config_interdependencies(cfg)

    # if (cfg.index.generate and not cfg.index.index_reference) or cfg.precalculated_knns.generate:
    #     faiss.omp_set_num_threads(FAISS_NUM_THREADS)

    # if cfg.documents.generate:
    #     generate_documents(cfg)

    # [rangehow]：让这个方法来自xforms，从而初始化那一堆超参数，写在条件外面强制执行，就是config总得带上。
    tokenizer, tokenizer_info = get_tokenizer(cfg.tokens.tokenizer)
    # [rangehow]: 既然已经要拿前面作为表征，还设置啥chunk_len，给我自己推断！
    cfg.chunks.chunk_len = tokenizer.model_max_length
    if cfg.tokens.generate:

        # [rangehow]：我要显式的获得一个tokenizer，自动获取xforms里大量写死的超参数，防止日后出错。
        # get_tokenizer()

        # cfg.tokenizer = AutoTokenizer.from_pretrained(cfg["tokens"]["tokenizer"]["name"],trust_remote_code=True)
        # 这个会创建一个二进制存储的分词文件，放在base_path/tokenizer_name/数据集.jsonl里面
        generate_tokens(cfg)

    # if cfg.chunks.generate:
    #     generate_chunks(cfg, tokenizer_info)

    if cfg.embeddings.generate:
        # get_bert()
        generate_embeddings(cfg, tokenizer_info)

    if cfg.enable_completeness_check:
        completeness_check(cfg, tokenizer_info)


if __name__ == "__main__":
    main()
