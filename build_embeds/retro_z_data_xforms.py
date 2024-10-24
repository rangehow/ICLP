import asyncio
import subprocess
from dataclasses import dataclass
from functools import partial
import multiprocessing
import datasets
from einops import rearrange
from joblib import delayed, Parallel
from typing import List, Optional
from omegaconf import DictConfig, OmegaConf
from rich.progress import track
from pathlib import Path
import pickle as pkl
import numpy as np
import jsonlines
import traceback
import logging
import random
import torch
import faiss
import math
import time
import os
from tqdm import tqdm
from datasets import Dataset, load_dataset
from datasets.distributed import split_dataset_by_node
from torch.utils.data import DataLoader

# rangehow: use sentence transformer instead of torch.hub
from sentence_transformers import SentenceTransformer
import multiprocessing as mp

# TODO: pull these out of retrieval and break connection to that codebase
import sys

sys.path.append("/mnt/rangehow/in-context-pretraining/build_embeds/RETRO-pytorch")
# from retro_pytorch.retrieval import  bert_embed #,tokenize
from chunk_logger import ChunkLogger, ChunkLoggerDummy


from submitit_utils import (
    WorkerFunctor,
    create_executor,
    await_completion_of_jobs,
    fetch_and_validate_neighbors_results,
    fetch_and_validate_embedding_results,
)

from retro_z_utils import (
    reshape_memmap_given_width,
    exists_and_is_file,
    exists_and_is_dir,
    write_jsonl_file,
    read_jsonl_file,
    read_jsonl_file_no_compress,
    range_chunked,
    init_logging,
    memmap,
    log,
)


from ipdb import set_trace as bp

WORKERS_PER_FILE_WEIGHTINGS = {
    "CommonCrawl": 4,
    "HackerNews": 0.125,
    "Enron_Emails": 0.125,
    "DM_Mathematics": 0.125,
    "BookCorpusFair": 0.125,
}

random.seed(88)
np.random.seed(88)


# TODO: get these from the tokenizer
# [rangehow]: 已经从tokenizer里获取了
@dataclass
class TokenizerInfo:
    pad_token: int
    unk_token: int
    cls_token: int
    sep_token: int
    mask_token: int
    embed_dim: int


from transformers import AutoTokenizer, AutoConfig


# [rangehow]: add this to initialize upper hyperparameter
def get_tokenizer(tokenizer_cfg: DictConfig):
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_cfg["name"], trust_remote_code=True
    )
    config = AutoConfig.from_pretrained(tokenizer_cfg["name"], trust_remote_code=True)

    tokenizer_info = TokenizerInfo(
        pad_token=tokenizer.pad_token_id,
        unk_token=tokenizer.unk_token_id,
        cls_token=tokenizer.cls_token_id,
        sep_token=tokenizer.sep_token_id,
        mask_token=tokenizer.mask_token_id,
        embed_dim=config.hidden_size,
    )

    return tokenizer, tokenizer_info


def _preprocess_dataset(instance):
    return {
        "text": list(
            map(
                lambda x: x.strip().split("##@@B")[1].replace("_nl_", "\n"),
                instance["text"],
            )
        )
    }


def _tokenize_dataset(instance, tokenizer_cfg):
    tokenizer, tokenizer_info = get_tokenizer(tokenizer_cfg)

    tokenized_doc_outer = tokenizer.batch_encode_plus(
        instance["text"],
        add_special_tokens=True,
        padding='max_length', # 因为删了chunk，所以要把这里补齐
        return_attention_mask=False,
        return_token_type_ids=False,
        truncation=True,
    )

    return {"tokens": tokenized_doc_outer.input_ids}


def get_bert(
    name: Optional[str] = None,
    device: Optional[int] = None,
) -> SentenceTransformer:

    print("device", device)
    model = SentenceTransformer(
        name,
        trust_remote_code=True,
        device=f"cuda:{device}" if device is not None else "cpu",
    )
    print("model was loaded on :", model.device)
    return model


@torch.no_grad()
def bert_embed(
    token_ids,
    pad_id=0.0,
    bert: SentenceTransformer = None,
):

    mask = token_ids != pad_id

    token_ids = token_ids.to(bert.device)
    mask = mask.to(bert.device)
    outputs = bert(dict(input_ids=token_ids, attention_mask=mask))

    return outputs["sentence_embedding"]


JSONL_PAYLOAD = "text"

GLOB_SEPARATOR = ","

NPY_GLOB = "*.npy"
LZ4_GLOB = "*.jsonl"
NPY_SUFFIX = ".npy"
LZ4_SUFFIX = ".lz4"
MAP_SUFFIX = ".map"
DIST_SUFFIX = ".dist"
LZ4_NPY_GLOB = "*.lz4.npy"

INDEX_FILENAME = "index"
KNNS_FILENAME = "knns.npy"
MAP_FILENAME = "embeddings.map"
INDICES_FILENAME = "embeddings.key"
CHUNKS_TO_DOCS_FILENAME = "chunks_to_docs.npy"

NUM_CPUS_PER_NODE = 10

JOB_BATCH_SIZE = 1024
CHUNK_BATCH_SIZE = 1024

# >= 50 means print stdout
JOBLIB_VERBOSITY = 50


import multiprocessing


def _parallel(n_jobs=-1):
    # print("n_jobs: ", n_jobs)
    # swj change
    return Parallel(n_jobs=-1, verbose=JOBLIB_VERBOSITY)
    # return Parallel(n_jobs=1, verbose=JOBLIB_VERBOSITY)


def _determine_num_chunks_per_seq(seq_len: int, chunk_len: int):
    assert chunk_len > 0, f"Invalid chunk_len: {chunk_len}"
    assert seq_len > 0, f"Invalid seq_len: {seq_len}"
    num_chunks_per_seq, mod = divmod(seq_len, chunk_len)
    assert mod == 0, f"Invalid mod: {mod}"
    # assert num_chunks_per_seq == 32, f'Invalid num_chunks_per_seq: {num_chunks_per_seq}'

    return num_chunks_per_seq


def _get_files_indices(embeddings_dir_path: Path):
    indices_file_path = embeddings_dir_path / INDICES_FILENAME
    logging.info(f"Loading files indices from {indices_file_path} and validating")

    assert exists_and_is_file(indices_file_path)
    with open(indices_file_path, "rb") as files_indices_file:
        files_indices = pkl.load(files_indices_file)

    chunk_max_index = 0
    # FIXME:
    # prev_filename = ''
    prev_chunk_max_index = 0
    index = 0
    for filename, indices in files_indices.items():
        chunk_min_index, chunk_max_index, file_index = indices
        assert len(filename) > 0, "Failed test: len(filename) > 0"
        # FIXME: decide whether to keep or remove this. 1g dataset did not sort
        # assert filename > prev_filename
        assert (
            file_index == index
        ), f"Invalid file_index ({file_index}) or index ({index})"
        assert chunk_min_index >= 0, f"Invalid chunk_min_index: {chunk_min_index}"
        assert (
            chunk_max_index > chunk_min_index
        ), f"Invalid chunk_min_index ({chunk_min_index}) or chunk_max_index ({chunk_max_index})"
        assert (
            chunk_min_index == prev_chunk_max_index
        ), f"Invalid chunk_min_index ({chunk_min_index}) or prev_chunk_max_index ({prev_chunk_max_index})"

        index += 1
        prev_chunk_max_index = chunk_max_index
        # FIXME:
        # prev_filename = filename

    return files_indices, chunk_max_index


# TODO: this could be done when generated chunks
def _create_aggregate_docs_map(
    tfds_dir_path: Path, chunks_dir_path: Path, files_indices, num_chunks_overall: int
):
    assert files_indices is not None, "Failed test: files_indices is not None"
    assert num_chunks_overall > 0, f"Invalid num_chunks_overall: {num_chunks_overall}"

    chunks_to_docs_filepath = tfds_dir_path / CHUNKS_TO_DOCS_FILENAME
    with log(f"Creating aggregate docs map at {chunks_to_docs_filepath}"):
        chunks_to_docs = np.memmap(
            chunks_to_docs_filepath,
            mode="w+",
            shape=(num_chunks_overall,),
            dtype=np.int32,
        )

    document_offset = 0
    for filename, indices in files_indices.items():
        chunk_min_index, chunk_max_index, _ = indices
        file_path = chunks_dir_path / filename
        chunks_map_file_path = Path(str(file_path) + MAP_SUFFIX)

        logging.info(
            f"Copying contents of file {chunks_map_file_path} to aggregate map"
        )
        with memmap(chunks_map_file_path, dtype=np.int32, mode="r") as chunks_map:
            assert (
                len(chunks_map) == chunk_max_index - chunk_min_index
            ), f"Invalid chunks map len ({len(chunks_map)}), chunk_max_index ({chunk_max_index}) or chunk_min_index ({chunk_min_index})"
            chunks_to_docs[chunk_min_index:chunk_max_index] = (
                chunks_map + document_offset
            )

            # TODO: simplify
            assert (
                np.max(chunks_map) == chunks_map[-1]
            ), "Failed test: np.max(chunks_map) == chunks_map[-1]"
            # +1 because the start of the next file will be a new document with a relative index of 0
            document_offset += np.max(chunks_map) + 1
    assert chunk_max_index == num_chunks_overall, f"Invalid chunk_max_index ({chunk_max_index}) or num_chunks_overall ({num_chunks_overall})"  # type: ignore

    return chunks_to_docs


def _get_aggregate_docs_map(tfds_dir_path: Path):
    chunks_to_docs_filepath = tfds_dir_path / CHUNKS_TO_DOCS_FILENAME
    return np.memmap(chunks_to_docs_filepath, mode="r", dtype=np.int32)


def _get_chunks_memmaps(chunks_dir_path: Path, files_indices, chunk_len: int):
    assert exists_and_is_dir(chunks_dir_path)
    assert files_indices is not None, "Failed test: files_indices is not None"
    # assert chunk_len > 0 and chunk_len == 64, f'Invalid chunk_len: {chunk_len}'

    chunks_memmaps = [None] * len(files_indices)
    for filename, indices in files_indices.items():
        logging.info(f"Mapping chunks from file {filename} with indices {indices}")
        _, _, filename_index = indices
        file_path = chunks_dir_path / filename
        chunks_flat = np.memmap(file_path, np.int32, "r")
        chunks, _ = reshape_memmap_given_width(chunks_flat, chunk_len)

        chunks_memmaps[filename_index] = chunks

    return chunks_memmaps


def _get_tfds_features(seq_len: int, chunk_len: int, k_: int):
    import tensorflow_datasets as tfds
    import tensorflow as tf

    # assert chunk_len > 0 and chunk_len == 64, f'Invalid chunk_len: {chunk_len}'
    # assert seq_len > 0 and seq_len == 2048, f'Invalid seq_len: {seq_len}'
    assert k_ > 0 and (k_ == 2 or k_ == 5), f"Invalid k_: {k_}"

    num_chunks_per_seq = _determine_num_chunks_per_seq(seq_len, chunk_len)

    features = {
        "example_tokens": tfds.features.Tensor(shape=(seq_len,), dtype=tf.int32),
        "example_mask": tfds.features.Tensor(shape=(seq_len,), dtype=tf.bool),
        "target_tokens": tfds.features.Tensor(shape=(seq_len,), dtype=tf.int32),
        # * 2 isn't for number of neighbors, but rather for the continuations
        "neighbor_tokens": tfds.features.Tensor(
            shape=(num_chunks_per_seq, k_, chunk_len * 2),
            dtype=tf.int32,
        ),
        "last_chunk_of_doc_flags": tfds.features.Tensor(
            shape=(num_chunks_per_seq,),
            dtype=tf.bool,
        ),
    }

    return tfds.features.FeaturesDict(features)  # type:ignore


def _add_ds_and_final_shard_to_filenames(tfds_dir_path: Path):
    # TODO: tighten this up
    file_paths = list(tfds_dir_path.glob("*.tfrecord-*"))
    num_shards = len(file_paths)
    for file_path in file_paths:
        filename = file_path.name
        file_path.rename(
            Path(os.path.dirname(file_path))
            / Path(str(filename).replace("=", "_") + f"-of-{num_shards:>05d}")
        )

    # a long time to sleep but really not worth blowing up generation for this
    time.sleep(60 * 1)


def _get_data_for_parallel_worker(
    modeling_chunks_index,
    chunks_dir_path_modeling,
    chunks_dir_path_retrieval,
    embeds_dir_path_modeling,
    embeds_dir_path_retrieval,
    tfds_dir_path,
    chunk_len,
    split,
):
    files_indices_modeling, chunks_memmaps_modeling, num_chunks_overall_modeling = (
        _get_chunks_data_for_tfds_jobs(
            chunks_dir_path_modeling, embeds_dir_path_modeling, chunk_len
        )
    )
    logging.info(
        f"Processing file {list(files_indices_modeling.keys())[modeling_chunks_index]}"
    )

    if split == "validation":
        assert (
            chunks_dir_path_modeling != chunks_dir_path_retrieval
        ), "Failed test chunks_dir_path_modeling != chunks_dir_path_retrieval"
        assert (
            embeds_dir_path_modeling != embeds_dir_path_retrieval
        ), "Failed test embeds_dir_path_modeling != embeds_dir_path_retrieval"

        files_indices_retrieval, chunks_memmaps_retrieval, _ = (
            _get_chunks_data_for_tfds_jobs(
                chunks_dir_path_retrieval, embeds_dir_path_retrieval, chunk_len
            )
        )
    else:
        assert (
            chunks_dir_path_modeling == chunks_dir_path_retrieval
        ), "Failed test chunks_dir_path_modeling == chunks_dir_path_retrieval"
        assert (
            embeds_dir_path_modeling == embeds_dir_path_retrieval
        ), "Failed test embeds_dir_path_modeling == embeds_dir_path_retrieval"

        chunks_memmaps_retrieval = chunks_memmaps_modeling
        files_indices_retrieval = files_indices_modeling

    chunks_to_docs_retrieval = _get_aggregate_docs_map(tfds_dir_path)

    return (
        chunks_memmaps_modeling,
        num_chunks_overall_modeling,
        files_indices_retrieval,
        chunks_memmaps_retrieval,
        chunks_to_docs_retrieval,
    )


def _create_num_cpus_chunk_aligned_batches(
    modeling_chunks_slice: slice, num_chunks_per_seq: int
):
    node_slice_length = modeling_chunks_slice.stop - modeling_chunks_slice.start
    divisor = NUM_CPUS_PER_NODE * num_chunks_per_seq
    # we round up here and will adjust our final batch size downwards if necessary, below
    seqs_per_node = math.ceil(node_slice_length / divisor)
    batches = list(
        range_chunked(
            modeling_chunks_slice.stop,
            seqs_per_node * num_chunks_per_seq,
            min_value=modeling_chunks_slice.start,
        )
    )
    final_batch_sz = batches[-1].stop - batches[-1].start
    adj_final_stop = _round_to_multiple(final_batch_sz, num_chunks_per_seq, "down")
    batches[-1] = slice(batches[-1].start, batches[-1].start + adj_final_stop)

    return batches


def _round_to_multiple(number: int, multiple: int, direction: str):
    assert direction in ["up", "down"], f"Invalid direction: {direction}"
    if direction == "up":
        return multiple * math.ceil(number / multiple)
    else:
        return multiple * math.floor(number / multiple)


def _convert_to_gpu_index(cpu_index: faiss.Index) -> faiss.Index:
    # convert an index to an gpu index that employs all machine gpus
    assert faiss.get_num_gpus() > 0
    co = faiss.GpuMultipleClonerOptions()
    co.shard = True
    # fp16 exact search provides very accurate results
    co.useFloat16 = True
    gpu_index = faiss.index_cpu_to_all_gpus(cpu_index, co=co)

    return gpu_index


# TODO: parameterize writing of distances
def _embeds_file_plus_index_to_cached_knns(
    index,
    file_start_index,
    embeddings_file_path: Path,
    embed_dim: int,
    k_: int,
    batch_size: int,
    knns,
    distances,
):
    assert exists_and_is_file(embeddings_file_path)
    assert distances is not None
    assert knns is not None
    assert batch_size > 0
    assert embed_dim > 0
    assert k_ > 0
    assert index

    with memmap(embeddings_file_path, np.float32, "r") as embeds_flat:
        embeddings, num_embeds = reshape_memmap_given_width(embeds_flat, embed_dim)

        for embed_slice in range_chunked(num_embeds, batch_size):
            with log(f"Calculating knns {embed_slice.start} / {num_embeds}"):
                relative_query_indices = np.arange(embed_slice.start, embed_slice.stop)
                queries = embeddings[relative_query_indices]
                # swj: search for neighbors
                neighbor_distances, absolute_neighbor_indices = index.search(
                    queries, k_
                )
                absolute_query_indices = relative_query_indices + file_start_index
                # bp()
                knns[absolute_query_indices] = absolute_neighbor_indices
                distances[absolute_query_indices] = neighbor_distances


def embeds_dir_plus_index_to_cached_knns(
    index_reference: bool,
    index_dir_path: Path,
    embeddings_dir_path: Path,
    knns_dir_path: Path,
    batch_size: int,
    embed_dim: int,
    k_: int,
    enable_gpu: bool,
):
    assert not index_reference or exists_and_is_dir(knns_dir_path)
    assert exists_and_is_dir(index_dir_path)
    assert exists_and_is_dir(embeddings_dir_path)
    assert batch_size > 0
    assert embed_dim > 0
    assert k_ > 0

    index_file_path = index_dir_path / INDEX_FILENAME
    knns_file_path = knns_dir_path / KNNS_FILENAME

    with log("Reading index and key file"):
        index = faiss.read_index(str(index_file_path))
        if enable_gpu:
            index = _convert_to_gpu_index(index)

        files_indices, _ = _get_files_indices(embeddings_dir_path)

    # TODO: we don't need index_map really if we're calculating a reference index, *except* for for this length
    #       probably we can break this dependency
    with memmap(embeddings_dir_path / MAP_FILENAME, np.uint16, "r") as index_map:
        knns_shape = (len(index_map), k_)
        with memmap(knns_file_path, np.uint32, "w+", shape=knns_shape) as knns, memmap(
            str(knns_file_path) + DIST_SUFFIX, np.float32, "w+", shape=knns_shape
        ) as distances:
            embeddings_file_paths = sorted(list(embeddings_dir_path.glob(NPY_GLOB)))
            with log(f"Processing {len(embeddings_file_paths)} embeddings files"):
                # swj: iterate over every embedding files
                for embeddings_file_path in embeddings_file_paths:
                    # bp()
                    file_start_index, _, _ = files_indices[embeddings_file_path.name]
                    _embeds_file_plus_index_to_cached_knns(
                        index,
                        file_start_index,
                        embeddings_file_path,
                        embed_dim,
                        k_,
                        batch_size,
                        knns,
                        distances,
                    )


# TODO: - most of this logic could be done when creating embeddings, not when creating the index
#       - per above, could also remove index map
def embeds_dir_to_index_swj(
    embeddings_dir_path: Path, index_dir_path: Path, index_key, embed_dim: int
):
    assert exists_and_is_dir(index_dir_path)
    assert embeddings_dir_path.exists()
    assert index_key

    index_file_path = index_dir_path / INDEX_FILENAME
    print("index_file_path: ", index_file_path)
    logging.info(
        f"Faiss version: {faiss.__version__}; number of GPUs: {faiss.get_num_gpus()}"
    )

    with log("Processing embeddings"):
        embeddings_file_paths = sorted(list(embeddings_dir_path.glob(LZ4_NPY_GLOB)))
        # embeddings_file_paths.reverse()
        embeddings_file_paths = [
            sorted(list(embeddings_dir_path.glob(LZ4_NPY_GLOB)))[0]
        ]
        for embeddings_file_path in embeddings_file_paths:
            with log(f"Processing embeddings file {embeddings_file_path.name}"), memmap(
                embeddings_file_path, np.float32, "r"
            ) as embeds_flat:
                embeddings, _ = reshape_memmap_given_width(embeds_flat, embed_dim)
                # train embeddings
                train_index_path = train_index(embeddings, index_file_path)

        # add embeddings to faiss
        index = faiss.read_index(train_index_path)
        embeddings_file_paths = sorted(list(embeddings_dir_path.glob(LZ4_NPY_GLOB)))
        with log(f"Processing {len(embeddings_file_paths)} embeddings files"):
            for embeddings_file_path in embeddings_file_paths:
                with log(
                    f"Processing embeddings file {embeddings_file_path.name}"
                ), memmap(embeddings_file_path, np.float32, "r") as embeds_flat:
                    embeddings, _ = reshape_memmap_given_width(embeds_flat, embed_dim)
                    with log("Adding embeddings to index"):
                        index.add(embeddings.astype("float32"))  # type: ignore

    with log("Writing index"):
        faiss.write_index(index, str(index_file_path))


def train_index(embeddings, index_file_path):
    dimension = 768
    ncentroids = 4096
    code_size = 64
    probe = 8
    cuda = 1
    output_path = str(index_file_path) + "/index.trained"
    if not os.path.exists(output_path):
        # Initialize faiss index
        quantizer = faiss.IndexFlatL2(dimension)

        start_index = faiss.IndexIVFPQ(quantizer, dimension, ncentroids, code_size, 8)
        start_index.nprobe = probe

        print("Training Index")
        np.random.seed(0)
        start = time.time()

        if cuda:
            # Convert to GPU index
            res = faiss.StandardGpuResources()
            co = faiss.GpuClonerOptions()
            co.useFloat16 = True
            gpu_index = faiss.index_cpu_to_gpu(res, 0, start_index, co)
            gpu_index.verbose = False
            # Train on GPU and back to CPU
            gpu_index.train(embeddings)
            start_index = faiss.index_gpu_to_cpu(gpu_index)
        else:
            # Faiss does not handle adding keys in fp16 as of writing this.
            start_index.train(embeddings)
            print("Training took {} s".format(time.time() - start))

        print("Writing index after training")
        start = time.time()
        faiss.write_index(start_index, output_path)
        print("Writing index took {} s".format(time.time() - start))
    return output_path


# TODO: - most of this logic could be done when creating embeddings, not when creating the index
#       - per above, could also remove index map
def embeds_dir_to_index(
    embeddings_dir_path: Path, index_dir_path: Path, index_key, embed_dim: int
):
    assert exists_and_is_dir(index_dir_path)
    assert embeddings_dir_path.exists()
    assert index_key

    index_file_path = index_dir_path / INDEX_FILENAME

    logging.info(
        f"Faiss version: {faiss.__version__}; number of GPUs: {faiss.get_num_gpus()}"
    )
    index = faiss.IndexFlat(EMBED_DIM, faiss.METRIC_L2)

    with log("Processing embeddings"):
        embeddings_file_paths = sorted(list(embeddings_dir_path.glob(LZ4_NPY_GLOB)))
        # embeddings_file_paths.reverse()
        with log(f"Processing {len(embeddings_file_paths)} embeddings files"):
            for embeddings_file_path in embeddings_file_paths:
                with log(
                    f"Processing embeddings file {embeddings_file_path.name}"
                ), memmap(embeddings_file_path, np.float32, "r") as embeds_flat:

                    embeddings, _ = reshape_memmap_given_width(embeds_flat, embed_dim)
                    with log("Adding embeddings to index"):
                        # bp()
                        # maybe out of memory
                        index.add(embeddings.astype("float32"))  # type: ignore
                        with log(
                            f"Writing index for {embeddings_file_path.name} in {str(index_dir_path)}"
                        ):
                            faiss.write_index(index, str(index_file_path))

    # with log('Writing index'):
    #     faiss.write_index(index, str(index_file_path))


# TODO: possibly overriding below todo, this could be done in chunk-creation
# TODO: this could be done while creating embeddings (not as a separate step afterwards)
#      we would just need to gather data back after parallel processing
#
def _create_map_and_key(embeddings_dir_path: Path, embed_dim: int):
    # bp()
    with log("Processing embeddings"):
        embeddings_file_paths = sorted(list(embeddings_dir_path.glob(NPY_GLOB)))
        # print("embeddings_file_paths: ", embeddings_file_paths)
        with log(f"Processing {len(embeddings_file_paths)} embeddings files"):
            file_end_index = 0
            file_end_index_to_filename = {}
            for embeddings_file_path in embeddings_file_paths:
                with log(
                    f"Processing embeddings file {embeddings_file_path.name}"
                ), memmap(embeddings_file_path, np.float32, "r") as embeds_flat:

                    _, num_embeds = reshape_memmap_given_width(embeds_flat, embed_dim)
                    file_end_index += num_embeds
                    file_end_index_to_filename[file_end_index] = (
                        embeddings_file_path.name
                    )

    with log("Writing map file"):
        filename_to_indices = {}
        # bp()
        with memmap(
            embeddings_dir_path / MAP_FILENAME, np.uint16, "w+", shape=(file_end_index,)
        ) as index_map:
            assert (
                len(file_end_index_to_filename) <= 65536
            ), "Exceeded max # files of 65536 (file end indices stored in uint16)"
            file_start_index = 0
            filename_index = 0
            # relies on python dictionaries being ordered (which, since 3.6/3.7 they are)
            for file_end_index, filename in file_end_index_to_filename.items():
                index_map[file_start_index:file_end_index] = filename_index
                filename_to_indices[filename] = (
                    file_start_index,
                    file_end_index,
                    filename_index,
                )
                filename_index += 1
                file_start_index = file_end_index

    with log("Writing key file"):
        with open(embeddings_dir_path / INDICES_FILENAME, "wb") as key_file:
            pkl.dump(filename_to_indices, key_file)


def _embed_chunk_batch(
    chunk_batch, model, tokenizer_info: TokenizerInfo, bert: SentenceTransformer
):

    # weijia hard code
    # chunk_batch = chunk_batch[:, :510]
    # rangehow：这段拼俩是干啥的……？给special tokens腾位置啊？
    # padded_batch = np.concatenate((chunk_batch, np.full((chunk_batch.shape[0], 2), tokenizer_info.pad_token)), axis=1)
    padded_batch = chunk_batch
    # padded_batch_ori = padded_batch.copy()

    for index, _ in enumerate(padded_batch):
        # 如果序列开头不是CLS，就集体往后挪一个位置，然后把第一个词变成CLS，这个会丢弃原始的最后一个词。
        if padded_batch[index, 0] != tokenizer_info.cls_token:
            padded_batch[index] = np.roll(padded_batch[index], 1)
            padded_batch[index, 0] = tokenizer_info.cls_token

        # 如果SEP不在，就把第一个pad给替换成sep，否则报错。
        if tokenizer_info.sep_token not in padded_batch[index]:
            pad_indices = np.where(padded_batch[index] == tokenizer_info.sep_token)
            assert len(pad_indices) == 1
            pad_indices = pad_indices[0]
            padded_batch[index, pad_indices[0]] = tokenizer_info.sep_token
    # bp()
    # print("padded_batch: ", padded_batch)
    padded_batch_torch = torch.from_numpy(padded_batch)
    # print(padded_batch_torch[padded_batch_torch!=0])
    # print("padded_batch_torch.shape[1]",padded_batch_torch.shape[1])
    assert padded_batch_torch.shape[1] <= 512

    batch_embed = bert_embed(
        padded_batch_torch,
        pad_id=tokenizer_info.pad_token,
        model_config=model,
        bert=bert,
    )
    return batch_embed


def _parallel_embed(
    tokens_file_path: Path,
    embeds_file_path: Path,
    batch_size: int,
    num_workers: int,
    worker_id: int,
    model: DictConfig,
    chunk_len: int,
    tokenizer_info,
    dataset: Dataset,
):

    # 理论上，这是一个进程的内部
    assert exists_and_is_dir(tokens_file_path)
    assert not embeds_file_path.is_dir()
    assert batch_size > 0
    assert num_workers > 0

    # assert chunk_len > 0 and chunk_len == 64
    assert model
    OmegaConf.update(model, "device", worker_id, merge=True)
    bert = get_bert(**model)
    dataset = split_dataset_by_node(dataset, worker_id, num_workers)

    def collate_fn(batch):
        return torch.tensor([item["tokens"] for item in batch])

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        collate_fn=partial(collate_fn),
    )

    emb_list = []
    
    for batch in tqdm(dataloader):
        batch_embed = bert_embed(batch, pad_id=tokenizer_info.pad_token, bert=bert)
        emb_list.append(batch_embed.detach().cpu().numpy())
    
    embeddings = np.vstack(emb_list)
    with log(f"Worker {worker_id} writing embeddings"):
        shard_filename = Path(
            str(embeds_file_path) + f"_{worker_id}_{num_workers}.npy"
        )
        np.save(shard_filename, embeddings)
    
    return len(dataset), embeddings.shape[0]
    
    with memmap(chunks_file_path, np.int32, "r+") as chunks_flat:
        num_chunks, mod = divmod(len(chunks_flat), chunk_len)
        assert num_chunks > 0 and mod == 0

        # 需要reshape是因为这个文件保存在本地似乎是平铺开的，按np.int32去解会得到一个一维的张量
        chunks = chunks_flat.reshape(num_chunks, chunk_len)

        # 找准这个进程需要处理的范围，然后在下面根据bsz去推断。
        shard_size = math.ceil(num_chunks / num_workers)
        start, end = shard_size * worker_id, shard_size * (worker_id + 1)
        shard = chunks[start:end]

        

        OmegaConf.update(model, "device", worker_id, merge=True)
        bert = get_bert(**model)
        with log(f"Worker {worker_id} processing chunks {start} to {end}"):
            for row in range(0, shard_size, batch_size):
                batch_chunk_npy = shard[row : row + batch_size]
                batch_embed = _embed_chunk_batch(
                    batch_chunk_npy, model, tokenizer_info, bert
                )
                emb_list.append(batch_embed.detach().cpu().numpy())
            embeddings = np.vstack(emb_list)

        with log(f"Worker {worker_id} writing embeddings"):
            shard_filename = Path(
                str(embeds_file_path) + f"_{worker_id}_{num_workers}.npy"
            )
            np.save(shard_filename, embeddings)

    return num_chunks, embeddings.shape[0]


def _parallel_chunks_file_to_embed_file(
    tokens_file_path: Path,
    embeds_file_path: Path,
    batch_size: int,
    model: DictConfig,
    chunk_len: int,
    parallel_cfg: DictConfig,
    tokenizer_info: TokenizerInfo,
):
    """单文件的处理程序，在这里变成多进程很合适。

    Args:
        chunks_file_path (Path): _description_
        embeds_file_path (Path): _description_
        batch_size (int): _description_
        model (DictConfig): _description_
        chunk_len (int): _description_
        embed_dim (int): _description_
        parallel_cfg (DictConfig): _description_
        tokenizer_info (TokenizerInfo): _description_
    """
    assert exists_and_is_dir(tokens_file_path)
    assert not embeds_file_path.is_dir()
    assert batch_size > 0
    # assert embed_dim > 0
    # assert chunk_len > 0 and chunk_len == 64
    assert parallel_cfg
    assert model

    # rangehow: 不允许workers的数量超过gpu数量，因为我强行要求一个worker绑一个device（如果有gpu的话）
    assert parallel_cfg.num_workers <= torch.cuda.device_count()

    dataset = datasets.load_from_disk(tokens_file_path)

    with log(f"Processing file {tokens_file_path.name}"):
        num_workers = parallel_cfg.num_workers
        processes = []

        if num_workers == 1:
            # 单进程直接执行
            with log("Running in single process mode"):
                worker = WorkerFunctor(
                    _parallel_embed,
                    tokens_file_path,
                    embeds_file_path,
                    batch_size,
                    num_workers,
                    0,  # worker_id = 0
                    model,
                    chunk_len,
                    tokenizer_info,
                    dataset,
                )
                worker()  # 直接调用，不创建新进程
                # 直接读取并重命名唯一的分片文件
                shard_filename = Path(str(embeds_file_path) + "_0_1.npy")
                if not shard_filename.exists():
                    raise RuntimeError(f"Output file {shard_filename} not found")
                shard_filename.rename(embeds_file_path)

        else:
            with log(f"Starting {num_workers} processes"):
                # 启动进程
                for worker_id in range(num_workers):
                    worker = WorkerFunctor(
                        _parallel_embed,
                        tokens_file_path,
                        embeds_file_path,
                        batch_size,
                        num_workers,
                        worker_id,
                        model,
                        chunk_len,
                        tokenizer_info,
                        dataset,
                    )
                    p = mp.Process(target=worker)
                    processes.append(p)
                    p.start()

            try:
                with log("Waiting for processes to complete"):
                    # 等待所有进程完成
                    for p in processes:
                        p.join()

                with log("Validating results"):
                    # 验证每个分片文件是否存在并获取总处理数量
                    processed_chunks = 0
                    for worker_id in range(num_workers):
                        shard_filename = Path(
                            str(embeds_file_path) + f"_{worker_id}_{num_workers}.npy"
                        )
                        if not shard_filename.exists():
                            raise RuntimeError(f"Shard file {shard_filename} not found")
                        processed_chunks += len(np.load(shard_filename))
                
                with log(f"Merging shards into {embeds_file_path}"):
                    with memmap(
                        embeds_file_path,
                        np.float32,
                        "w+",
                        shape=(processed_chunks, tokenizer_info.embed_dim),
                    ) as embeds:
                        embeds_index = 0
                        for worker_id in range(num_workers):
                            shard_filename = Path(
                                str(embeds_file_path)
                                + f"_{worker_id}_{num_workers}.npy"
                            )
                            embeds_shard = np.load(shard_filename)
                            embeds[embeds_index : embeds_index + len(embeds_shard)] = (
                                embeds_shard
                            )
                            embeds_index += len(embeds_shard)
                            shard_filename.unlink()

            except Exception as e:
                logging.critical(
                    f"Error while processing/merging results in file {tokens_file_path.name}:\n{e}"
                )


def parallel_chunks_files_to_embeds_files(
    tokens_dir_path: Path,
    embeds_dir_path: Path,
    batch_size: int,
    model: DictConfig,
    chunk_len: int,
    parallel_cfg: DictConfig,
    tokenizer_info: TokenizerInfo,
):
    # assert exists_and_is_dir(tokens_dir_path)
    assert exists_and_is_dir(embeds_dir_path)
    assert batch_size > 0

    assert parallel_cfg
    assert model
    
    # tokens_file_paths = sorted(
    #     [item for item in tokens_dir_path.iterdir() if item.is_dir()]
    # )
    for tokens_file_path in tokens_dir_path:
        if os.path.exists(embeds_dir_path / tokens_file_path.name.rsplit(".")[0]):
            print(f"skip processing {embeds_dir_path / tokens_file_path.name}")
            tokens_dir_path.remove(tokens_file_path)

    with log(f"Processing {len(tokens_dir_path)} chunks files"):
        for tokens_file_path in tokens_dir_path:
            _parallel_chunks_file_to_embed_file(
                tokens_file_path,
                embeds_dir_path / tokens_file_path.name,
                batch_size,
                model,
                chunk_len,
                parallel_cfg,
                tokenizer_info,
            )


def _chunks_file_to_embed_file(
    chunks_file_path: Path,
    embeds_file_path: Path,
    model: DictConfig,
    chunk_len: int,
    batch_size: int,
    embed_dim: int,
    tokenizer_info: TokenizerInfo,
):
    assert exists_and_is_file(chunks_file_path)
    assert not embeds_file_path.is_dir()
    assert batch_size > 0
    assert model

    bert = get_bert(**model)
    with log(f"Processing {chunks_file_path.name}"):
        with memmap(chunks_file_path, np.int32, "r") as chunks_flat:
            # bp()
            chunks, num_chunks = reshape_memmap_given_width(chunks_flat, chunk_len)

            with memmap(
                embeds_file_path,
                np.float32,
                "w+",
                shape=(num_chunks, tokenizer_info.embed_dim),
            ) as embeds:
                for slice_ in range_chunked(num_chunks, batch_size):
                    chunk_batch = chunks[slice_]
                    embed_batch = _embed_chunk_batch(
                        chunk_batch, model, tokenizer_info, bert=bert
                    )
                    embeds[slice_] = embed_batch.cpu()


def chunks_files_to_embeds_files(
    chunks_dir_path: Path,
    embeds_dir_path: Path,
    model: DictConfig,
    chunk_len: int,
    batch_size: int,
    tokenizer_info: TokenizerInfo,
):
    assert exists_and_is_dir(chunks_dir_path)
    assert exists_and_is_dir(embeds_dir_path)
    assert batch_size > 0
    # assert embed_dim > 0
    # assert chunk_len > 0 and chunk_len == 64
    assert model

    chunks_file_paths = sorted(list(chunks_dir_path.glob(NPY_GLOB)))
    # bp()

    with log(f"Processing {len(chunks_file_paths)} chunks files"):
        for chunks_file_path in chunks_file_paths:
            _chunks_file_to_embed_file(
                chunks_file_path,
                embeds_dir_path / chunks_file_path.name,
                model,
                chunk_len,
                batch_size,
                tokenizer_info.embed_dim,
                tokenizer_info,
            )

    _create_map_and_key(embeds_dir_path, tokenizer_info.embed_dim)


def _create_chunks_and_map(
    tokens_file_path: Path,
    chunks_file_path: Path,
    chunk_len: int,
    total_chunks: int,
    tokenizer_info: TokenizerInfo,
):
    assert exists_and_is_file(tokens_file_path)
    assert not chunks_file_path.is_dir()
    print("tokens_file_path: ", tokens_file_path, "total_chunks: ", total_chunks)
    # assert total_chunks > 0
    # assert chunk_len > 0 and chunk_len == 64

    with read_jsonl_file(tokens_file_path) as tokens_reader, memmap(
        chunks_file_path, np.int32, "w+", shape=(total_chunks, chunk_len)
    ) as chunks, memmap(
        str(chunks_file_path) + MAP_SUFFIX, np.int32, "w+", shape=(total_chunks,)
    ) as chunks_map:
        print("tokens_file_path: ", tokens_file_path)
        chunk_index = 0
        # each line in tokens file maps to equivalent line in documents file and so line index == doc_index
        """
        for doc_index, line in enumerate(tokens_reader)
            print(line)
            if doc_index > 3:
                break
        """
        for doc_index, line in enumerate(tokens_reader):
            # print(line["doc_id"])
            # bp()
            # assert doc_index == line["doc_id"]
            # pad to end of chunk
            tokens = line["tokens"]
            # bp()
            # if len(tokens) <= 2:
            #     # TODO this is a bit duplicative of "discarding" check/log below - remove it when confident
            #     logging.warn(f'No tokens found while processing doc {doc_index} in file {tokens_file_path.name} - not discarding')
            # bp()

            # [rangehow]: 这个地方要求一个文档的tokens在chunk_len之内是因为分词时已经做了强制截断了。
            div, mod = divmod(len(tokens), chunk_len)
            # [rangehow]: 思考一下……，这里其实很麻烦，pad的目的是要齐，但是这个补到maxlen的chunk_len可能不如doc长

            if mod != 0:
                tokens.extend([tokenizer_info.pad_token] * (chunk_len - mod))
                div += 1

            # [rangehow]: 解除一个文档只能映射到一个chunk上的限制
            if div != 1:
                bp()
            assert div == 1, "目前的截断设置下，一个文档不能对应多个chunk"
            for doc_chunk_index in range(div):
                tokens_slice = slice(
                    doc_chunk_index * chunk_len, (doc_chunk_index + 1) * chunk_len
                )
                try:
                    chunks[chunk_index] = tokens[tokens_slice]
                except Exception as e:
                    print(e)
                    bp()
                chunks_map[chunk_index] = doc_index
                chunk_index += 1
            # print("chunk_index: ", chunk_index, "doc_id: ", line["doc_id"])
            # assert chunk_index == line["doc_id"]+1
    assert chunk_index == total_chunks, "目前的截断设置下，一个文档不能对应多个chunk"


def _tokens_file_to_chunks_files(
    tokens_file_path: Path, chunks_file_path: Path, chunk_len: int, tokenizer_info
):
    assert exists_and_is_file(tokens_file_path)
    assert not chunks_file_path.is_dir()
    # assert chunk_len > 0 and chunk_len == 512
    init_logging()

    with log(f"Processing {tokens_file_path.name}"):
        with log("Calculating total number of chunks"):
            total_chunks = 0
            with read_jsonl_file(tokens_file_path) as tokens_reader:
                for i, line in tqdm(enumerate(tokens_reader)):
                    div, mod = divmod(len(line["tokens"]), chunk_len)
                    if (div != 0) and (div != 1):
                        bp()
                    assert div == 0 or div == 1
                    total_chunks += div if mod == 0 else div + 1

        with log("Creating chunks and chunks map"):
            _create_chunks_and_map(
                tokens_file_path,
                chunks_file_path,
                chunk_len,
                total_chunks,
                tokenizer_info,
            )


def tokens_files_to_chunks_files(
    tokens_dir_path: Path, chunks_dir_path: Path, chunk_len: int, tokenizer_info
):

    assert exists_and_is_dir(tokens_dir_path) and exists_and_is_dir(chunks_dir_path)
    # assert chunk_len > 0 and chunk_len == 512

    # swj change
    tokens_file_paths = sorted(list(tokens_dir_path.glob(LZ4_GLOB)))

    # tokens_file_paths = sorted(list(tokens_dir_path.glob("jsonl")))

    for tokens_file_path in tokens_file_paths:
        if os.path.exists(str(chunks_dir_path / tokens_file_path.name) + NPY_SUFFIX):
            print(
                f"skip processing {str(chunks_dir_path / tokens_file_path.name) + NPY_SUFFIX}"
            )
            tokens_file_paths.remove(tokens_file_path)

    with log(f"Processing {len(tokens_file_paths)} tokens files"):
        # for debugging:

        if len(tokens_file_paths) == 1:
            _tokens_file_to_chunks_files(
                tokens_file_paths[0],
                Path(str(chunks_dir_path / tokens_file_paths[0].name) + NPY_SUFFIX),
                chunk_len,
                tokenizer_info,
            )
        else:
            _parallel()(
                delayed(_tokens_file_to_chunks_files)(
                    tokens_file_path,
                    Path(str(chunks_dir_path / tokens_file_path.name) + NPY_SUFFIX),
                    chunk_len,
                    tokenizer_info,
                )
                for tokens_file_path in track(tokens_file_paths)
            )


# TODO: this can be slow if the input files to jsonl_dir_to_docs_text_list_file were large (because only ||izes at file level)
#       could think about splitting files or doing something more sophisticated, if this becomes a problem
def _docs_files_to_tokens_file(
    docs_file_path, tokens_file_path: Path, tokenizer_cfg: DictConfig
):
    assert exists_and_is_file(docs_file_path)
    assert not tokens_file_path.is_dir()
    assert tokenizer_cfg
    init_logging()

    with log(f"Tokenizing doc {docs_file_path.name}"):
        # , streaming=True
        dataset = load_dataset("json", data_files=str(docs_file_path))["train"]

        trick_instance = dataset.take(1)
        if "##@@B" in trick_instance:
            dataset = dataset.map(
                partial(_preprocess_dataset),
                batched=True,
                batch_size=10000,
                num_proc=multiprocessing.cpu_count(),
            )
        dataset = dataset.map(
            partial(_tokenize_dataset, tokenizer_cfg=tokenizer_cfg),
            batched=True,
            num_proc=multiprocessing.cpu_count(),
        )
        dataset = dataset.select_columns(["tokens"])

        dataset.save_to_disk(
            tokens_file_path.with_suffix(""), num_proc=multiprocessing.cpu_count()
        )

        # with write_jsonl_file(tokens_file_path) as tokens_writer:
        #     for doc_index, doc in tqdm(
        #         enumerate(dataset),
        #     ):
        #         # tokens = _tokenize_doc(doc['text'], tokenizer_cfg)[:MAX_TOKENS]
        #         # [rangehow]: 移除最大tokens的限制，没必要在这里做，这个把sep都给去掉了，简直是bug。
        #         # tokens = _tokenize_doc(doc['text'], tokenizer_cfg)
        #         # swj change
        #         tokens_writer.write({"tokens": doc["tokens"]})


def docs_files_to_tokens_files(
    docs_dir_path: Path, tokens_dir_path: Path, tokenizer_cfg: DictConfig
):
    assert docs_dir_path.exists() and docs_dir_path.is_dir()
    assert tokens_dir_path.exists() and tokens_dir_path.is_dir()
    assert tokenizer_cfg is not None

    # swj
    # docs_file_paths = sorted(list(docs_dir_path.glob("*.jsonl")))

    docs_file_paths = sorted(list(docs_dir_path.glob("*.jsonl")))

    # 跳过已经处理好的文件

    for docs_file_path in docs_file_paths:
        if os.path.exists(tokens_dir_path / docs_file_path.name.rsplit(".")[0]):
            print(f"skip processing {tokens_dir_path / docs_file_path.name}")
            docs_file_paths.remove(docs_file_path)

    # docs_file_paths = sorted(list(docs_dir_path.glob(LZ4_GLOB)))
    with log(f"Tokenizing {len(docs_file_paths)} documents"):
        # debugging: _docs_files_to_tokens_file(docs_file_paths[0], tokens_dir_path / docs_file_paths[0].name, tokenizer_cfg)

        if len(docs_file_paths) == 1:
            _docs_files_to_tokens_file(
                docs_file_paths[0],
                tokens_dir_path / docs_file_paths[0].name,
                tokenizer_cfg,
            )
        else:
            # dataset = load_dataset("json",data_files=docs_file_paths, streaming=True)["train"]
            # dataset = dataset.map(_tokenize_dataset,batched=True)

            # [rangehow]: 这个函数写的蠢到姥姥家了，受不了一点。在文件之间开启进程，在文件内使用单进程，甚至还是串行tokenize。一个文件稍微长点都得编1896天。
            # [rangehow]: 现在已经改成batch tokenize了，如果在外面整体的形成datasets，倒是可以整体多进程，只是文件名不好处理，如果不影响速度就不管这里。
            _parallel()(
                delayed(_docs_files_to_tokens_file)(
                    docs_file_path, tokens_dir_path / docs_file_path.name, tokenizer_cfg
                )
                for docs_file_path in track(docs_file_paths)
            )


def _copy_and_compress(source_file_path: Path, target_file_path: Path):
    assert source_file_path.exists() and source_file_path.is_file()
    assert not target_file_path.is_dir()
    doc_id = 0
    with log(f"Processing file {source_file_path.name}"):
        with jsonlines.open(source_file_path, "r") as source_reader, write_jsonl_file(
            target_file_path
        ) as target_writer:
            for line in tqdm(source_reader):
                line["doc_id"] = doc_id
                doc_id += 1
                target_writer.write(line)


# swj
def _copy_and_compress_new(source_file_path: Path, target_file_path: Path, doc_id: int):
    assert source_file_path.exists() and source_file_path.is_file()
    assert not target_file_path.is_dir()
    with log(f"Processing file {source_file_path.name}"):
        with read_jsonl_file_no_compress(
            source_file_path
        ) as source_reader, write_jsonl_file(target_file_path) as target_writer:
            for line in tqdm(source_reader):
                if "content" in line:
                    line["text"] = line["content"]
                    line.pop("content")
                if len(line["text"].split()[:30]) < 5:
                    # bp()
                    print("too short, skip!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                    continue
                line["doc_id"] = doc_id
                doc_id += 1
                target_writer.write(line)
    return doc_id


def import_docs(source_path: Path, docs_dir_path: Path, globs):
    assert exists_and_is_dir(source_path) and exists_and_is_dir(docs_dir_path)
    assert globs

    docs_file_paths_unsorted = []
    globs_list = globs.split(GLOB_SEPARATOR)
    for glob in globs_list:
        docs_file_paths_unsorted.extend(list(source_path.glob(glob)))

    docs_file_paths = sorted(docs_file_paths_unsorted)
    with log(f"Processing {len(docs_file_paths)} files"):
        # debugging: _copy_and_compress(docs_file_paths[0], Path(str(docs_dir_path / docs_file_paths[0].name) + LZ4_SUFFIX))
        # swj: no multiprocess
        doc_id = 0
        for d in tqdm(docs_file_paths):
            # _copy_and_compress(d, Path(str(docs_dir_path / d.name) + LZ4_SUFFIX))
            doc_id = _copy_and_compress_new(
                d, Path(str(docs_dir_path / d.name) + LZ4_SUFFIX), doc_id
            )
            print(f"finish one doc: {d} with doc_id: {doc_id}")
        # _copy_and_compress(docs_file_paths[0], Path(str(docs_dir_path / docs_file_paths[0].name) + LZ4_SUFFIX))
        # _parallel()(delayed(_copy_and_compress)(docs_file_path, Path(str(docs_dir_path / docs_file_path.name) + LZ4_SUFFIX))
        # for docs_file_path in track(docs_file_paths))
