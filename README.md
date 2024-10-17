# in-context-pretraining

## Installation

Add faiss as a submodule when cloning the repo (we require the [faiss OIVFBBS code](https://github.com/facebookresearch/faiss/tree/main/demos/offline_ivf]) that is not included in the faiss conda package in the demos folder):

```
git clone https://github.com/swj0419/in-context-pretraining.git
cd ~/in-context-pretraining
git add submodule https://github.com/facebookresearch/faiss.git
```

Set up your environment using the following commands:

```bash
conda create -n iclm python=3.10
conda activate iclm
conda install -c pytorch/label/nightly -c nvidia faiss-gpu=1.7.4
conda install numpy==1.26.0
pip install -r requirements.txt
```

## Sort Pretraining Data using In-Context Pretraining

We provide an example corpus in `data/b3g` to demonstrate our pipeline. The corpus is in jsonl format (`data/b3g/chunk.jsonl`), with each line representing one document. To use your own corpus, place it under `data` and update the data directory in the relevant configuration file.

### Retrieve Neighbor Documents

#### Encode Documents into Embeddings

Tokenize documents and generate embeddings for each document in the corpus:

```bash
cd build_embeds
python retro_z_data.py --config-name example_config
```

- Modify `source_path` in `example_config` to specify the data directory (default is `data/b3g`).
- Output embedding directory is set in `base_path`.
- The program submits jobs using the slurm system. Configure slurm settings in lines 30-39 of `example_config`.

#### Efficient kNN search

先安装milvus

```bash
pip install -U pymilvus
```

然后

```bash
cd build_index/lite
```

打开`index.py` ,修改在文件开头的超参数，需要修改的通常只有EMBEDDING_DIM和numpy_mmap_file_dir，前者和上一步保持统一，后者指向上一步最后产生的embedding文件就好。

建立索引

```bash
python index.py
```

搜索最近邻

```bash
python search.py
```

这一步之后会在lite/下产生一个nearest_neighbors_ids.npy，其中每行是top_k个最近邻的ID

### Sort documents based on kNNs

knn file指向上一步生成的近邻npy，text file指向最开始的纯文本文件即可

```bash
cd sort/
python sort.py --knn_file build_index/lite/nearest_neighbors_ids.npy --text_file data/fineweb_edu/fineweb_edu_500.jsonl
```
python sort.py --knn_file /mnt/rangehow/in-context-pretraining/output/embed/fineweb_edu_500/tokenizer-BAAI_bge_large_en_v1.5/seq_len-5120/chunk_len-512/model-BAAI_bge_large_en_v1.5/nearest_neighbors.npy --text_file ../data/fineweb_edu/fineweb_edu_500.jsonl



🚨**Note**🚨: After completing the sorting process, you will have a sorted `jsonl` file. This file is organized in such a way that documents which are closely related, as determined by the kNN results, are grouped together. When your pretraining code reads this file line by line, it will encounter related documents not only within the same context but also between adjacent contexts. However, it is crucial to maintain document similarity only within the same input context but **not across adjacent contexts**. Your pretraining code might require additional preprocessing to ensure **diversity** between adjacent contexts.