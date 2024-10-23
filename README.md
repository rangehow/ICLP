# in-context-pretraining



## 更新日志
### 2024/10/23
- 完全移除chunk，修改大量多进程逻辑。


### 2024/10/22

- 去除sort的检查ngram逻辑，更细致的多进程处理
- 提高了15x的tokenize速度，我嘞个豆啊！dataset本身就是mmap文件，没必要流式。





### 2024/10/21

- 在build_embeds这一步中加入了完备性检查参数enable_completeness_check，可以在配置文件里设置，建议开启。
- 在面对北哥那面的数据集时，抽出一条判断需不需要做处理，见retro_z_data_xforms.py：
    ```python 
        trick_instance= dataset.take(1)
        if "##@@B" in trick_instance:
            dataset = dataset.map(
                partial(_preprocess_dataset), batched=True, batch_size=10000
            )
    ``` 
    具体的处理逻辑可以在_preprocess_dataset调整。
- 第一步build_embeds自动跳过已经处理好的文件，即使generate设置为True也会检测是否已经生成过完全相同配置的各种文件
- 在build index尝试使用gpu搜索，大幅度提速，只在10w级试验过不会爆显存。
- 修复了sort.py的大量不正确行为。包括merge，以及保存逻辑。
- 边角bug，不完备的索引类型在极小样本比如500上有可能会返回-1，在sort内加入考虑了这种情况。同时小于10万的都采用完备搜索。
- 在第二第三步许多的完备检测代码，暂不赘述。
#### NOTE
- 检查出一些向量的最近邻即使在确定的索引类型下也不是自己，这是因为潜在的原始文档重复/或前512个token重复问题。
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