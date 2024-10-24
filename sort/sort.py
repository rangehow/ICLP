import sys
from tqdm import tqdm
import argparse
import pickle
from collections import OrderedDict, defaultdict
import statistics
from pathlib import Path
import random
from multiprocessing import Pool
import multiprocessing
import numpy as np
import time
import json
import os
from ipdb import set_trace as bp
from check_match import data_stats
from rich.progress import track

random.seed(0)


def jaccard_similarity(set1, set2):
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    return len(intersection) / len(union)


def generate_ngrams(text, n):
    ngrams = set()
    for i in range(len(text) - n + 1):
        ngram = text[i : i + n]
        ngrams.add(ngram)
    return ngrams


def ngram_similarity(doc1, doc2, n):
    ngrams_doc1 = generate_ngrams(doc1, n)
    ngrams_doc2 = generate_ngrams(doc2, n)
    return jaccard_similarity(ngrams_doc1, ngrams_doc2)


class sort_class:
    def __init__(self, output_file, context_len, text_key, text_file, knn_file):
        self.text_file = text_file
        self.knn_file = knn_file

        with open(self.text_file, "r") as f:
            self.num_docs = sum(1 for _ in f)

        self.seen_docs = set()
        self.unseen_docs = set(range(self.num_docs))
        print(f"num docs: {self.num_docs}")

        # 拼接文档的时候还会根据相似度滤除一下
        self.doc_sim_threshold = 0.85
        self.n = 3  # n-gram
        self.context_len = context_len
        self.output_file = output_file
        self.text_key = text_key

        self.cluster_size = 21

        self.cur_k = None
        self.filter_docs = []

        self.cluster2docs = defaultdict(list)
        self.doc2cluster = {}

        self.knns = np.load(self.knn_file, mmap_mode="r")

    def check_cluster_sizes(self):
        oversized_clusters = []
        for cluster_id, docs in self.cluster2docs.items():
            if len(docs) > self.cluster_size:
                oversized_clusters.append(cluster_id)

        if oversized_clusters:
            print(f"以下簇的大小超过了限制 {self.cluster_size}:")
            for cluster_id in oversized_clusters:
                print(f"簇 {cluster_id}: {len(self.cluster2docs[cluster_id])} 个文档")
        else:
            print(f"所有簇的大小都不超过限制 {self.cluster_size}")

    def sort(self):

        cluster_id = 0
        # 小心这里，不算1的话第一个chunk会超出cluster_size
        cur_cluster_len = 1

        self.cur_k = self.unseen_docs.pop()
        self.cluster2docs[cluster_id].append(self.cur_k)
        self.doc2cluster[self.cur_k] = cluster_id
        self.seen_docs.add(self.cur_k)

        with tqdm(total=self.num_docs - 1) as pbar:
            while self.unseen_docs:
                knn = self.knns[self.cur_k, :]

                first_doc = self.output_first_doc_knn(knn)
                # if knn[0]!=self.cur_k: print(f"{self.cur_k},{knn[0]}")
                if (first_doc is None) or (cur_cluster_len >= self.cluster_size):

                    self.cur_k = self.unseen_docs.pop()
                    cluster_id += 1
                    cur_cluster_len = 0
                else:
                    self.cur_k = first_doc
                    self.unseen_docs.remove(self.cur_k)

                self.cluster2docs[cluster_id].append(self.cur_k)
                self.doc2cluster[self.cur_k] = cluster_id
                cur_cluster_len += 1
                self.seen_docs.add(self.cur_k)
                pbar.update(1)
        print("合并前簇的数量：", len(self.cluster2docs))
        # self.check_cluster_sizes()
        data_stats(self.cluster2docs)
        pickle.dump(
            self.cluster2docs, open(f"{self.output_file}/cluster2docs.pk", "wb")
        )
        pickle.dump(self.doc2cluster, open(f"{self.output_file}/doc2cluster.pk", "wb"))

    def output_first_doc_knn_not_in_the_cluster(self, knn, cluster_id):
        for k in knn:
            if k != -1:
                k_cluster = self.doc2cluster[k]

                while (
                    k_cluster != cluster_id
                    and len(self.cluster2docs[k_cluster]) < self.cluster_size * 4
                ):
                    return k, k_cluster

        return None, None

    def check_all_docs_assigned(self, cluster2docs):
        all_docs = set()
        for cluster in cluster2docs.values():
            all_docs.update(cluster)

        all_docs = sorted(all_docs)

        if len(all_docs) != 100000:
            print(f"警告：文档总数不是 100000，而是 {len(all_docs)}")

        for i, doc in enumerate(all_docs):
            if i != doc:
                print(f"缺失的文档索引：从 {i} 到 {doc - 1}")
                return False

        print("所有文档索引都已分配")
        return True

    def merge(self):
        # self.cluster2docs = pickle_load(f"{self.output_file}/cluster2docs.pk")
        # self.doc2cluster = pickle_load(f"{self.output_file}/doc2cluster.pk")
        # data_stats(self.cluster2docs)

        merged_clusters_num = 0

        # 因为要在迭代中删除self.cluster2docs的东西所以得迭代copy
        for cluster, cluster_docs in tqdm(self.cluster2docs.copy().items()):
            if len(cluster_docs) < self.cluster_size:
                merged_clusters_num += 1
                # print(merged_clusters_num)
                # 如果发现了一个小于预设长度的簇，就拆散这个簇，找到里面每个元素
                for doc in cluster_docs:
                    # bp()
                    knn = self.knns[doc, :]

                    top1k, top1k_cluster = self.output_first_doc_knn_not_in_the_cluster(
                        knn, cluster
                    )
                    # bp()

                    k_cluster_docs = self.cluster2docs[top1k_cluster]
                    # bp()
                    # add k to doc
                    # k_cluster_docs.append(k)
                    k_cluster_docs.insert(k_cluster_docs.index(top1k), doc)

                    # update the cluster
                    self.cluster2docs[top1k_cluster] = k_cluster_docs
                    self.doc2cluster[doc] = top1k_cluster
                del self.cluster2docs[cluster]
        print(
            f"merged_clusters_num:{merged_clusters_num},合并后簇的数量:{len(self.cluster2docs)}"
        )
        data_stats(self.cluster2docs)

        # 完备性检查
        self.check_all_docs_assigned(self.cluster2docs)
        pickle.dump(
            self.cluster2docs, open(f"{self.output_file}/cluster2docs_merge.pk", "wb")
        )

    def output_first_doc_knn(self, knn):
        # 索引可能会使得这个并不总是在knn[0]召回自身，所以没必要knn[1:]
        for k in knn:
            if k not in self.seen_docs and k != -1:
                return k
        return None

    def write_docs(self):
        sort_doc = self.cluster2list()

        output_folder = f"{self.output_file}/data"
        Path(output_folder).mkdir(parents=True, exist_ok=True)

        num_processes = multiprocessing.cpu_count()
        chunks, chunk_lengths = self.divide_into_chunks(sort_doc, num_processes)

        # 对chunk_len求前缀和，计算每个任务的起始索引
        start_idx = np.cumsum([0] + chunk_lengths[:-1])  # 添加0作为第一个起始索引

        # 创建参数列表
        args_list = [(chunk, start_idx[i]) for i, chunk in enumerate(chunks)]

        print(f"data ready: ", len(args_list))
        with multiprocessing.Pool(processes=num_processes) as pool:
            for _ in track(
                pool.imap_unordered(self.write_docs_wrapper, args_list),
                total=len(args_list),
            ):
                pass

    def divide_into_chunks(self, lst, n):
        n = min(n, len(lst))  # 确保 n 不大于列表长度
        chunks = []
        chunk_lengths = []
        for i in range(n):
            start = i * len(lst) // n
            end = (i + 1) * len(lst) // n
            chunks.append(lst[start:end])
            chunk_lengths.append(end - start)
        return chunks, chunk_lengths

    def write_docs_wrapper(self, args):
        return self.write_docs_single(*args)

    def write_docs_single(self, sort_doc_chunks, file_index):
        output_folder = f"{self.output_file}/data"
        prev_doc = None
        filter_docs = []

        with open(self.text_file, "r") as text_file:
            docs = text_file.readlines()

        for chunk_index, sort_doc_chunk in enumerate(sort_doc_chunks):
            output_data = []

            for doc_id in sort_doc_chunk:
                doc = docs[doc_id]
                output_data.append(json.dumps(doc, separators=(",", ":")))

            # 为每个chunk写入单独的文件
            current_file_index = file_index + chunk_index
            with open(f"{output_folder}/train_{current_file_index}.jsonl", "w") as f:
                f.write("\n".join(output_data))

    def cluster2list(self):
        self.cluster2docs = pickle.load(
            open(f"{self.output_file}/cluster2docs_merge.pk", "rb")
        )
        sort_doc = []
        for cluster_id, docs in tqdm(self.cluster2docs.items()):
            sort_doc.append(docs)
        return sort_doc

    # def cluster2list(self):
    #     self.cluster2docs = pickle.load(open(f"{self.output_file}/cluster2docs_merge.pk", "rb"))
    #     sort_doc = []
    #     for cluster_id, docs in tqdm(self.cluster2docs.items()):
    #         sort_doc.extend(docs)
    #     return sort_doc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default=".")
    parser.add_argument("--knn_file", type=str, required=True)
    parser.add_argument("--text_file", type=str, required=True)

    args = parser.parse_args()

    output_dir = args.output_dir
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    sort_member = sort_class(output_dir, 4096, "text", args.text_file, args.knn_file)

    sort_member.sort()
    sort_member.merge()
    sort_member.write_docs()
