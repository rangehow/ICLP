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

random.seed(0)

def jaccard_similarity(set1, set2):
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    return len(intersection) / len(union)

def generate_ngrams(text, n):
    ngrams = set()
    for i in range(len(text) - n + 1):
        ngram = text[i:i + n]
        ngrams.add(ngram)
    return ngrams

def ngram_similarity(doc1, doc2, n):
    ngrams_doc1 = generate_ngrams(doc1, n)
    ngrams_doc2 = generate_ngrams(doc2, n)
    return jaccard_similarity(ngrams_doc1, ngrams_doc2)

class sort_class():
    def __init__(self, output_file, context_len, text_key, text_file, knn_file):
        self.text_file = text_file
        self.knn_file = knn_file
        
        with open(self.text_file, 'r') as f:
            self.num_docs = sum(1 for _ in f)
        
        self.seen_docs = set()
        self.unseen_docs = set(range(self.num_docs))
        print(f"num docs: {self.num_docs}")

        self.doc_sim_threshold = 0.85
        self.n = 3
        self.context_len = context_len
        self.output_file = output_file
        self.text_key = text_key
        
        self.cluster_size = 21

        self.cur_k = None
        self.filter_docs = []
        
        self.cluster2docs = defaultdict(list)
        self.doc2cluster = {}

        self.knns = np.load(self.knn_file, mmap_mode="r")

    def sort(self):
        cluster_id = 0
        cur_cluster_len = 0

        self.cur_k = self.unseen_docs.pop()
        self.cluster2docs[cluster_id].append(self.cur_k)
        self.seen_docs.add(self.cur_k)
        
        with tqdm(total=self.num_docs-1) as pbar:
            while self.unseen_docs:
                knn = self.knns[self.cur_k, :]
                first_doc = self.output_first_doc_knn(knn)

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

        pickle.dump(self.cluster2docs, open(f"{self.output_file}/cluster2docs.pk", "wb"))
        pickle.dump(self.doc2cluster, open(f"{self.output_file}/doc2cluster.pk", "wb"))
    
    def output_first_doc_knn(self, knn):
        for k in knn[1:10]:
            if k not in self.seen_docs:
                return k
        return None

    def write_docs(self):
        sort_doc = self.cluster2list()
        output_folder = f"{self.output_file}/data"
        Path(output_folder).mkdir(parents=True, exist_ok=True)
        
        num_processes = 32
        chunks = self.divide_into_chunks(sort_doc, num_processes)
        
        args_list = [(chunk, i) for i, chunk in enumerate(chunks)]

        print(f"data ready: ", len(args_list))
        with multiprocessing.Pool(processes=num_processes) as pool:
            for _ in tqdm(pool.imap(self.write_docs_wrapper, args_list), total=len(args_list)):
                pass

    def divide_into_chunks(self, lst, n):
        batch_size = len(lst) // n
        for i in range(0, len(lst), batch_size):
            yield lst[i:i + batch_size]

    def write_docs_wrapper(self, args):
        return self.write_docs_single(*args)

    def write_docs_single(self, sort_doc_chunk, file_index):
        output_folder = f"{self.output_file}/data"
        prev_doc = None
        filter_docs = []
        
        with open(self.text_file, 'r') as text_file:
            docs = text_file.readlines()
        
        with open(f"{output_folder}/train_{file_index}.jsonl", "w") as f:
            for doc_id in tqdm(sort_doc_chunk):
                doc = json.loads(docs[doc_id])
                if prev_doc is not None:
                    try:
                        doc_sim = ngram_similarity(doc[self.text_key][:100], prev_doc[self.text_key][:100], self.n)
                    except:
                        print("None doc")
                        print(doc)
                        filter_docs.append(self.cur_k)
                        continue
                    if doc_sim > self.doc_sim_threshold:
                        filter_docs.append(self.cur_k)
                        continue
                f.write(json.dumps(doc) + "\n")
                prev_doc = doc
        print(f"filter docs: {len(filter_docs)}")

    def cluster2list(self):
        self.cluster2docs = pickle.load(open(f"{self.output_file}/cluster2docs.pk", "rb"))
        sort_doc = []
        for cluster_id, docs in tqdm(self.cluster2docs.items()):
            sort_doc.extend(docs)
        return sort_doc

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
    sort_member.write_docs()