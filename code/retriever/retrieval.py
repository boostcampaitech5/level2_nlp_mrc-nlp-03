import json
import os
import pickle
import time
import torch
import faiss
import numpy as np
import pandas as pd
import logging
from torch.utils.data import DataLoader
from contextlib import contextmanager
from abc import abstractmethod
from typing import List, Optional, Tuple, Union
from transformers import AutoTokenizer
from datasets import Dataset, concatenate_datasets, load_from_disk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from tqdm.auto import tqdm

try:
    from retriever.DPR_model import DPR
    from retriever.utils_retriever import Preprocessor
except ImportError:
    from DPR_model import DPR
    from utils_retriever import Preprocessor

from rank_bm25 import BM25Okapi, BM25Plus


@contextmanager
def timer(name):
    """with문과 함께 쓰여, with문 내의 코드 실행 시간 측정

    Args:
        name (str): _description_
    """
    t0 = time.time()  # with문이 시작되기 전에 실행
    yield  # with문 내부 코드 실행
    print(f"[{name}] done in {time.time() - t0:.3f} s")  # with문 끝난 뒤에 실행


class _BaseRetriever:
    def __init__(
        self,
        tokenize_fn,
        preprocessor=Preprocessor,
        data_path: Optional[str] = "./data/",
        context_path: Optional[str] = "wikipedia_documents.json",
    ) -> None:
        """
        Arguments:
            tokenize_fn:
                기본 text를 tokenize해주는 함수입니다.
                아래와 같은 함수들을 사용할 수 있습니다.
                - lambda x: x.split(' ')
                - Huggingface Tokenizer
                - konlpy.tag의 Mecab

            data_path:
                데이터가 보관되어 있는 경로입니다.

            context_path:
                Passage들이 묶여있는 파일명입니다.

            data_path/context_path가 존재해야합니다.

        Summary:
            Passage 파일을 불러오고 passage embedding을 계산합니다.
        """

        self.tokenize_fn = tokenize_fn
        self.preprocessor = preprocessor
        self.data_path = data_path
        with open(os.path.join(data_path, context_path), "r", encoding="utf-8") as f:
            wiki = json.load(f)
        self.wiki_df = pd.DataFrame(wiki).T
        self.wiki_df["preprocessed_text"] = self.wiki_df["text"].apply(
            self.preprocessor.preprocess
        )
        self.wiki_df["preprocessed_text"] = self.wiki_df.apply(
            lambda row: f"{row['title']} {row['preprocessed_text']}", axis=1
        )
        self.wiki_df.drop_duplicates(
            subset=["preprocessed_text"], inplace=True, ignore_index=True
        )

        self.p_embedding = None  # get_sparse_embedding()로 생성합니다

        self.get_embedding()

    def retrieve(
        self, query_or_dataset: Union[str, Dataset], topk: Optional[int] = 1
    ) -> Union[Tuple[List, List], pd.DataFrame]:
        """
        Arguments:
            query_or_dataset (Union[str, Dataset]):
                str이나 Dataset으로 이루어진 Query를 받습니다.
                str 형태인 하나의 query만 받으면 `get_relevant_doc`을 통해 유사도를 구합니다.
                Dataset 형태는 query를 포함한 HF.Dataset을 받습니다.
                이 경우 `get_relevant_doc_bulk`를 통해 유사도를 구합니다.
            topk (Optional[int], optional): Defaults to 1.
                상위 몇 개의 passage를 사용할 것인지 지정합니다.

        Returns:
            1개의 Query를 받는 경우  -> Tuple(List, List)
            다수의 Query를 받는 경우 -> pd.DataFrame: [description]

        Note:
            다수의 Query를 받는 경우,
                Ground Truth가 있는 Query (train/valid) -> 기존 Ground Truth Passage를 같이 반환합니다.
                Ground Truth가 없는 Query (test) -> Retrieval한 Passage만 반환합니다.
        """

        if isinstance(query_or_dataset, str):
            query = self.preprocessor.preprocess(query_or_dataset)
            doc_scores, doc_indeces = self.get_relevant_doc(query, k=topk)
            print("[Search query]\n", query_or_dataset, "\n")

            for i in range(topk):
                print(f"Top-{i+1} passage with score {doc_scores[i]:4f}")
                print(self.wiki_df["text"][doc_indeces[i]])

            return (
                doc_scores,
                [self.wiki_df["text"][doc_indeces[i]] for i in range(topk)],
            )

        elif isinstance(query_or_dataset, Dataset):
            # Retrieve한 Passage를 pd.DataFrame으로 반환합니다.
            total = []
            queries = [
                self.preprocessor.preprocess(query)
                for query in query_or_dataset["question"]
            ]
            with timer("query exhaustive search"):
                doc_scores, doc_indeces = self.get_relevant_doc_bulk(queries, k=topk)
            for idx, example in enumerate(
                tqdm(query_or_dataset, desc="Sparse retrieval: ")
            ):
                tmp = {
                    # Query와 해당 id를 반환합니다.
                    "question": example["question"],
                    "id": example["id"],
                    # Retrieve한 Passage의 id, context를 반환합니다.
                    "title": [self.wiki_df["title"][pid] for pid in doc_indeces[idx]],
                    "context": [self.wiki_df["text"][pid] for pid in doc_indeces[idx]],
                    "document_id": [
                        self.wiki_df["document_id"][pid] for pid in doc_indeces[idx]
                    ],
                }
                if "context" in example.keys() and "answers" in example.keys():
                    # validation 데이터를 사용하면 ground_truth context와 answer도 반환합니다.
                    tmp["original_context"] = example["context"]
                    tmp["original_document_id"] = example["document_id"]
                    tmp["answers"] = example["answers"]
                total.append(tmp)

            return pd.DataFrame(total)

    def print_performance(self, dataset: Dataset, topk_list: List) -> None:
        """_summary_
        Top-k accuracy를 계산해서 출력합니다.
        Args:
            dataset (Dataset): Top-k accuracy를 평가하기 위해 사용할 데이터셋
            topk_list (List): 원하는 k 값의 리스트. 예를 들어, Top-10과 Top-20을 보고 싶다면 [10, 20]
        """
        topk_list.sort()
        with timer("bulk query"):
            result_df = self.retrieve(dataset, topk=max(topk_list))

        topk_corrects = [0] * len(topk_list)

        for idx, row in result_df.iterrows():
            org_context = self.preprocessor.preprocess(row["original_context"])
            contexts = [
                self.preprocessor.preprocess(context) for context in row["context"]
            ]
            if org_context not in contexts:
                continue

            k = contexts.index(org_context) + 1

            for i, topk in enumerate(topk_list):
                if k <= topk:
                    topk_corrects[i] += 1

        topk_accuracy = [correct / len(result_df) for correct in topk_corrects]

        for i in range(len(topk_list)):
            print(f"Top-{topk_list[i]} accuracy: {topk_accuracy[i] * 100:.1f}%")

    @abstractmethod
    def get_embedding(self) -> None:
        """child class에서구현"""
        raise NotImplementedError

    @abstractmethod
    def get_relevant_doc(self, query: str, k: Optional[int] = 1) -> Tuple[List, List]:
        """child class에서구현"""
        raise NotImplementedError

    @abstractmethod
    def get_relevant_doc_bulk(
        self, queries: List, k: Optional[int] = 1
    ) -> Tuple[List, List]:
        """child class에서구현"""
        raise NotImplementedError


class TfidfRetriever(_BaseRetriever):
    def __init__(
        self,
        tokenize_fn,
        preprocessor: Preprocessor,
        data_path: Optional[str] = "../data/",
        context_path: Optional[str] = "wikipedia_documents.json",
    ) -> None:
        super().__init__(
            tokenize_fn,
            preprocessor,
            data_path,
            context_path,
        )

    def get_embedding(self) -> None:
        """
        Summary:
            Passage Embedding을 만들고
            TFIDF와 Embedding을 pickle로 저장합니다.
            만약 미리 저장된 파일이 있으면 저장된 pickle을 불러옵니다.
        """

        # Pickle을 저장합니다.
        pickle_name = f"sparse_embedding.bin"
        tfidfv_name = f"tfidv.bin"
        emd_path = os.path.join(self.data_path, pickle_name)
        tfidfv_path = os.path.join(self.data_path, tfidfv_name)

        if os.path.isfile(emd_path) and os.path.isfile(tfidfv_path):
            with open(emd_path, "rb") as file:
                self.p_embedding = pickle.load(file)
            with open(tfidfv_path, "rb") as file:
                self.tfidfv = pickle.load(file)
            print("Embedding pickle load.")
            print("passage embedding shape : {}".format(self.p_embedding.shape))
        else:
            # Transform by vectorizer
            self.tfidfv = TfidfVectorizer(
                tokenizer=self.tokenize_fn,
                ngram_range=(1, 2),
                max_features=50000,
            )
            print("Build passage embedding")
            self.p_embedding = self.tfidfv.fit_transform(
                self.wiki_df["preprocessed_text"].to_list()
            )
            print("passage embedding shape : {}".format(self.p_embedding.shape))
            with open(emd_path, "wb") as file:
                pickle.dump(self.p_embedding, file)
            with open(tfidfv_path, "wb") as file:
                pickle.dump(self.tfidfv, file)
            print("Embedding pickle saved.")

    def get_relevant_doc(self, query: str, k: Optional[int] = 1) -> Tuple[List, List]:
        """
        Arguments:
            query (str):
                하나의 Query를 받습니다.
            k (Optional[int]): 1
                상위 몇 개의 Passage를 반환할지 정합니다.
        Note:
            vocab 에 없는 이상한 단어로 query 하는 경우 assertion 발생 (예) 뙣뙇?
        """

        with timer("transform"):
            query_vec = self.tfidfv.transform([query])
        assert (
            np.sum(query_vec) != 0
        ), "오류가 발생했습니다. 이 오류는 보통 query에 vectorizer의 vocab에 없는 단어만 존재하는 경우 발생합니다."

        with timer("query ex search"):
            result = np.dot(query_vec, self.p_embedding.T)
        if not isinstance(result, np.ndarray):
            result = result.toarray()

        sorted_result = np.argsort(result.squeeze())[::-1]

        doc_score = result.squeeze()[sorted_result].tolist()[:k]
        doc_indeces = sorted_result.tolist()[:k]
        return doc_score, doc_indeces

    def get_relevant_doc_bulk(
        self, queries: List, k: Optional[int] = 1
    ) -> Tuple[List, List]:
        """
        Arguments:
            queries (List):
                하나의 Query를 받습니다.
            k (Optional[int]): 1
                상위 몇 개의 Passage를 반환할지 정합니다.
        Note:
            vocab 에 없는 이상한 단어로 query 하는 경우 assertion 발생 (예) 뙣뙇?
        """

        query_vec = self.tfidfv.transform(queries)
        assert (
            np.sum(query_vec) != 0
        ), "오류가 발생했습니다. 이 오류는 보통 query에 vectorizer의 vocab에 없는 단어만 존재하는 경우 발생합니다."

        result = query_vec * self.p_embedding.T
        if not isinstance(result, np.ndarray):
            result = result.toarray()
        doc_scores = []
        doc_indices = []
        for i in range(result.shape[0]):
            sorted_result = np.argsort(result[i, :])[::-1]
            doc_scores.append(result[i, :][sorted_result].tolist()[:k])
            doc_indices.append(sorted_result.tolist()[:k])
        return doc_scores, doc_indices


class FaissRetriever(TfidfRetriever):
    def __init__(
        self,
        tokenize_fn,
        num_clusters=64,
        data_path: Optional[str] = "../data/",
        context_path: Optional[str] = "wikipedia_documents.json",
    ) -> None:
        """
        Arguments:
            tokenize_fn:
                기본 text를 tokenize해주는 함수입니다.
                아래와 같은 함수들을 사용할 수 있습니다.
                - lambda x: x.split(' ')
                - Huggingface Tokenizer
                - konlpy.tag의 Mecab

            data_path:
                데이터가 보관되어 있는 경로입니다.

            context_path:
                Passage들이 묶여있는 파일명입니다.

            data_path/context_path가 존재해야합니다.

        Summary:
            Passage 파일을 불러오고 TfidfVectorizer를 선언하는 기능을 합니다.
        """

        super().__init__(
            tokenize_fn,
            data_path,
            context_path,
        )
        self.build_faiss(num_clusters=num_clusters)

    def build_faiss(self, num_clusters=64) -> None:
        """
        Summary:
            속성으로 저장되어 있는 Passage Embedding을
            Faiss indexer에 fitting 시켜놓습니다.
            이렇게 저장된 indexer는 `get_relevant_doc`에서 유사도를 계산하는데 사용됩니다.

        Note:
            Faiss는 Build하는데 시간이 오래 걸리기 때문에,
            매번 새롭게 build하는 것은 비효율적입니다.
            그렇기 때문에 build된 index 파일을 저정하고 다음에 사용할 때 불러옵니다.
            다만 이 index 파일은 용량이 1.4Gb+ 이기 때문에 여러 num_clusters로 시험해보고
            제일 적절한 것을 제외하고 모두 삭제하는 것을 권장합니다.
        """

        indexer_name = f"faiss_clusters{num_clusters}.index"
        indexer_path = os.path.join(self.data_path, indexer_name)
        if os.path.isfile(indexer_path):
            print("Load Saved Faiss Indexer.")
            self.indexer = faiss.read_index(indexer_path)

        else:
            p_emb = self.p_embedding.astype(np.float32).toarray()
            emb_dim = p_emb.shape[-1]

            num_clusters = num_clusters
            quantizer = faiss.IndexFlatL2(emb_dim)

            self.indexer = faiss.IndexIVFScalarQuantizer(
                quantizer, quantizer.d, num_clusters, faiss.METRIC_L2
            )
            self.indexer.train(p_emb)
            self.indexer.add(p_emb)
            faiss.write_index(self.indexer, indexer_path)
            print("Faiss Indexer Saved.")

    def get_relevant_doc(self, query: str, k: Optional[int] = 1) -> Tuple[List, List]:
        """
        Arguments:
            query (str):
                하나의 Query를 받습니다.
            k (Optional[int]): 1
                상위 몇 개의 Passage를 반환할지 정합니다.
        Note:
            vocab 에 없는 이상한 단어로 query 하는 경우 assertion 발생 (예) 뙣뙇?
        """

        query_vec = self.tfidfv.transform([query])
        assert (
            np.sum(query_vec) != 0
        ), "오류가 발생했습니다. 이 오류는 보통 query에 vectorizer의 vocab에 없는 단어만 존재하는 경우 발생합니다."

        q_emb = query_vec.toarray().astype(np.float32)
        with timer("query faiss search"):
            D, I = self.indexer.search(q_emb, k)

        return D.tolist()[0], I.tolist()[0]

    def get_relevant_doc_bulk(
        self, queries: List, k: Optional[int] = 1
    ) -> Tuple[List, List]:
        """
        Arguments:
            queries (List):
                하나의 Query를 받습니다.
            k (Optional[int]): 1
                상위 몇 개의 Passage를 반환할지 정합니다.
        Note:
            vocab 에 없는 이상한 단어로 query 하는 경우 assertion 발생 (예) 뙣뙇?
        """

        query_vecs = self.tfidfv.transform(queries)
        assert (
            np.sum(query_vecs) != 0
        ), "오류가 발생했습니다. 이 오류는 보통 query에 vectorizer의 vocab에 없는 단어만 존재하는 경우 발생합니다."

        q_embs = query_vecs.toarray().astype(np.float32)
        D, I = self.indexer.search(q_embs, k)

        return D.tolist(), I.tolist()


class BaseDenseRetriever(_BaseRetriever):
    def __init__(
        self,
        tokenize_fn,
        preprocessor: Preprocessor,
        retriever_path: str,
        data_path: Optional[str] = "./data/",
        passage_path: Optional[str] = "wikipedia_passages.json",
        context_path: Optional[str] = "wikipedia_documents.json",
    ) -> None:
        """
        DPR Retriever입니다.
        Args:
            tokenize_fn: sparse retriever와의 호환성을 위해서 만든 Argument입니다. 이 tokenizer는 사용되지 않으며, 훈련시에 사용했던 tokenizer를 사용합니다.
            data_path (Optional[str], optional): data 폴더의 경로입니다. Defaults to "./data/".
            passage_path (Optional[str], optional): wikipedia passage json파일의 경로입니다. Defaults to "wikipedia_passages.json".
            context_path (Optional[str], optional): wikipedia document json파일의 경로입니다. Defaults to "wikipedia_documents.json".
        """
        if tokenize_fn is not None:
            logging.warning(
                "tokenize_fn으로 전달한 tokenizer는 사용되지 않습니다. DPR 학습 시에 사용한 tokenizer를 사용합니다."
            )
            tokenize_fn = None
        with open(os.path.join(retriever_path, "encoder_model_name.txt"), "r") as f:
            model_name = f.readline()
        self.device = "cuda" if torch.cuda.is_available else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = DPR(None, for_train=False, output_dir=retriever_path)
        self.model = self.model.to(self.device)
        self.model.eval()
        with open(os.path.join(data_path, passage_path)) as f:
            passages = json.load(f)
        self.passages_df = pd.DataFrame(passages).T
        super().__init__(None, preprocessor, data_path, context_path)

    def get_embedding(self):
        pickle_name = f"passage_embedding.bin"
        emd_path = os.path.join(self.data_path, pickle_name)

        if os.path.isfile(emd_path):
            with open(emd_path, "rb") as file:
                self.p_embedding = pickle.load(file)
            print("Embedding pickle load.")
            print("passage embedding shape : {}".format(self.p_embedding.shape))
        else:
            # passage_dataset = PassageDataset(
            #     os.path.join(self.data_path, self.passage_path),
            #     tokenizer=self.tokenizer,
            #     max_len=512,
            # )

            # def collate_fn(batch):
            #     keys = batch[0].keys()
            #     batched_data = {}

            #     for key in keys:
            #         batched_data[key] = torch.tensor([sample[key] for sample in batch])

            #     return batched_data

            # dataloader = DataLoader(
            #     passage_dataset,
            #     batch_size=512,
            #     num_workers=4,
            #     collate_fn=collate_fn,
            # )
            passages = [
                self.tokenizer.sep_token.join(pair)
                for pair in zip(self.passages_df["title"], self.passages_df["text"])
            ]
            embeddings = []
            batch_size = 512
            with torch.no_grad():
                for idx in tqdm(
                    range(0, len(passages), batch_size), desc="Build passage embedding"
                ):
                    tokenized = self.tokenizer(
                        passages[idx : idx + batch_size],
                        max_length=512,
                        padding="max_length",
                        truncation=True,
                        return_tensors="pt",
                    )
                    embedding = self.model.get_passage_embedding(
                        {
                            "input_ids": tokenized["input_ids"].to(self.device),
                            "token_type_ids": tokenized["token_type_ids"].to(
                                self.device
                            ),
                            "attention_mask": tokenized["attention_mask"].to(
                                self.device
                            ),
                        }
                    )
                    embeddings.append(embedding)

            self.p_embedding = torch.cat(embeddings, dim=0)

            print("passage embedding shape : {}".format(self.p_embedding.shape))
            with open(emd_path, "wb") as file:
                pickle.dump(self.p_embedding, file)
            print("Embedding pickle saved.")

    def get_relevant_doc(self, query: str, k: Optional[int] = 1) -> Tuple[List, List]:
        tokenized_query = self.tokenizer(query, return_tensors="pt").to(self.device)
        with torch.no_grad():
            q_embedding = self.model.get_question_embedding(tokenized_query)
        result = torch.matmul(q_embedding, self.p_embedding.T)
        if not isinstance(result, np.ndarray):
            result = result.detach().cpu().numpy()

        sorted_result = np.argsort(result.squeeze())[::-1]
        p_indeces = []
        doc_ids = []
        index = 0
        # top k개의 passage 중에 동일한 document에서 나온 passage가 있을 수 있습니다. 중복된 document를 제외하고 top k개의 document를 반환합니다.
        while len(p_indeces) < k:
            doc_id = self.passages_df["document_id"][sorted_result[index]]
            if doc_id not in doc_ids:
                p_indeces.append(sorted_result[index])
                doc_ids.append(doc_id)
            index += 1
        doc_scores = result.squeeze()[p_indeces].tolist()
        doc_indeces = [
            self.wiki_df.index[self.wiki_df["document_id"] == doc_id].item()
            for doc_id in doc_ids
        ]

        return doc_scores, doc_indeces

    def get_relevant_doc_bulk(
        self, queries: List, k: Optional[int] = 1
    ) -> Tuple[List, List]:
        """
        Arguments:
            queries (List):
                여러 개의 Query를 받습니다.
            k (Optional[int]): 1
                상위 몇 개의 Passage를 반환할지 정합니다.
        """
        tokenized_queries = self.tokenizer(
            queries,
            return_tensors="pt",
            max_length=64,
            padding="max_length",
            truncation=True,
        ).to(self.device)

        batch_size = 128
        q_embedding = []

        with torch.no_grad():
            for i in range(0, len(queries), batch_size):
                batch = {k: v[i : i + batch_size] for k, v in tokenized_queries.items()}
                q_embedding.append(self.model.get_question_embedding(batch))

        q_embedding = torch.cat(q_embedding, dim=0)
        result = torch.matmul(q_embedding, self.p_embedding.T)
        if not isinstance(result, np.ndarray):
            result = result.detach().cpu().numpy()

        bulk_doc_scores = []
        bulk_doc_indeces = []
        for i in range(result.shape[0]):
            sorted_result = np.argsort(result[i, :])[::-1]
            p_indeces = []
            doc_ids = []
            index = 0
            # top k개의 passage 중에 동일한 document에서 나온 passage가 있을 수 있습니다. 중복된 document를 제외하고 top k개의 document를 반환합니다.
            while len(p_indeces) < k:
                doc_id = self.passages_df["document_id"][sorted_result[index]]
                if doc_id not in doc_ids:
                    p_indeces.append(sorted_result[index])
                    doc_ids.append(doc_id)
                index += 1
            doc_scores = result[i, :][p_indeces].tolist()
            doc_indeces = [
                self.wiki_df.index[self.wiki_df["document_id"] == doc_id].item()
                for doc_id in doc_ids
            ]
            bulk_doc_scores.append(doc_scores)
            bulk_doc_indeces.append(doc_indeces)
        return bulk_doc_scores, bulk_doc_indeces


class BM25SparseRetrieval(_BaseRetriever):
    def __init__(
        self,
        tokenize_fn,
        preprocessor: Preprocessor,
        data_path: Optional[str] = "../data/",
        context_path: Optional[str] = "wikipedia_documents.json",
    ) -> None:
        super().__init__(tokenize_fn, preprocessor, data_path, context_path)

    def get_embedding(self) -> None:
        """
        Summary:
            Passage Embedding을 만들고
            bm25와 Embedding을 pickle로 저장합니다.
            만약 미리 저장된 파일이 있으면 저장된 pickle을 불러옵니다.
        """

        bm25_name = f"bm25.bin"
        bm25_path = os.path.join(self.data_path, bm25_name)

        if os.path.isfile(bm25_path):
            with open(bm25_path, "rb") as file:
                self.bm25 = pickle.load(file)
            print("Embedding pickle load.")
            print("corpus size : ", self.bm25.corpus_size)
            # print("passage embedding shape : {}".format(self.bm25.corpus_size))

        else:
            print("Build passage embedding")
            self.bm25 = BM25Okapi(
                self.wiki_df["preprocessed_text"].to_list(), tokenizer=self.tokenize_fn
            )
            # self.bm25 = BM25Plus(self.contexts, tokenizer=self.tokenize_fn)
            with open(bm25_path, "wb") as file:
                pickle.dump(self.bm25, file)
            print("BM25 pickle saved.")

    def get_relevant_doc(self, query: str, k: Optional[int] = 1) -> Tuple[List, List]:
        with timer("transform"):
            tokenized_query = self.tokenize_fn(query)
        with timer("query ex search"):
            result = self.bm25.get_scores(tokenized_query)
        sorted_result = np.argsort(result)[::-1]
        doc_score = result[sorted_result].tolist()[:k]
        doc_indeces = sorted_result.tolist()[:k]
        return doc_score, doc_indeces

    def get_relevant_doc_bulk(
        self, queries: List, k: Optional[int] = 1
    ) -> Tuple[List, List]:
        with timer("transform"):
            tokenized_queries = [self.tokenize_fn(query) for query in queries]
        with timer("query ex search"):
            result = np.array(
                [
                    self.bm25.get_scores(tokenized_query)
                    for tokenized_query in tqdm(tokenized_queries)
                ]
            )
        doc_scores = []
        doc_indeces = []
        for i in range(result.shape[0]):
            sorted_result = np.argsort(result[i, :])[::-1]
            doc_scores.append(result[i, :][sorted_result].tolist()[:k])
            doc_indeces.append(sorted_result.tolist()[:k])
        return doc_scores, doc_indeces


def get_args():
    import argparse

    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--method", default="bm25", type=str, help="Type of Retriever")
    parser.add_argument(
        "--dataset_name", default="./data/train_dataset", type=str, help=""
    )
    parser.add_argument(
        "--tokenizer_name",
        default="KoichiYasuoka/roberta-base-korean-morph-upos",
        type=str,
        help="Name of pretrained tokenizer for sparse retriever",
    )
    parser.add_argument("--data_path", default="./data", type=str, help="")
    parser.add_argument(
        "--context_path",
        default="wikipedia_documents.json",
        type=str,
        help="Path of wikipedia documents",
    )
    parser.add_argument(
        "--passage_path",
        default="wikipedia_passages.json",
        type=str,
        help="Path of wikipedia passages",
    )
    parser.add_argument("--topk", default=10, type=int, help="")
    parser.add_argument(
        "--retriever_output_path",
        default="/opt/level2_nlp_mrc-nlp-03/retriever_outputs/klue-bert-base",
        type=str,
        help="훈련된 DPR 모델의 output 경로",
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    # Retriever의 성능을 평가해볼 수 있는 공간입니다.

    args = get_args()
    org_dataset = load_from_disk(args.dataset_name)
    val_ds = org_dataset["validation"]
    print("*" * 40, "query dataset", "*" * 40)
    print(val_ds)

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name,
        use_fast=False,
    )

    preprocessor = Preprocessor(no_other_languages=False, quoat_normalize=True)

    if args.method == "faiss":
        retriever = FaissRetriever(
            tokenize_fn=tokenizer.tokenize,
            num_clusters=64,
            data_path=args.data_path,
            context_path=args.context_path,
        )

    elif args.method == "tfidf":
        retriever = TfidfRetriever(
            tokenize_fn=tokenizer.tokenize,
            preprocessor=preprocessor,
            data_path=args.data_path,
            context_path=args.context_path,
        )

    elif args.method == "dpr":
        retriever = BaseDenseRetriever(
            tokenize_fn=tokenizer,
            preprocessor=preprocessor,
            retriever_path=args.retriever_output_path,
            data_path=args.data_path,
            passage_path=args.passage_path,
            context_path=args.context_path,
        )

    elif args.method == "bm25":
        retriever = BM25SparseRetrieval(
            tokenize_fn=tokenizer.tokenize,
            preprocessor=preprocessor,
            data_path=args.data_path,
            context_path=args.context_path,
        )

    retriever.print_performance(val_ds, topk_list=[10, 20, 50, 100])
