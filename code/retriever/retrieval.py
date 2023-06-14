import json
import os
import pickle
import time
import torch
import faiss
import numpy as np
import pandas as pd
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
    from retriever.DPR_dataset import PassageDataset
except ImportError:
    from DPR_model import DPR
    from DPR_dataset import PassageDataset

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
        self.data_path = data_path
        with open(os.path.join(data_path, context_path), "r", encoding="utf-8") as f:
            wiki = json.load(f)
        self.wiki_df = pd.DataFrame(wiki).T.set_index("document_id", drop=False)

        self.p_embedding = None  # get_sparse_embedding()로 생성합니다
        self.indexer = None  # build_faiss()로 생성합니다.

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

        assert self.p_embedding is not None, "get_sparse_embedding() 메소드를 먼저 수행해줘야합니다."

        if isinstance(query_or_dataset, str):
            doc_scores, doc_indices = self.get_relevant_doc(query_or_dataset, k=topk)
            print("[Search query]\n", query_or_dataset, "\n")

            for i in range(topk):
                print(f"Top-{i+1} passage with score {doc_scores[i]:4f}")
                print(self.wiki_df["text"][doc_indices[i]])

            return (
                doc_scores,
                [self.wiki_df["text"][doc_indices[i]] for i in range(topk)],
            )

        elif isinstance(query_or_dataset, Dataset):
            # Retrieve한 Passage를 pd.DataFrame으로 반환합니다.
            total = []
            with timer("query exhaustive search"):
                doc_scores, doc_indices = self.get_relevant_doc_bulk(
                    query_or_dataset["question"], k=topk
                )
            for idx, example in enumerate(
                tqdm(query_or_dataset, desc="Sparse retrieval: ")
            ):
                tmp = {
                    # Query와 해당 id를 반환합니다.
                    "question": example["question"],
                    "id": example["id"],
                    # Retrieve한 Passage의 id, context를 반환합니다.
                    "context": [self.wiki_df["text"][pid] for pid in doc_indices[idx]],
                    "document_id": [
                        self.wiki_df["document_id"][pid] for pid in doc_indices[idx]
                    ],
                }
                if "context" in example.keys() and "answers" in example.keys():
                    # validation 데이터를 사용하면 ground_truth context와 answer도 반환합니다.
                    tmp["original_context"] = example["context"]
                    tmp["original_document_id"] = example["document_id"]
                    tmp["answers"] = example["answers"]
                total.append(tmp)

            return pd.DataFrame(total)

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
            print(
                "passage embedding shape : {}".format(
                    self.p_embedding["embedding"].shape
                )
            )
        else:
            # Transform by vectorizer
            self.tfidfv = TfidfVectorizer(
                tokenizer=self.tokenize_fn,
                ngram_range=(1, 2),
                max_features=50000,
            )
            print("Build passage embedding")
            embedding = self.tfidfv.fit_transform(self.wiki_df["text"].to_list())
            self.p_embedding = {
                "embedding": embedding,
                "document_id": self.wiki_df["document_id"].to_numpy(),
            }
            print(
                "passage embedding shape : {}".format(
                    self.p_embedding["embedding"].shape
                )
            )
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
            result = np.dot(query_vec, self.p_embedding["embedding"].T)
        if not isinstance(result, np.ndarray):
            result = result.toarray()

        sorted_result = np.argsort(result.squeeze())[::-1]

        doc_score = result.squeeze()[sorted_result].tolist()[:k]
        doc_ids = self.p_embedding["document_id"][sorted_result[:k]].tolist()
        return doc_score, doc_ids

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

        result = query_vec * self.p_embedding["embedding"].T
        if not isinstance(result, np.ndarray):
            result = result.toarray()
        doc_scores = []
        doc_indices = []
        for i in range(result.shape[0]):
            sorted_result = np.argsort(result[i, :])[::-1]
            doc_scores.append(result[i, :][sorted_result].tolist()[:k])
            doc_indices.append(
                self.p_embedding["document_id"][sorted_result[:k]].tolist()
            )
        return doc_scores, doc_indices


class LatentSemanticIndexingRetriever(_BaseRetriever):
    def __init__(
        self,
        tokenize_fn,
        num_components: int = 10,
        data_path: Optional[str] = "../data/",
        context_path: Optional[str] = "wikipedia_documents.json",
    ) -> None:
        self.num_components = num_components
        self.lsi = TruncatedSVD(n_components=self.num_components, n_iter=100)
        super().__init__(tokenize_fn, data_path, context_path)

    def get_sparse_embedding(self) -> None:
        """
        Summary:
            Passage Embedding을 만들고
            TFIDF와 Embedding을 pickle로 저장합니다.
            만약 미리 저장된 파일이 있으면 저장된 pickle을 불러옵니다.
        """

        # Pickle을 저장합니다.
        pickle_name = f"sparse_embedding_lsi_{self.num_components}.bin"
        tfidfv_name = f"tfidv.bin"
        lsi_name = f"lsi_{self.num_components}.bin"
        emd_path = os.path.join(self.data_path, pickle_name)
        tfidfv_path = os.path.join(self.data_path, tfidfv_name)
        lsi_path = os.path.join(self.data_path, lsi_name)

        if (
            os.path.isfile(emd_path)
            and os.path.isfile(tfidfv_path)
            and os.path.isfile(lsi_path)
        ):
            with open(emd_path, "rb") as file:
                self.p_embedding = pickle.load(file)
            with open(tfidfv_path, "rb") as file:
                self.tfidfv: TfidfVectorizer = pickle.load(file)
            print("Embedding pickle load.")
            print("passage embedding shape : {}".format(self.p_embedding.shape))
        else:
            print("Build passage embedding")
            self.p_embedding = self.tfidfv.fit_transform(self.contexts)
            self.p_embedding = self.lsi.fit_transform(self.p_embedding)
            print("passage embedding shape : {}".format(self.p_embedding.shape))
            with open(emd_path, "wb") as file:
                pickle.dump(self.p_embedding, file)
            with open(tfidfv_path, "wb") as file:
                pickle.dump(self.tfidfv, file)
            with open(lsi_path, "wb") as file:
                pickle.dump(self.lsi, file)
            print("Embedding pickle saved.")

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

        assert self.p_embedding is not None, "get_sparse_embedding() 메소드를 먼저 수행해줘야합니다."

        if isinstance(query_or_dataset, str):
            doc_scores, doc_indices = self.get_relevant_doc(query_or_dataset, k=topk)
            print("[Search query]\n", query_or_dataset, "\n")

            for i in range(topk):
                print(f"Top-{i+1} passage with score {doc_scores[i]:4f}")
                print(self.contexts[doc_indices[i]])

            return (doc_scores, [self.contexts[doc_indices[i]] for i in range(topk)])

        elif isinstance(query_or_dataset, Dataset):
            # Retrieve한 Passage를 pd.DataFrame으로 반환합니다.
            total = []
            with timer("query exhaustive search"):
                doc_scores, doc_indices = self.get_relevant_doc_bulk(
                    query_or_dataset["question"], k=topk
                )
            for idx, example in enumerate(
                tqdm(query_or_dataset, desc="Sparse retrieval: ")
            ):
                tmp = {
                    # Query와 해당 id를 반환합니다.
                    "question": example["question"],
                    "id": example["id"],
                    # Retrieve한 Passage의 id, context를 반환합니다.
                    "context": " ".join(
                        [self.contexts[pid] for pid in doc_indices[idx]]
                    ),
                }
                if "context" in example.keys() and "answers" in example.keys():
                    # validation 데이터를 사용하면 ground_truth context와 answer도 반환합니다.
                    tmp["original_context"] = example["context"]
                    tmp["answers"] = example["answers"]
                total.append(tmp)

            return pd.DataFrame(total)

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
            # query_vec = self.lsi.fit_transform(query_vec)
        assert (
            np.sum(query_vec) != 0
        ), "오류가 발생했습니다. 이 오류는 보통 query에 vectorizer의 vocab에 없는 단어만 존재하는 경우 발생합니다."

        with timer("query ex search"):
            result = np.dot(query_vec, self.p_embedding.T)
        if not isinstance(result, np.ndarray):
            result = result.toarray()

        sorted_result = np.argsort(result.squeeze())[::-1]
        doc_score = result.squeeze()[sorted_result].tolist()[:k]
        doc_indices = sorted_result.tolist()[:k]
        return doc_score, doc_indices

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
        query_vec = self.lsi.fit_transform(query_vec)
        assert (
            np.sum(query_vec) != 0
        ), "오류가 발생했습니다. 이 오류는 보통 query에 vectorizer의 vocab에 없는 단어만 존재하는 경우 발생합니다."

        result = np.dot(query_vec, self.p_embedding.T)
        if not isinstance(result, np.ndarray):
            result = result.toarray()
        doc_scores = []
        doc_indices = []
        for i in range(result.shape[0]):
            sorted_result = np.argsort(result[i, :])[::-1]
            doc_scores.append(result[i, :][sorted_result].tolist()[:k])
            doc_indices.append(sorted_result.tolist()[:k])
        return doc_scores, doc_indices


class LatentDirichletAllocationRetriever(_BaseRetriever):
    def __init__(
        self,
        tokenize_fn,
        data_path: Optional[str] = "../data/",
        context_path: Optional[str] = "wikipedia_documents.json",
    ) -> None:
        super().__init__(
            tokenize_fn,
            data_path,
            context_path,
        )


class FaissRetriever(_BaseRetriever):
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
            retrieve와 같은 기능을 하지만 faiss.indexer를 사용합니다.
        """

        assert self.indexer is not None, "build_faiss()를 먼저 수행해주세요."

        if isinstance(query_or_dataset, str):
            doc_scores, doc_indices = self.get_relevant_doc(query_or_dataset, k=topk)
            print("[Search query]\n", query_or_dataset, "\n")

            for i in range(topk):
                print(f"Top-{i+1} passage with score {doc_scores[i]:4f}")
                print(self.contexts[doc_indices[i]])

            return (doc_scores, [self.contexts[doc_indices[i]] for i in range(topk)])

        elif isinstance(query_or_dataset, Dataset):
            # Retrieve한 Passage를 pd.DataFrame으로 반환합니다.
            total = []

            with timer("query faiss search"):
                doc_scores, doc_indices = self.get_relevant_doc_bulk(
                    query_or_dataset["question"], k=topk
                )
            for idx, example in enumerate(
                tqdm(query_or_dataset, desc="Sparse retrieval: ")
            ):
                tmp = {
                    # Query와 해당 id를 반환합니다.
                    "question": example["question"],
                    "id": example["id"],
                    # Retrieve한 Passage의 id, context를 반환합니다.
                    "context": " ".join(
                        [self.contexts[pid] for pid in doc_indices[idx]]
                    ),
                }
                if "context" in example.keys() and "answers" in example.keys():
                    # validation 데이터를 사용하면 ground_truth context와 answer도 반환합니다.
                    tmp["original_context"] = example["context"]
                    tmp["answers"] = example["answers"]
                total.append(tmp)

            return pd.DataFrame(total)

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


class SparseRetrieval(_BaseRetriever):
    def __init__(
        self,
        tokenize_fn,
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

        self.data_path = data_path
        with open(os.path.join(data_path, context_path), "r", encoding="utf-8") as f:
            wiki = json.load(f)

        self.contexts = list(
            dict.fromkeys([v["text"] for v in wiki.values()])
        )  # set 은 매번 순서가 바뀌므로
        print(f"Lengths of unique contexts : {len(self.contexts)}")
        self.ids = list(range(len(self.contexts)))

        # Transform by vectorizer
        self.tfidfv = TfidfVectorizer(
            tokenizer=tokenize_fn,
            ngram_range=(1, 2),
            max_features=50000,
        )

        self.p_embedding = None  # get_sparse_embedding()로 생성합니다
        self.indexer = None  # build_faiss()로 생성합니다.

        self.get_sparse_embedding()

    def get_sparse_embedding(self) -> None:
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
            print("Build passage embedding")
            self.p_embedding = self.tfidfv.fit_transform(self.contexts)
            print("passage embedding shape : {}".format(self.p_embedding.shape))
            with open(emd_path, "wb") as file:
                pickle.dump(self.p_embedding, file)
            with open(tfidfv_path, "wb") as file:
                pickle.dump(self.tfidfv, file)
            print("Embedding pickle saved.")

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

        assert self.p_embedding is not None, "get_sparse_embedding() 메소드를 먼저 수행해줘야합니다."

        if isinstance(query_or_dataset, str):
            doc_scores, doc_indices = self.get_relevant_doc(query_or_dataset, k=topk)
            print("[Search query]\n", query_or_dataset, "\n")

            for i in range(topk):
                print(f"Top-{i+1} passage with score {doc_scores[i]:4f}")
                print(self.contexts[doc_indices[i]])

            return (doc_scores, [self.contexts[doc_indices[i]] for i in range(topk)])

        elif isinstance(query_or_dataset, Dataset):
            # Retrieve한 Passage를 pd.DataFrame으로 반환합니다.
            total = []
            with timer("query exhaustive search"):
                doc_scores, doc_indices = self.get_relevant_doc_bulk(
                    query_or_dataset["question"], k=topk
                )
            for idx, example in enumerate(
                tqdm(query_or_dataset, desc="Sparse retrieval: ")
            ):
                tmp = {
                    # Query와 해당 id를 반환합니다.
                    "question": example["question"],
                    "id": example["id"],
                    # Retrieve한 Passage의 id, context를 반환합니다.
                    "context": " ".join(
                        [self.contexts[pid] for pid in doc_indices[idx]]
                    ),
                }
                if "context" in example.keys() and "answers" in example.keys():
                    # validation 데이터를 사용하면 ground_truth context와 answer도 반환합니다.
                    tmp["original_context"] = example["context"]
                    tmp["answers"] = example["answers"]
                total.append(tmp)

            cqas = pd.DataFrame(total)
            return cqas

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
            result = query_vec * self.p_embedding.T
        if not isinstance(result, np.ndarray):
            result = result.toarray()

        sorted_result = np.argsort(result.squeeze())[::-1]
        doc_score = result.squeeze()[sorted_result].tolist()[:k]
        doc_indices = sorted_result.tolist()[:k]
        return doc_score, doc_indices

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

    def retrieve_faiss(
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
            retrieve와 같은 기능을 하지만 faiss.indexer를 사용합니다.
        """

        assert self.indexer is not None, "build_faiss()를 먼저 수행해주세요."

        if isinstance(query_or_dataset, str):
            doc_scores, doc_indices = self.get_relevant_doc_faiss(
                query_or_dataset, k=topk
            )
            print("[Search query]\n", query_or_dataset, "\n")

            for i in range(topk):
                print("Top-%d passage with score %.4f" % (i + 1, doc_scores[i]))
                print(self.contexts[doc_indices[i]])

            return (doc_scores, [self.contexts[doc_indices[i]] for i in range(topk)])

        elif isinstance(query_or_dataset, Dataset):
            # Retrieve한 Passage를 pd.DataFrame으로 반환합니다.
            queries = query_or_dataset["question"]
            total = []

            with timer("query faiss search"):
                doc_scores, doc_indices = self.get_relevant_doc_bulk_faiss(
                    queries, k=topk
                )
            for idx, example in enumerate(
                tqdm(query_or_dataset, desc="Sparse retrieval: ")
            ):
                tmp = {
                    # Query와 해당 id를 반환합니다.
                    "question": example["question"],
                    "id": example["id"],
                    # Retrieve한 Passage의 id, context를 반환합니다.
                    "context": " ".join(
                        [self.contexts[pid] for pid in doc_indices[idx]]
                    ),
                }
                if "context" in example.keys() and "answers" in example.keys():
                    # validation 데이터를 사용하면 ground_truth context와 answer도 반환합니다.
                    tmp["original_context"] = example["context"]
                    tmp["answers"] = example["answers"]
                total.append(tmp)

            return pd.DataFrame(total)

    def get_relevant_doc_faiss(
        self, query: str, k: Optional[int] = 1
    ) -> Tuple[List, List]:
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

    def get_relevant_doc_bulk_faiss(
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
        tokenize_fn,  # 사용하지 않습니다. 호환성을 위해서 dummy로 넣어두었습니다.
        retriever_path: str,
        data_path: Optional[str] = "./data/",
        passage_path: Optional[str] = "wikipedia_passages.json",
        context_path: Optional[str] = "wikipedia_documents.json",
    ) -> None:
        with open(os.path.join(retriever_path, "encoder_model_name.txt"), "r") as f:
            model_name = f.readline()
        self.device = "cuda" if torch.cuda.is_available else "cpu"
        self.passage_path = passage_path
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = DPR(None, for_train=False, output_dir=retriever_path)
        self.model = self.model.to(self.device)
        self.model.eval()
        super().__init__(tokenize_fn, data_path, context_path)

    def get_embedding(self):
        pickle_name = f"passage_embedding.bin"
        emd_path = os.path.join(self.data_path, pickle_name)

        if os.path.isfile(emd_path):
            with open(emd_path, "rb") as file:
                self.p_embedding = pickle.load(file)
            print("Embedding pickle load.")
            print(
                "passage embedding shape : {}".format(
                    self.p_embedding["embedding"].shape
                )
            )
        else:
            passage_dataset = PassageDataset(
                os.path.join(self.data_path, self.passage_path),
                tokenizer=self.tokenizer,
                max_len=384,
            )

            def collate_fn(batch):
                keys = batch[0].keys()
                batched_data = {}

                for key in keys:
                    batched_data[key] = torch.tensor([sample[key] for sample in batch])

                return batched_data

            dataloader = DataLoader(
                passage_dataset,
                batch_size=512,
                num_workers=4,
                collate_fn=collate_fn,
            )
            embedding = []
            document_id = []
            with torch.no_grad():
                for data in tqdm(
                    dataloader,
                    desc="Build passage embedding",
                    total=len(dataloader),
                ):
                    embeddings = self.model.get_passage_embedding(
                        {
                            "input_ids": data["input_ids"].to(self.device),
                            "token_type_ids": data["token_type_ids"].to(self.device),
                            "attention_mask": data["attention_mask"].to(self.device),
                        }
                    )
                    embedding.append(embeddings)
                    document_id.append(data["document_id"])

            embedding = torch.cat(embedding, dim=0)
            document_id = torch.cat(document_id, dim=0).numpy()

            self.p_embedding = {"embedding": embedding, "document_id": document_id}

            print(
                "passage embedding shape : {}".format(
                    self.p_embedding["embedding"].shape
                )
            )
            with open(emd_path, "wb") as file:
                pickle.dump(self.p_embedding, file)
            print("Embedding pickle saved.")

    def get_relevant_doc(self, query: str, k: Optional[int] = 1) -> Tuple[List, List]:
        tokenized_query = self.tokenizer(query, return_tensors="pt").to(self.device)
        with torch.no_grad():
            q_embedding = self.model.get_question_embedding(tokenized_query)
        result = torch.matmul(q_embedding, self.p_embedding["embedding"].T)
        if not isinstance(result, np.ndarray):
            result = result.detach().cpu().numpy()

        sorted_result = np.argsort(result.squeeze())[::-1]
        top_k_indeces = []
        doc_ids = []
        index = 0
        # top k개의 passage 중에 동일한 document에서 나온 passage가 있을 수 있습니다. 중복된 document를 제외하고 top k개의 document를 반환합니다.
        while len(top_k_indeces) < k:
            doc_id = self.p_embedding["document_id"][sorted_result[index]]
            if doc_id not in doc_ids:
                top_k_indeces.append(sorted_result[index])
                doc_ids.append(doc_id)
            index += 1
        doc_scores = result.squeeze()[top_k_indeces].tolist()
        return doc_scores, doc_ids

    def get_relevant_doc_bulk(
        self, queries: List, k: Optional[int] = 1
    ) -> Tuple[List, List]:
        """
        Arguments:
            queries (List):
                여러 개의 Query를 받습니다.
            k (Optional[int]): 1
                상위 몇 개의 Passage를 반환할지 정합니다.
        Note:
            vocab 에 없는 이상한 단어로 query 하는 경우 assertion 발생 (예) 뙣뙇?
        """
        tokenized_queries = self.tokenizer(
            queries,
            return_tensors="pt",
            max_length=64,
            padding="max_length",
            truncation=True,
        ).to(self.device)

        batch_size = 128
        len_queries = len(queries)
        iteration = len_queries // batch_size + bool(len_queries % batch_size)
        q_embedding = []

        with torch.no_grad():
            for i in range(iteration):
                batch = {
                    k: v[i * batch_size : min((i + 1) * batch_size, len_queries)]
                    for k, v in tokenized_queries.items()
                }
                q_embedding.append(self.model.get_question_embedding(batch))

        q_embedding = torch.cat(q_embedding, dim=0)
        result = torch.matmul(q_embedding, self.p_embedding["embedding"].T)
        if not isinstance(result, np.ndarray):
            result = result.detach().cpu().numpy()

        bulk_doc_scores = []
        bulk_doc_ids = []
        for i in range(result.shape[0]):
            sorted_result = np.argsort(result[i, :])[::-1]
            top_k_indeces = []
            doc_ids = []
            index = 0
            # top k개의 passage 중에 동일한 document에서 나온 passage가 있을 수 있습니다. 중복된 document를 제외하고 top k개의 document를 반환합니다.
            while len(top_k_indeces) < k:
                doc_id = self.p_embedding["document_id"][sorted_result[index]]
                if doc_id not in doc_ids:
                    top_k_indeces.append(sorted_result[index])
                    doc_ids.append(doc_id)
                index += 1
            doc_scores = result[i, :][top_k_indeces].tolist()
            bulk_doc_scores.append(doc_scores)
            bulk_doc_ids.append(doc_ids)
        return bulk_doc_scores, bulk_doc_ids


class BM25SparseRetrieval(_BaseRetriever):
    def __init__(
        self,
        tokenize_fn,
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

        self.data_path = data_path
        with open(os.path.join(data_path, context_path), "r", encoding="utf-8") as f:
            wiki = json.load(f)

        self.contexts = list(
            dict.fromkeys([v["text"] for v in wiki.values()])
        )  # set 은 매번 순서가 바뀌므로
        print(f"Lengths of unique contexts : {len(self.contexts)}")
        # self.ids = list(range(len(self.contexts)))
        self.ids = np.array(
            list(dict.fromkeys([v["document_id"] for v in wiki.values()]))
        )
        self.tokenize_fn = tokenize_fn

        self.bm25 = None  # get_sparse_embedding()로 생성
        self.indexer = None  # build_faiss()로 생성합니다.

        self.get_sparse_embedding()

    def get_sparse_embedding(self) -> None:
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

            # print("passage embedding shape : {}".format(self.bm25.corpus_size))

        else:
            print("Build passage embedding")
            # tokenized_corpus = [self.tokenize_fn(title + text) for title, text in zip(self.wiki_title, self.wiki_text)]
            self.bm25 = BM25Okapi(self.contexts, tokenizer=self.tokenize_fn)
            # self.bm25 = BM25Plus(self.contexts, tokenizer=self.tokenize_fn)
            with open(bm25_path, "wb") as file:
                pickle.dump(self.bm25, file)
            print("BM25 pickle saved.")

    def retrieve(
        self,
        query_or_dataset: Union[str, Dataset],
        topk: Optional[int] = 1,
    ) -> Union[Tuple[List, List], pd.DataFrame]:
        assert self.bm25 is not None, "get_sparse_embedding() 메소드를 먼저 수행해줘야합니다."

        if isinstance(query_or_dataset, str):
            doc_scores, doc_indices = self.get_relevant_doc(query_or_dataset, k=topk)
            print("[Search query]\n", query_or_dataset, "\n")

            for i in range(topk):
                print(f"Top-{i+1} passage with score {doc_scores[i]:4f}")
                print(self.contexts[doc_indices[i]])

            return (doc_scores, [self.contexts[doc_indices[i]] for i in range(topk)])

        elif isinstance(query_or_dataset, Dataset):
            # Retrieve한 Passage를 pd.DataFrame으로 반환합니다.
            total = []
            with timer("query exhaustive search"):
                # doc_scores, doc_indices -> (num_query, top_k)
                doc_scores, doc_indices = self.get_relevant_doc_bulk(
                    query_or_dataset["question"], k=topk
                )
            for idx, example in enumerate(
                tqdm(query_or_dataset, desc="Sparse retrieval: ")
            ):
                tmp = {
                    # Query와 해당 id를 반환합니다.
                    "question": example["question"],
                    "id": example["id"],
                    # Retrieve한 Passage의 id, context를 반환합니다.
                    "context": " ".join(
                        [self.contexts[pid] for pid in doc_indices[idx]]
                    ),
                }
                if "context" in example.keys() and "answers" in example.keys():
                    # validation 데이터를 사용하면 ground_truth context와 answer도 반환합니다.
                    tmp["original_context"] = example["context"]
                    tmp["answers"] = example["answers"]
                total.append(tmp)

            cqas = pd.DataFrame(total)
            return cqas

    def get_relevant_doc(self, query: str, k: Optional[int] = 1) -> Tuple[List, List]:
        with timer("transform"):
            tokenized_query = self.tokenize_fn(query)
        with timer("query ex search"):
            result = self.bm25.get_scores(tokenized_query)
        sorted_result = np.argsort(result)[::-1]
        doc_score = result[sorted_result].tolist()[:k]
        doc_indices = sorted_result.tolist()[:k]
        # doc_indices = self.ids[sorted_result].tolist()[:k]
        return doc_score, doc_indices

    def get_relevant_doc_bulk(
        self, queries: List, k: Optional[int] = 1
    ) -> Tuple[List, List]:
        with timer("transform"):
            tokenized_queris = [self.tokenize_fn(query) for query in queries]
        with timer("query ex search"):
            result = np.array(
                [
                    self.bm25.get_scores(tokenized_query)
                    for tokenized_query in tqdm(tokenized_queris)
                ]
            )
        doc_scores = []
        doc_indices = []
        for i in range(result.shape[0]):
            sorted_result = np.argsort(result[i, :])[::-1]
            doc_scores.append(result[i, :][sorted_result].tolist()[:k])
            doc_indices.append(sorted_result.tolist()[:k])
            # doc_indices.append(self.ids[sorted_result].tolist()[:k])
        return doc_scores, doc_indices


def get_args():
    import argparse

    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--method", default="dpr", type=str, help="")
    parser.add_argument(
        "--dataset_name", default="./data/train_dataset", type=str, help=""
    )
    parser.add_argument(
        "--model_name_or_path",
        default="klue/roberta-base",
        type=str,
        help="",
    )
    parser.add_argument("--data_path", default="./data", type=str, help="")
    parser.add_argument(
        "--context_path", default="wikipedia_documents.json", type=str, help=""
    )
    parser.add_argument(
        "--passage_path",
        default="wikipedia_passages.json",
        type=str,
        help="path of wikipedia passages",
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
    args = get_args()

    # Test sparse
    org_dataset = load_from_disk(args.dataset_name)
    full_ds = concatenate_datasets(
        [
            org_dataset["train"].flatten_indices(),
            org_dataset["validation"].flatten_indices(),
        ]
    )  # train dev 를 합친 4192 개 질문에 대해 모두 테스트
    print("*" * 40, "query dataset", "*" * 40)
    print(full_ds)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        use_fast=False,
    )

    # retriever = TfidfRetriever(
    #     tokenize_fn=tokenizer.tokenize,
    #     data_path=args.data_path,
    #     context_path=args.context_path,
    # )
    # retriever.get_sparse_embedding()

    query = "대통령을 포함한 미국의 행정부 견제권을 갖는 국가 기관은?"

    if args.method == "faiss":
        retriever = FaissRetriever(
            tokenize_fn=tokenizer.tokenize,
            num_clusters=64,
            data_path=args.data_path,
            context_path=args.context_path,
        )
        # test single query
        with timer("single query by faiss"):
            scores, indices = retriever.retrieve(query, topk=args.topk)

        # test bulk
        with timer("bulk query by exhaustive search"):
            df = retriever.retrieve(full_ds, topk=args.topk)
            df["correct"] = df["original_context"] == df["context"]

            print("correct retrieval result by faiss", df["correct"].sum() / len(df))

    elif args.method == "tfidf":
        retriever = TfidfRetriever(
            tokenize_fn=tokenizer.tokenize,
            data_path=args.data_path,
            context_path=args.context_path,
        )
        with timer("bulk query by exhaustive search"):
            df = retriever.retrieve(full_ds, topk=args.topk)
            df["correct"] = False
            df["correct_index"] = None
            if not isinstance(df.loc[0, "context"], list):
                df["context"] = df["context"].map(lambda x: eval(x))
            max_index = 0.0
            for i, row in df.iterrows():
                context = row["context"]
                label = row["original_context"]
                for j, c in enumerate(context):
                    if label in c:
                        df.loc[i, "correct"] = True
                        df.loc[i, "correct_index"] = float(j)
                        if float(j) > max_index:
                            max_index = float(j)
                        break
            print(f"max_index : {int(max_index)}")
            # df["correct"] = df["original_context"] == df["context"]
            print(
                "correct retrieval result by exhaustive search",
                df["correct"].sum() / len(df),
            )
            print("mean of correct_index", df["correct_index"].mean())

    elif args.method == "dpr":
        retriever = BaseDenseRetriever(
            tokenize_fn=None,
            retriever_path=args.retriever_output_path,
            data_path=args.data_path,
            passage_path=args.passage_path,
            context_path=args.context_path,
        )
        with timer("bulk query by exhaustive search"):
            df = retriever.retrieve(full_ds, topk=args.topk)
            df["correct"] = False
            df["correct_index"] = None
            if not isinstance(df.loc[0, "context"], list):
                df["context"] = df["context"].map(lambda x: eval(x))
            max_index = 0.0
            for i, row in df.iterrows():
                context = row["context"]
                label = row["original_context"]
                for j, c in enumerate(context):
                    if label in c:
                        df.loc[i, "correct"] = True
                        df.loc[i, "correct_index"] = float(j)
                        if float(j) > max_index:
                            max_index = float(j)
                        break
            print(f"max_index : {int(max_index)}")
            # df["correct"] = df["original_context"] == df["context"]
            print(
                "correct retrieval result by exhaustive search",
                df["correct"].sum() / len(df),
            )
            print("mean of correct_index", df["correct_index"].mean())

    elif args.method == "bm25":
        retriever = BM25SparseRetrieval(
            tokenize_fn=tokenizer.tokenize,
            data_path=args.data_path,
            context_path=args.context_path,
        )
        with timer("bulk query by exhaustive search"):
            df = retriever.retrieve(full_ds, topk=args.topk)
            df["correct"] = df["original_context"] == df["context"]
            print(
                "correct retrieval result by exhaustive search",
                df["correct"].sum() / len(df),
            )

        with timer("single query by exhaustive search"):
            scores, indices = retriever.retrieve(query, topk=args.topk)