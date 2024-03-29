import torch
import json
import pandas as pd
from datasets import load_from_disk
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
from tqdm.auto import tqdm

try:
    from retriever.retriever_arguments import DataTrainingArguments
except ImportError:
    from retriever_arguments import DataTrainingArguments


class DPRDataset(Dataset):
    def __init__(
        self,
        data_args: DataTrainingArguments,
        tokenizer: PreTrainedTokenizer,
        split: str,
    ):
        dataset = load_from_disk(data_args.dataset_name)[split]
        self.dataset = dataset.map(
            self.get_tokenized_passage,
            fn_kwargs={
                "tokenizer": tokenizer,
                "max_passage_length": data_args.max_passage_length,
                "max_question_length": data_args.max_question_length,
            },
            remove_columns=dataset.column_names,
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def get_tokenized_passage(
        self, example, tokenizer, max_passage_length, max_question_length
    ):
        """
        context에서 정답을 포함하는 passage만 tokenize해서 반환합니다.
        context가 max_passage_length보다 길 경우 정답 토큰 앞 뒤로 truncation합니다.
        question도 tokenize해서 같이 반환합니다.
        """

        max_passage_length -= 2  # for special tokens
        context = example["context"]
        question = example["question"]
        tokenized_context = tokenizer(
            context,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        tokenized_question = tokenizer(
            question,
            max_length=max_question_length,
            padding="max_length",
            truncation=True,
        )
        answer_char = example["answers"]["text"]
        start_char = example["answers"]["answer_start"][0]
        end_char = start_char + len(answer_char)
        offset_mapping = tokenized_context.pop("offset_mapping")
        len_tokens = len(offset_mapping)

        def add_special_tokens(tokenized):
            """
            passage를 special token 없이 tokenize했으므로 앞에 [CLS] 토큰을 붙이고 뒤에 [SEP]토큰을 붙여줍니다.
            max_length까지 [PAD]토큰을 붙입니다.
            """

            # 빈 string을 tokenize 했으므로 [CLS][SEP]만 나옵니다.
            special_tokens = tokenizer("")

            input_ids = (
                [special_tokens["input_ids"][0]]
                + tokenized["input_ids"]
                + [special_tokens["input_ids"][1]]
            )
            token_type_ids = (
                [special_tokens["token_type_ids"][0]]
                + tokenized["token_type_ids"]
                + [special_tokens["token_type_ids"][1]]
            )
            attention_mask = (
                [special_tokens["attention_mask"][0]]
                + tokenized["attention_mask"]
                + [special_tokens["attention_mask"][1]]
            )

            tokenized = {
                "input_ids": input_ids,
                "token_type_ids": token_type_ids,
                "attention_mask": attention_mask,
            }
            tokenized = tokenizer.pad(
                tokenized, max_length=max_passage_length + 2, padding="max_length"
            )

            return tokenized

        # context가 이미 max_passage_length보다 짧은 경우 그대로 반환합니다.
        if len_tokens <= max_passage_length:
            tokenized_context = add_special_tokens(tokenized_context)
            return {
                "p_input_ids": tokenized_context["input_ids"],
                "p_token_type_ids": tokenized_context["token_type_ids"],
                "p_attention_mask": tokenized_context["attention_mask"],
                "q_input_ids": tokenized_question["input_ids"],
                "q_token_type_ids": tokenized_question["token_type_ids"],
                "q_attention_mask": tokenized_question["attention_mask"],
            }

        # train.py의 prepare_train_features 함수와 같은 로직으로 정답 토큰을 찾습니다.
        token_start_index = 0
        token_end_index = len_tokens - 1

        while (
            offset_mapping[token_start_index][0] <= start_char
            and token_start_index < len_tokens
        ):
            token_start_index += 1
        token_start_index -= 1

        while offset_mapping[token_end_index][1] >= end_char and token_end_index >= 0:
            token_end_index -= 1
        token_end_index += 1

        passage_mid_index = (token_start_index + token_end_index) // 2

        passage_start_index = passage_mid_index - (
            max_passage_length // 2 + max_passage_length % 2
        )
        passage_end_index = passage_mid_index + (max_passage_length // 2)

        # start index나 end index가 context를 벗어나는 경우 context 안으로 다시 옮기고, 반대편 index를 옮겨서 최종적인 passage의 길이를 유지합니다.
        # 예를 들어, max_passage length가 10인데, 정답 토큰이 0번이어서 start index가 -5, end index가 4였다면 start index를 0, end index를 9로 옮깁니다.
        if passage_start_index < 0:
            passage_end_index += -passage_start_index
            passage_start_index = 0

        if passage_end_index >= len_tokens:
            passage_start_index -= passage_end_index - (len_tokens - 1)
            passage_end_index = len_tokens - 1

        tokenized_context = add_special_tokens(
            {
                k: v[passage_start_index:passage_end_index]
                for k, v in tokenized_context.items()
            }
        )

        return {
            "p_input_ids": tokenized_context["input_ids"],
            "p_token_type_ids": tokenized_context["token_type_ids"],
            "p_attention_mask": tokenized_context["attention_mask"],
            "q_input_ids": tokenized_question["input_ids"],
            "q_token_type_ids": tokenized_question["token_type_ids"],
            "q_attention_mask": tokenized_question["attention_mask"],
        }


if __name__ == "__main__":
    from transformers import AutoTokenizer
    from retriever_arguments import DataTrainingArguments

    tokenizer = AutoTokenizer.from_pretrained("klue/bert-base")
    args = DataTrainingArguments()
    dataset = DPRDataset(args, tokenizer, "validation")

    print(dataset[0])
