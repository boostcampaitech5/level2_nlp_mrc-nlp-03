"""
Open-Domain Question Answering 을 수행하는 inference 코드 입니다.

대부분의 로직은 train.py 와 비슷하나 retrieval, predict 부분이 추가되어 있습니다.
"""


import logging
import sys, os
from typing import Callable, Dict, List, Tuple
from collections import defaultdict

sys.path.append("code/retriever")
import numpy as np
from models import ReadModel, MultiReadModel
from arguments import DataTrainingArguments, ModelArguments
from datasets import Dataset, DatasetDict, Features, Sequence, Value, load_from_disk
from datetime import datetime, timedelta, timezone
import evaluate
from retriever.retrieval import TfidfRetriever, FaissRetriever, BaseDenseRetriever
from retriever.utils_retriever import Preprocessor
from trainer_qa import QuestionAnsweringTrainer
from transformers import (
    AutoConfig,
    AutoModelForQuestionAnswering,
    BertTokenizerFast,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    TrainingArguments,
    set_seed,
    PreTrainedTokenizer,
)
from utils_qa import check_no_error, postprocess_qa_predictions
import hydra
from omegaconf import DictConfig
from utils_viewer import pred_df2html

logger = logging.getLogger(__name__)


@hydra.main(version_base="1.3", config_path="../configs", config_name="inference")
def main(cfg: DictConfig):
    # 가능한 arguments 들은 ./arguments.py 나 transformer package 안의 src/transformers/training_args.py 에서 확인 가능합니다.
    # --help flag 를 실행시켜서 확인할 수 도 있습니다.

    # Argument 정의된 dataclass들을 instanciate
    model_args = ModelArguments(**cfg.get("model"))
    data_args = DataTrainingArguments(**cfg.get("data"))
    training_args = TrainingArguments(**cfg.get("trainer"))
    run_name = datetime.now(timezone(timedelta(hours=9))).strftime("%Y-%m-%d_%H:%M:%S")
    training_args.output_dir = os.path.join(training_args.output_dir, run_name)

    # training_args.do_train = True

    print(f"model is from {model_args.model_name_or_path}")
    print(f"data is from {data_args.dataset_name}")

    # logging 설정
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # verbosity 설정 : Transformers logger의 정보로 사용합니다 (on main process only)
    # logger.info("Training/evaluation parameters %s", training_args)

    # 모델을 초기화하기 전에 난수를 고정합니다.
    set_seed(training_args.seed)

    datasets = load_from_disk(data_args.dataset_name)
    print(datasets)

    # AutoConfig를 이용하여 pretrained model 과 tokenizer를 불러옵니다.
    # argument로 원하는 모델 이름을 설정하면 옵션을 바꿀 수 있습니다.
    config = AutoConfig.from_pretrained(
        model_args.config_name
        if model_args.config_name
        else model_args.model_name_or_path,
    )
    tokenizer = BertTokenizerFast.from_pretrained(
        model_args.tokenizer_name
        if model_args.tokenizer_name is not None
        else model_args.model_name_or_path,
        use_fast=True,
    )
    model = MultiReadModel.from_pretrained(
        model_args.model_name_or_path,
        config=config,
    )
    tokenizer_for_retriever = AutoTokenizer.from_pretrained(
        data_args.tokenizer_for_retriever
    )

    preprocessor = Preprocessor(
        no_other_languages=data_args.no_other_languages,
        quoat_normalize=data_args.quoat_normalize,
    )

    tokenizer_ret = BertTokenizerFast.from_pretrained(
        "KoichiYasuoka/roberta-base-korean-morph-upos"
    )
    # True일 경우 : run passage retrieval
    if data_args.eval_retrieval:
        datasets = run_retrieval(
            tokenizer_for_retriever.tokenize,
            preprocessor,
            datasets,
            cfg,
            training_args,
            data_args,
        )

    # eval or predict mrc model
    if training_args.do_eval or training_args.do_predict:
        run_mrc(data_args, training_args, model_args, datasets, tokenizer, model)


def run_retrieval(
    tokenize_fn: Callable[[str], List[str]],
    preprocessor: Preprocessor,
    datasets: DatasetDict,
    cfg: DictConfig,
    training_args: TrainingArguments,
    data_args: DataTrainingArguments,
    data_path: str = "./data",
    context_path: str = "wikipedia_documents.json",
) -> DatasetDict:
    # configs/retriever 에 yaml로 정의된 retriever를 instantiate
    # 바꾸고싶으면 configs/inference.yaml 의 retriever 키의 value를 바꾸면 됨.
    retriever = hydra.utils.instantiate(
        cfg.retriever, tokenize_fn=tokenize_fn, preprocessor=preprocessor
    )

    # Query에 맞는 Passage들을 Retrieval 합니다.
    df = retriever.retrieve(datasets["validation"], topk=data_args.top_k_retrieval)

    # test data 에 대해선 정답이 없으므로 id question context 로만 데이터셋이 구성됩니다.
    if training_args.do_predict:
        f = Features(
            {
                "title": Sequence(
                    feature=Value(dtype="string", id=None), length=-1, id=None
                ),
                "context": Sequence(
                    feature=Value(dtype="string", id=None), length=-1, id=None
                ),
                "question": Value(dtype="string", id=None),
                "id": Value(dtype="string", id=None),
            }
        )
        df = df[["title", "context", "question", "id"]]

    # train data 에 대해선 정답이 존재하므로 id question context answer 로 데이터셋이 구성됩니다.
    elif training_args.do_eval:
        f = Features(
            {
                "title": Value(dtype="string", id=None),
                "context": Value(dtype="string", id=None),
                "answers": Sequence(
                    feature={
                        "text": Value(dtype="string", id=None),
                        "answer_start": Value(dtype="int32", id=None),
                    },
                    length=-1,
                    id=None,
                ),
                "question": Value(dtype="string", id=None),
                "id": Value(dtype="string", id=None),
            }
        )
        df = df[["title", "context", "answers", "question", "id"]]
    datasets = DatasetDict({"validation": Dataset.from_pandas(df, features=f)})
    return datasets


def run_mrc(
    data_args: DataTrainingArguments,
    training_args: TrainingArguments,
    model_args: ModelArguments,
    datasets: DatasetDict,
    tokenizer: PreTrainedTokenizer,
    model,
) -> None:
    # eval 혹은 prediction에서만 사용함
    column_names = datasets["validation"].column_names

    title_column_name = "title" if "title" in column_names else column_names[0]
    context_column_name = "context" if "context" in column_names else column_names[1]
    question_column_name = "question" if "question" in column_names else column_names[2]
    answer_column_name = "answers" if "answers" in column_names else column_names[3]

    # 오류가 있는지 확인합니다.
    last_checkpoint, max_seq_length = check_no_error(
        data_args, training_args, datasets, tokenizer
    )

    # Validation preprocessing / 전처리를 진행합니다.
    def prepare_validation_features(examples):
        # truncation과 padding(length가 짧을때만)을 통해 tokenization을 진행하며, stride를 이용하여 overflow를 유지합니다.
        # 각 example들은 이전의 context와 조금씩 겹치게됩니다.
        tokenized_examples = defaultdict(list)
        for i in range(len(examples[question_column_name])):
            question = examples[question_column_name][i]
            titles = examples[title_column_name][i]
            contexts = examples[context_column_name][i]
            for j in range(len(contexts)):
                tokenized = tokenizer(
                    f"{question} {tokenizer.sep_token} ^{titles[j]}^"
                    if data_args.add_title
                    else question,
                    contexts[j],
                    truncation="only_second",
                    max_length=max_seq_length,
                    stride=data_args.doc_stride,
                    return_overflowing_tokens=True,
                    return_offsets_mapping=True,
                    # roberta모델을 사용할 경우 False, bert를 사용할 경우 True로 표기해야합니다.
                    return_token_type_ids=False
                    if "roberta" in model.model_type
                    else True,
                    padding="max_length" if data_args.pad_to_max_length else False,
                )

                tokenized["example_id"] = [examples["id"][i]] * len(
                    tokenized["input_ids"]
                )
                tokenized["context_index"] = [j] * len(tokenized["input_ids"])
                tokenized["sequence_ids"] = []
                for k in range(len(tokenized["input_ids"])):
                    tokenized["sequence_ids"].append(tokenized.sequence_ids(k))

                for k, v in tokenized.items():
                    tokenized_examples[k].extend(v)

        # 길이가 긴 context가 등장할 경우 truncate를 진행해야하므로, 해당 데이터셋을 찾을 수 있도록 mapping 가능한 값이 필요합니다.
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
        list_sequence_ids = tokenized_examples.pop("sequence_ids")

        for i in range(len(tokenized_examples["input_ids"])):
            input_ids = tokenized_examples["input_ids"][i]
            # sequence id를 설정합니다 (to know what is the context and what is the question).
            sequence_ids = list_sequence_ids[i]

            # Set to None the offset_mapping을 None으로 설정해서 token position이 context의 일부인지 쉽게 판별 할 수 있습니다.
            tokenized_examples["offset_mapping"][i] = [
                (o if sequence_ids[k] == 1 else None)
                for k, o in enumerate(tokenized_examples["offset_mapping"][i])
            ]

            if data_args.add_title:
                # 현재는 [question] [SEP] [title] [SEP] [context] 형태고, token type ids도 순서대로 [0, 0, 0, 0, 1]입니다.
                # [question] [SEP] [title] [context]로 바꾸고, token type ids도 [0, 0, 1, 1]로 바꾸어줍니다.
                sep_token_indeces = []
                for idx, id in enumerate(input_ids):
                    if id == tokenizer.sep_token_id:
                        sep_token_indeces.append(idx)

                assert (
                    len(sep_token_indeces) == 3
                ), "title을 추가하는데 문제가 발생했습니다. 예기치 못한 학습이 이루어질 수 있으니 보고 바랍니다."

                if "token_type_ids" in tokenized_examples.keys():
                    for idx in range(sep_token_indeces[0] + 1, sep_token_indeces[1]):
                        tokenized_examples["token_type_ids"][i][idx] = 1

                    del tokenized_examples["token_type_ids"][i][sep_token_indeces[1]]

                del tokenized_examples["input_ids"][i][sep_token_indeces[1]]
                del tokenized_examples["attention_mask"][i][sep_token_indeces[1]]
                del tokenized_examples["offset_mapping"][i][sep_token_indeces[1]]
        return tokenized_examples

    eval_dataset = datasets["validation"]

    # Validation Feature 생성
    eval_dataset = eval_dataset.map(
        prepare_validation_features,
        batched=True,
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=column_names,
        load_from_cache_file=not data_args.overwrite_cache,
    )

    # Data collator
    # flag가 True이면 이미 max length로 padding된 상태입니다.
    # 그렇지 않다면 data collator에서 padding을 진행해야합니다.
    data_collator = DataCollatorWithPadding(
        tokenizer, max_length=data_args.max_seq_length
    )

    # Post-processing:
    def post_processing_function(
        examples,
        features,
        predictions: Tuple[np.ndarray, np.ndarray],
        training_args: TrainingArguments,
    ) -> EvalPrediction:
        # Post-processing: start logits과 end logits을 original context의 정답과 match시킵니다.
        (
            predictions,
            start_prediction_pos,
            context,
            question,
        ) = postprocess_qa_predictions(
            examples=examples,
            features=features,
            predictions=predictions,
            max_answer_length=data_args.max_answer_length,
            output_dir=training_args.output_dir,
        )

        # Metric을 구할 수 있도록 Format을 맞춰줍니다.
        formatted_predictions = [
            {"id": k, "prediction_text": v} for k, v in predictions.items()
        ]
        start_prediction_pos = [
            {"id": k, "prediction_start": v} for k, v in start_prediction_pos.items()
        ]

        return formatted_predictions, start_prediction_pos, context, question

    metric = evaluate.load("squad")

    def compute_metrics(p: EvalPrediction) -> Dict:
        return metric.compute(predictions=p.predictions, references=p.label_ids)

    print("init trainer...")
    # Trainer 초기화
    trainer = QuestionAnsweringTrainer(
        model=model,
        args=training_args,
        train_dataset=None,
        eval_dataset=eval_dataset,
        eval_examples=datasets["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        post_process_function=post_processing_function,
        compute_metrics=compute_metrics,
    )

    logger.info("*** Evaluate ***")

    #### eval dataset & eval example - predictions.json 생성됨
    if training_args.do_predict:
        prediction_results = trainer.predict(
            test_dataset=eval_dataset, test_examples=datasets["validation"]
        )

        # predictions.json 은 postprocess_qa_predictions() 호출시 이미 저장됩니다.
        print(
            "No metric can be presented because there is no correct answer given. Job done!"
        )

        csv_path = os.path.join(training_args.output_dir, "pred_results.csv")
        prediction_results.to_csv(csv_path, index=False)
        pred_df2html(csv_path)

    if training_args.do_eval:
        metrics, eval_preds = trainer.evaluate()
        metrics["eval_samples"] = len(eval_dataset)

        trainer.log_metrics("test", metrics)
        trainer.save_metrics("test", metrics)

        eval_preds.to_csv(
            os.path.join(training_args.output_dir, "eval_results.csv"), index=False
        )


if __name__ == "__main__":
    main()
