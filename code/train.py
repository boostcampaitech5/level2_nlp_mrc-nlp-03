import logging
import os
import sys

from arguments import DataTrainingArguments, ModelArguments
from datasets import DatasetDict, load_from_disk, concatenate_datasets, load_dataset
from models import ReadModel, MultiReadModel
import evaluate
from trainer_qa import QuestionAnsweringTrainer
from transformers import (
    AutoConfig,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    BertTokenizerFast,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    TrainingArguments,
    set_seed,
)
from utils_qa import check_no_error, postprocess_qa_predictions
import yaml
from pathlib import Path
import hydra
from omegaconf import DictConfig
from datetime import datetime, timedelta, timezone
import wandb
from utils_viewer import eval_df2html

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_SILENT"] = "true"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"
logger = logging.getLogger(__name__)


@hydra.main(version_base="1.3", config_path="../configs", config_name="train")
def main(cfg: DictConfig):
    # 가능한 arguments 들은 ./arguments.py 나 transformer package 안의 src/transformers/training_args.py 에서 확인 가능합니다.
    # --help flag 를 실행시켜서 확인할 수 도 있습니다.

    # Argument 정의된 dataclass들을 instantiate
    model_args = ModelArguments(**cfg.get("model"))
    data_args = DataTrainingArguments(**cfg.get("data"))
    training_args = TrainingArguments(**cfg.get("trainer"))

    dirs = os.listdir(training_args.output_dir)
    for dir in dirs:  # 빈 디렉토리 삭제
        path = os.path.join(training_args.output_dir, dir)
        if not os.path.exists(path):
            os.rmdir(path)
    run_name = datetime.now(timezone(timedelta(hours=9))).strftime("%Y-%m-%d_%H:%M:%S")
    training_args.run_name = run_name
    training_args.output_dir = os.path.join(training_args.output_dir, run_name)
    wandb.init(project="MRC", entity=None, name=run_name)
    wandb.config.update(
        {
            "data_path": data_args.dataset_name,
            "max_seq_length": data_args.max_seq_length,
            "doc_stride": data_args.doc_stride,
        }
    )

    print(model_args.model_name_or_path)
    print(f"model is from {model_args.model_name_or_path}")
    print(f"data is from {data_args.dataset_name}")

    # logging 설정
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -    %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # verbosity 설정 : Transformers logger의 정보로 사용합니다 (on main process only)
    # logger.info("Training/evaluation parameters %s", training_args)

    # 모델을 초기화하기 전에 난수를 고정합니다.
    set_seed(training_args.seed)
    if os.path.exists(data_args.dataset_name):
        datasets = load_from_disk(data_args.dataset_name)
    else:
        datasets = load_dataset(data_args.dataset_name)

    print(datasets)

    # AutoConfig를 이용하여 pretrained model 과 tokenizer를 불러옵니다.
    # argument로 원하는 모델 이름을 설정하면 옵션을 바꿀 수 있습니다.
    config = AutoConfig.from_pretrained(
        model_args.config_name
        if model_args.config_name is not None
        else model_args.model_name_or_path,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name
        if model_args.tokenizer_name is not None
        else model_args.model_name_or_path,
        use_fast=True,
    )
    model = MultiReadModel.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        tokenizer=tokenizer,
    )

    # print(
    #     type(training_args),
    #     type(model_args),
    #     type(datasets),
    #     type(tokenizer),
    #     type(model),
    # )

    # do_train mrc model 혹은 do_eval mrc model
    if training_args.do_train or training_args.do_eval:
        run_mrc(data_args, training_args, model_args, datasets, tokenizer, model)

    wandb.finish()


def run_mrc(
    data_args: DataTrainingArguments,
    training_args: TrainingArguments,
    model_args: ModelArguments,
    datasets: DatasetDict,
    tokenizer,
    model,
) -> None:
    # dataset을 전처리합니다.
    # training과 evaluation에서 사용되는 전처리는 아주 조금 다른 형태를 가집니다.
    if training_args.do_train:
        column_names = datasets["train"].column_names
    else:
        column_names = datasets["validation"].column_names

    title_column_name = "title" if "title" in column_names else column_names[0]
    context_column_name = "context" if "context" in column_names else column_names[1]
    question_column_name = "question" if "question" in column_names else column_names[2]
    answer_column_name = "answers" if "answers" in column_names else column_names[4]

    # 오류가 있는지 확인합니다.
    last_checkpoint, max_seq_length = check_no_error(
        data_args, training_args, datasets, tokenizer
    )
    print(f"MAX_LENGTH: {max_seq_length}")

    # Train preprocessing / 전처리를 진행합니다.
    def prepare_train_features(examples):
        # truncation과 padding(length가 짧을때만)을 통해 toknization을 진행하며, stride를 이용하여 overflow를 유지합니다.
        # 각 example들은 이전의 context와 조금씩 겹치게됩니다.
        if data_args.add_title:
            examples[question_column_name] = [
                f"{question} {tokenizer.sep_token} ^{title}^"
                for question, title in zip(
                    examples[question_column_name], examples[title_column_name]
                )
            ]
        tokenized_examples = tokenizer(
            examples[question_column_name],
            examples[context_column_name],
            # https://huggingface.co/docs/transformers/pad_truncation#:~:text=The%20truncation%20argument%20controls%20truncation.%20It%20can%20be%20a%20boolean%20or%20a%20string%3A
            truncation="only_second",
            max_length=max_seq_length,
            stride=data_args.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            # roberta모델을 사용할 경우 False, bert를 사용할 경우 True로 표기해야합니다.
            return_token_type_ids=False if "roberta" in model.model_type else True,
            padding="max_length" if data_args.pad_to_max_length else False,
        )

        # 길이가 긴 context가 등장할 경우 truncate를 진행해야하므로, 해당 데이터셋을 찾을 수 있도록 mapping 가능한 값이 필요합니다.
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
        # token의 캐릭터 단위 position를 찾을 수 있도록 offset mapping을 사용합니다.
        # start_positions과 end_positions을 찾는데 도움을 줄 수 있습니다.
        offset_mapping = tokenized_examples.pop("offset_mapping")

        # 데이터셋에 "start position", "enc position" label을 부여합니다.
        tokenized_examples["start_positions"] = []
        tokenized_examples["end_positions"] = []

        for i, offsets in enumerate(offset_mapping):
            input_ids = tokenized_examples["input_ids"][i]
            cls_index = input_ids.index(tokenizer.cls_token_id)  # cls index

            # sequence id를 설정합니다 (to know what is the context and what is the question).
            sequence_ids = tokenized_examples.sequence_ids(i)

            # 하나의 example이 여러개의 span을 가질 수 있습니다.
            sample_index = sample_mapping[i]
            answers = examples[answer_column_name][sample_index]

            # answer가 없을 경우 cls_index를 answer로 설정합니다(== example에서 정답이 없는 경우 존재할 수 있음).
            if len(answers["answer_start"]) == 0:
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                # text에서 정답의 Start/end character index
                start_char = answers["answer_start"][0]
                end_char = start_char + len(answers["text"][0])

                # text에서 current span의 Start token index
                token_start_index = 0
                while sequence_ids[token_start_index] != 1:
                    token_start_index += 1

                # text에서 current span의 End token index
                token_end_index = len(input_ids) - 1
                while sequence_ids[token_end_index] != 1:
                    token_end_index -= 1

                # 정답이 span을 벗어났는지 확인합니다(정답이 없는 경우 CLS index로 label되어있음).
                if start_char == 0 and end_char == 0:
                    tokenized_examples["start_positions"].append(cls_index)
                    tokenized_examples["end_positions"].append(cls_index)
                elif not (
                    offsets[token_start_index][0] <= start_char
                    and offsets[token_end_index][1] >= end_char
                ):
                    tokenized_examples["start_positions"].append(cls_index)
                    tokenized_examples["end_positions"].append(cls_index)
                else:
                    # token_start_index 및 token_end_index를 answer의 끝으로 이동합니다.
                    # Note: answer가 마지막 단어인 경우 last offset을 따라갈 수 있습니다(edge case).
                    while (
                        token_start_index < len(offsets)
                        and offsets[token_start_index][0] <= start_char
                    ):
                        token_start_index += 1
                    tokenized_examples["start_positions"].append(token_start_index - 1)
                    while offsets[token_end_index][1] >= end_char:
                        token_end_index -= 1
                    tokenized_examples["end_positions"].append(token_end_index + 1)

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
                tokenized_examples["start_positions"][i] -= 1
                tokenized_examples["end_positions"][i] -= 1

        return tokenized_examples

    if training_args.do_train:
        if "train" not in datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = datasets["train"]

        b = len(train_dataset)
        # dataset에서 train feature를 생성합니다.
        train_dataset = train_dataset.map(
            prepare_train_features,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
        )
        print(f"Train Examples {b} -> {len(train_dataset)}")

    # Validation preprocessing
    def prepare_validation_features(examples):
        # truncation과 padding(length가 짧을때만)을 통해 toknization을 진행하며, stride를 이용하여 overflow를 유지합니다.
        # 각 example들은 이전의 context와 조금씩 겹치게됩니다.
        if data_args.add_title:
            examples[question_column_name] = [
                f"{question} {tokenizer.sep_token} ^{title}^"
                for question, title in zip(
                    examples[question_column_name], examples[title_column_name]
                )
            ]
        tokenized_examples = tokenizer(
            examples[question_column_name],
            examples[context_column_name],
            truncation="only_second",
            max_length=max_seq_length,
            stride=data_args.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            # roberta모델을 사용할 경우 False, bert를 사용할 경우 True로 표기해야합니다.
            return_token_type_ids=False if "roberta" in model.model_type else True,
            padding="max_length" if data_args.pad_to_max_length else False,
        )

        # 길이가 긴 context가 등장할 경우 truncate를 진행해야하므로, 해당 데이터셋을 찾을 수 있도록 mapping 가능한 값이 필요합니다.
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

        # evaluation을 위해, prediction을 context의 substring으로 변환해야합니다.
        # corresponding example_id를 유지하고 offset mappings을 저장해야합니다.
        tokenized_examples["example_id"] = []
        tokenized_examples["context_index"] = []

        for i in range(len(tokenized_examples["input_ids"])):
            input_ids = tokenized_examples["input_ids"][i]
            # sequence id를 설정합니다 (to know what is the context and what is the question).
            sequence_ids = tokenized_examples.sequence_ids(i)

            # 하나의 example이 여러개의 span을 가질 수 있습니다.
            sample_index = sample_mapping[i]
            tokenized_examples["example_id"].append(examples["id"][sample_index])
            tokenized_examples["context_index"].append(0)

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

    if training_args.do_eval:
        eval_dataset = datasets["validation"]
        b = len(eval_dataset)
        # Validation Feature 생성
        eval_dataset = eval_dataset.map(
            prepare_validation_features,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
        )
        print(f"Examples {b} -> {len(eval_dataset)}")
    # Data collator
    # flag가 True이면 이미 max length로 padding된 상태입니다.
    # 그렇지 않다면 data collator에서 padding을 진행해야합니다.
    data_collator = DataCollatorWithPadding(
        tokenizer, pad_to_multiple_of=max_seq_length
    )

    # Post-processing:
    def post_processing_function(examples, features, predictions, training_args):
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

        references = [
            {"id": ex["id"], "answers": ex[answer_column_name]}
            for ex in datasets["validation"]
        ]

        return (
            EvalPrediction(predictions=formatted_predictions, label_ids=references),
            start_prediction_pos,
            context,
            question,
        )

    metric = evaluate.load("squad")

    def compute_metrics(p: EvalPrediction):
        return metric.compute(predictions=p.predictions, references=p.label_ids)

    # Trainer 초기화
    trainer = QuestionAnsweringTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        eval_examples=datasets["validation"] if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        post_process_function=post_processing_function,
        compute_metrics=compute_metrics,
    )

    # Training
    if training_args.do_train:
        if last_checkpoint is not None:
            checkpoint = last_checkpoint
        elif os.path.isdir(model_args.model_name_or_path):
            checkpoint = model_args.model_name_or_path
        else:
            checkpoint = None
        train_result = trainer.train(resume_from_checkpoint=None)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics
        metrics["train_samples"] = len(train_dataset)

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

        output_train_file = os.path.join(training_args.output_dir, "train_results.txt")

        with open(output_train_file, "w") as writer:
            logger.info("***** Train results *****")
            for key, value in sorted(train_result.metrics.items()):
                logger.info(f"  {key} = {value}")
                writer.write(f"{key} = {value}\n")

        # State 저장
        trainer.state.save_to_json(
            os.path.join(training_args.output_dir, "trainer_state.json")
        )

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics, eval_preds = trainer.evaluate()
        metrics["eval_samples"] = len(eval_dataset)

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

        csv_path = os.path.join(training_args.output_dir, "eval_results.csv")
        eval_preds.to_csv(csv_path, index=False)
        eval_df2html(csv_path)


if __name__ == "__main__":
    main()
