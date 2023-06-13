import hydra
import os
from utils_retriever import set_seed
from omegaconf import DictConfig
from DPR_model import DPR
from DPR_dataset import DPRDataset
from retriever_arguments import DataTrainingArguments, ModelArguments
from transformers import (
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)


@hydra.main(version_base="1.3", config_path="./configs", config_name="train")
def main(cfg: DictConfig):
    model_args = ModelArguments(**cfg.get("model"))
    data_args = DataTrainingArguments(**cfg.get("data"))
    training_args = TrainingArguments(**cfg.get("trainer"), remove_unused_columns=False)

    set_seed(training_args.seed)

    model = DPR(model_args.model_name_or_path, for_train=True)
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    if training_args.do_train:
        train_dataset = DPRDataset(data_args, tokenizer=tokenizer, split="train")
    else:
        train_dataset = None
    if training_args.do_eval:
        eval_dataset = DPRDataset(data_args, tokenizer=tokenizer, split="validation")
    else:
        eval_dataset = None

    trainer = Trainer(
        model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    trainer.train()
    trainer.save_model()
    trainer.save_state()
    with open(
        os.path.join(training_args.output_dir, "encoder_model_name.txt"), mode="w"
    ) as f:
        f.write(model_args.model_name_or_path)


if __name__ == "__main__":
    main()
