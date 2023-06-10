from DPR_model import DPR
from DPR_dataset import DPRDataset
from transformers import (
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)


def main():
    model = DPR("klue/bert-base")
    tokenizer = AutoTokenizer.from_pretrained("klue/bert-base")
    dataset = DPRDataset("../data/train_dataset", tokenizer, "train", 384, 64)
    training_args = TrainingArguments(
        output_dir="../test_output",
        do_train=True,
        num_train_epochs=3,
        remove_unused_columns=False,
        per_device_train_batch_size=32,
        logging_steps=100,
    )
    trainer = Trainer(model, args=training_args, train_dataset=dataset)

    trainer.train()
    trainer.save_model()
    trainer.save_state()


if __name__ == "__main__":
    main()
