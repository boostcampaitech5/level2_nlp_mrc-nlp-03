from transformers import (
    PreTrainedTokenizerFast,
    PreTrainedTokenizer,
    BartForConditionalGeneration,
)
from datasets import (
    Dataset,
    load_from_disk,
    load_dataset,
    concatenate_datasets,
)
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from typing import Union
from functools import partial
from tqdm import tqdm
import transformers
from utils_viewer import df2html

torch.manual_seed(42)
transformers.set_seed(42)

class Collator:
    def __init__(self, tokenizer):
        self.tokenizer=tokenizer

    def __call__(self, batch):
        input_ids=[torch.LongTensor(sample['input_ids']) for sample in batch]
        input_ids=pad_sequence(input_ids,batch_first=True,padding_value=self.tokenizer.pad_token_id)
        attention_mask=[torch.LongTensor(sample['attention_mask']) for sample in batch]
        attention_mask=pad_sequence(attention_mask,batch_first=True,padding_value=self.tokenizer.pad_token_id)
        return {
            "input_ids":input_ids, 
            "attention_mask":attention_mask, 
        }

def flatten_list(example):
    example['answer']=example['answer'][0]
    example['answer_start']=example['answer_start'][0]
    return example

def preprocess(example, tokenizer:Union[PreTrainedTokenizerFast,PreTrainedTokenizer]):
    texts=example['answer']+' <unused0> '+ example['context']

    inputs=tokenizer(texts,
                     padding="longest",
                     truncation=True,
                     max_length=1024,
                     add_special_tokens=True,
                     return_token_type_ids=False)

    inputs["input_ids"] = torch.LongTensor([tokenizer.bos_token_id]+inputs['input_ids']+[tokenizer.eos_token_id])
    inputs["attention_mask"] = torch.LongTensor([1,1]+inputs['attention_mask'])
    return inputs

def main(cfg):
    device=torch.device('cuda'if torch.cuda.is_available() else 'cpu')
    tokenizer :PreTrainedTokenizerFast = PreTrainedTokenizerFast.from_pretrained(cfg.model_name)
    model :BartForConditionalGeneration = BartForConditionalGeneration.from_pretrained(cfg.model_name)
    model=model.to(device)

    dataset:dataset=load_from_disk(cfg.data_path)
    dataset=concatenate_datasets([dataset['train'], dataset['validation']])
    dataset=dataset.flatten()
    dataset=dataset.rename_columns({
        "answers.text":"answer",
        "answers.answer_start":"answer_start"
    })
    dataset=dataset.map(
        flatten_list
    )
    proc_dataset=dataset.map(
        partial(preprocess, tokenizer=tokenizer),
        remove_columns=dataset.column_names,
        num_proc=1,
        batch_size=32,
    )

    dataloader=DataLoader(
        proc_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=Collator(tokenizer)
    )

    gen_questions=[]
    for batch in tqdm(dataloader,desc="Generating Questions..."):
        batch["input_ids"]=batch["input_ids"].to(device)
        batch["attention_mask"]=batch["attention_mask"].to(device)
        question_ids = model.generate(**batch,
                                      num_beams=args.num_beams,
                                      max_length=256,
                                      eos_token_id=tokenizer.eos_token_id)
        question_ids=question_ids.detach().cpu().squeeze().tolist()
        for q in question_ids:
            new_questions=tokenizer.decode(q, skip_special_tokens=True)
            gen_questions.append(new_questions)
    dataset=dataset.add_column("new_question",gen_questions)

    csv_name='./code/question_generation/new_questions_{}.csv'
    df=dataset.to_pandas()
    df.to_csv('./code/question_generation/new_questions_{}.csv'.format(args.num_beams),index=False)
    df2html(csv_name)


def get_args():
    from argparse import ArgumentParser
    parser=ArgumentParser()
    parser.add_argument("--model_name",type=str,default="Sehong/kobart-QuestionGeneration")
    parser.add_argument("--data_path",type=str,default="./data/train_dataset")
    parser.add_argument("--batch_size",type=int,default=64)
    parser.add_argument("--num_beams",type=int,default=4)
    args=parser.parse_args()
    return args


if __name__=="__main__":
    args=get_args()
    main(args)