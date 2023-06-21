import pandas as pd
from datasets import load_from_disk, Dataset
import re

def get_offset(context:str, answer:str):
    return context.find(answer)



def main():
    dataset=load_from_disk("./data/train_dataset")
    train_set:pd.DataFrame=dataset['train'].to_pandas()
    train_set=train_set.set_index("id")

    with open('./code/question_generation/chatgpt_train_aug.txt','r') as f:
        lines=f.read()

    qs=lines.split('\n\n\n')
    ids=[]
    qass=[]
    for question in qs:
        idqas=question.split('\n\n')
        ids.append(idqas[0])
        qass.append(idqas[1:])
    print(ids[0],qass[0])

    df=pd.DataFrame()

    id_col=[]
    doc_id_col=[]
    t_col=[]
    c_col=[]
    q_col=[]
    a_col=[]
    aug_id=0
    here=0
    for id,qas in zip(ids,qass):
        for qa in qas:
            answer_text, question=qa.split('\n')
            question,answer_text=question.split(':')[1].strip(),answer_text.split(':')[1].strip()
            context:str=train_set["context"][id]
            answer_start=context.find(answer_text)
            if answer_start==-1:
                here+=1
                continue
            answers={"answer_start":[answer_start],"text":[answer_text]}

            id_col.append(id+str(aug_id).zfill(5))
            doc_id_col.append(train_set["document_id"][id])
            t_col.append(train_set["title"][id])
            c_col.append(context)
            q_col.append(question)
            a_col.append(answers)
            aug_id+=1

    df['title']=t_col
    df['context']=c_col
    df['question']=q_col
    df['id']=id_col
    df['answers']=a_col
    df['document_id']=doc_id_col

    arrow_data=Dataset.from_pandas(df)
    arrow_data.save_to_disk('./data/question_generation/')

if __name__=="__main__":
    main()