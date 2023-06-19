import string
import pandas as pd
from pathlib import Path
import os

# HTML 템플릿
TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>evaluation results</title>
</head>
<body>
    $paragraphs
</body>
</html>
"""

# 문단 템플릿
PARAGRAPH_TEMPLATE = """
    <h1>$id</h1>
    <h2>title : $title</h2>
    <h2>answer : $answer</h2>
    <h3>question : $question</h3>
    <h3>new_question : $new_question</h3>
    <p>
        $paragraph
    </p>
"""


def fill_color_text(context, answer_start, answer_text):
    blue_start='<span style="background-color: blue;color: white;">'
    span_end='</span>'

    # if isinstance(answer_start,list):
    #     answer_start=answer_start[0]
    # if isinstance(answer_text,list):
    #     answer_text=answer_text[0]

    context=context[:answer_start] + blue_start + context[answer_start:answer_start+len(answer_text)] + span_end + context[answer_start+len(answer_text):]

    return context


def df2paragraph(df:pd.DataFrame, paragraph_template):
    # 문단을 생성하여 저장할 문자열 초기화
    paragraphs_html = ""

    # 각 문단 데이터를 기반으로 문단 생성
    for i, data in df.iterrows():
        dict_data={}
        dict_data['id']=data['id']
        dict_data['title']=data['title']
        dict_data['answer']=data['answer']
        dict_data['question']=data['question']
        dict_data['new_question']=data['new_question']
        dict_data['paragraph']=fill_color_text(data['context'],data['answer_start'],data['answer'])
        paragraph_html = string.Template(paragraph_template).substitute(dict_data)
        paragraphs_html += paragraph_html

    return paragraphs_html

def df2html(path: str = "./outputs/train_dataset/results.csv"):
    df = pd.read_csv(path)
    # 최종 HTML 생성
    final_html = string.Template(TEMPLATE).substitute(paragraphs=df2paragraph(df,PARAGRAPH_TEMPLATE))

    # HTML 파일로 저장
    with open(path.replace(".csv",".html"), "w") as file:
        file.write(final_html)



if __name__=="__main__":
    df2html("/opt/ml/tests/level2_nlp_mrc-nlp-03/code/question_generation/new_questions_2.csv")
