import string
import pandas as pd
from pathlib import Path
import os

# HTML 템플릿
EVAL_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>evaluation results</title>
</head>
<body>
    <h2>정답 : <span style="color: blue;">파란색</span></h2>
    <h2>예측 : <span style="color: red;">빨간색</span></h2>
    $paragraphs
</body>
</html>
"""

PRED_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>prediction results</title>
</head>
<body>
    $paragraphs
</body>
</html>
"""

# 문단 템플릿
EVAL_PARAGRAPH_TEMPLATE = """
    <h1>$id</h1>
    <h2>exact match : $is_match</h2>
    <h2>question : $question</h2>
    <h3>answer : $answer</h3>
    <h3>predict : $predict</h3>
    <p>
        $paragraph
    </p>
"""

PRED_PARAGRAPH_TEMPLATE = """
    <h1>$id</h1>
    <h2>question : $question</h2>
    <h3>predict : $predict</h3>
    <p>
        $paragraph
    </p>
"""

def eval_fill_color_text(context, pred_start, pred_text, answer_start, answer_text):
    pred_end=pred_start+len(pred_text)
    answer_end=answer_start+len(answer_text)
    if pred_start<answer_end<pred_end:
        answer_end=pred_start-1
    elif pred_start<answer_start<pred_end:
        answer_start=pred_end+1

    red_start='<span style="background-color: red;color: white;">'
    blue_start='<span style="background-color: blue;color: white;">'
    span_end='</span>'

    if pred_start==answer_start and pred_text==answer_text:
        context=context[:answer_start] + blue_start + context[answer_start:answer_start+len(answer_text)] + span_end + context[answer_start+len(answer_text):]
    elif pred_start>answer_end:
        context=context[:pred_start] + red_start + context[pred_start:pred_end] + span_end + context[pred_end:]
        context=context[:answer_start] + blue_start + context[answer_start:answer_end] + span_end + context[answer_end:]
    elif pred_end<answer_start:
        context=context[:answer_start] + blue_start + context[answer_start:answer_end] + span_end + context[answer_end:]
        context=context[:pred_start] + red_start + context[pred_start:pred_end] + span_end + context[pred_end:]

    return context

def pred_fill_color_text(context, pred_start, pred_text):
    pred_end=pred_start+len(pred_text)

    span_start='<span style="background-color: blue;color: white;">'
    span_end='</span>'

    context=context[:pred_start] + span_start + context[pred_start:pred_end] + span_end + context[pred_end:]

    return context


def eval_df2paragraph(df:pd.DataFrame, paragraph_template):
    # 문단을 생성하여 저장할 문자열 초기화
    paragraphs_html = ""

    # 각 문단 데이터를 기반으로 문단 생성
    for i, data in df.iterrows():
        dict_data={}
        dict_data['id']=data['id']
        dict_data['question']=data['question']
        dict_data['answer']=data['answer_text']
        dict_data['predict']=data['prediction_text']
        dict_data['is_match']= (data['prediction_text']==data['answer_text'])
        dict_data['paragraph']=eval_fill_color_text(
            data['context'],
            data['prediction_start'],
            data['prediction_text'],
            data['answer_start'],
            data['answer_text'],
        )
        paragraph_html = string.Template(paragraph_template).substitute(dict_data)
        paragraphs_html += paragraph_html

    return paragraphs_html

def pred_df2paragraph(df:pd.DataFrame, paragraph_template):
    # 문단을 생성하여 저장할 문자열 초기화
    paragraphs_html = ""

    # 각 문단 데이터를 기반으로 문단 생성
    for i, data in df.iterrows():
        dict_data={}
        dict_data['id']=data['id']
        dict_data['question']=data['question']
        dict_data['predict']=data['prediction_text']
        dict_data['paragraph']=pred_fill_color_text(
            data['context'],
            data['prediction_start'],
            data['prediction_text'],
        )
        paragraph_html = string.Template(paragraph_template).substitute(dict_data)
        paragraphs_html += paragraph_html

    return paragraphs_html


def eval_df2html(path: str = "./outputs/train_dataset/eval_results.csv"):
    path=Path(path)
    df = pd.read_csv(path)
    # 최종 HTML 생성
    final_html = string.Template(EVAL_TEMPLATE).substitute(paragraphs=eval_df2paragraph(df,EVAL_PARAGRAPH_TEMPLATE))

    # HTML 파일로 저장
    with open(os.path.join(path.parent,"evaluation_results.html"), "w") as file:
        file.write(final_html)

def pred_df2html(path: str = "./outputs/test_dataset/pred_results.csv"):
    path=Path(path)
    df = pd.read_csv(path)
    # 최종 HTML 생성
    final_html = string.Template(PRED_TEMPLATE).substitute(paragraphs=pred_df2paragraph(df,PRED_PARAGRAPH_TEMPLATE))

    # HTML 파일로 저장
    with open(os.path.join(path.parent,"prediction_results.html"), "w") as file:
        file.write(final_html)

if __name__=="__main__":
    eval_df2html("/opt/ml/tests/level2_nlp_mrc-nlp-03/outputs/train_dataset/2023-06-18_07:28:56/eval_results.csv")
    eval_df2html("/opt/ml/tests/level2_nlp_mrc-nlp-03/outputs/train_dataset/2023-06-18_08:16:27/eval_results.csv")
    # pred_df2html()