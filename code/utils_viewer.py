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
    <h2>exact match : $is_match</h2>
    <p>
        $paragraph
    </p>
"""

def fill_color_text(context, pred_start, pred_text, answer_start, answer_text):
    pred_end=pred_start+len(pred_text)
    answer_end=answer_start+len(answer_text)
    if pred_start<answer_end<pred_end:
        answer_end=pred_start-1
    elif pred_start<answer_start<pred_end:
        answer_start=pred_end+1

    red_start='<span style="color: red;">'
    blue_start='<span style="color: blue;">'
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


def df2paragraph(df:pd.DataFrame, paragraph_template):
    # 문단을 생성하여 저장할 문자열 초기화
    paragraphs_html = ""

    # 각 문단 데이터를 기반으로 문단 생성
    for i, data in df.iterrows():
        dict_data={}
        dict_data['id']=data['id']
        dict_data['is_match']= (data['prediction_start']==data['answer_start']) and (data['prediction_text']==data['answer_text'])
        dict_data['paragraph']=fill_color_text(
            data['context'],
            data['prediction_start'],
            data['prediction_text'],
            data['answer_start'],
            data['answer_text'],
        )
        paragraph_html = string.Template(paragraph_template).substitute(dict_data)
        paragraphs_html += paragraph_html

    return paragraphs_html


def df2html(path: str = "./outputs/train_dataset/eval_results.csv"):
    path=Path(path)
    df = pd.read_csv(path)
    # 최종 HTML 생성
    final_html = string.Template(TEMPLATE).substitute(paragraphs=df2paragraph(df,PARAGRAPH_TEMPLATE))

    # HTML 파일로 저장
    with open(os.path.join(path.parent,"evaluation_results.html"), "w") as file:
        file.write(final_html)

if __name__=="__main__":
    CSV_PATH=Path("./outputs/train_dataset/eval_results.csv")
    df2html(CSV_PATH)