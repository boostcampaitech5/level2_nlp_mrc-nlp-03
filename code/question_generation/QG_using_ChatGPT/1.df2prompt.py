import string
import pandas as pd
from pathlib import Path
import os

# HTML 템플릿
TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>results</title>
</head>
<body>
    $paragraphs
</body>
</html>
"""

# 문단 템플릿
PARAGRAPH_TEMPLATE = """
    <h2>
        $id <br>
        다음 예시처럼 문맥에서 중요한 단답형 정답을 고르고, 그 정답에 대해 문맥을 보고 바로 답할 수 있는 질문의 쌍들을 5개 생성하시오.
    </h2>
    <p>
    <br>
    조건 1: 예시에서 사용된 단답형 span은 제외<br>
    조건 2: 각 span 당 하나의 질문만 생성<br>
    조건 3: 단답형 정답은 문맥에서 변형 불가<br>
    (형식은 예시와 같이 정답과 질문을 구분하여 제시)<br><br>
문맥 : ```
<br>
$paragraph
<br>
```<br>
정답 예시 : $answer <br>
질문 예시 : $question <br><br>
정답1 : <br>
질문1 : <br><br>
정답2 : <br>
질문2 : <br><br>
정답3 : <br>
질문3 : <br><br>
정답4 : <br>
질문4 : <br><br>
정답5 : <br>
질문5 : 
    </p>
"""


def fill_color_text(context, answer_start, answer_text):
    blue_start='<span style="background-color: blue;color: white;">'
    span_end='</span>'

    context=context[:answer_start] + blue_start + context[answer_start:answer_start+len(answer_text)] + span_end + context[answer_start+len(answer_text):]

    return context


def df2paragraph(df:pd.DataFrame, paragraph_template):
    # 문단을 생성하여 저장할 문자열 초기화
    paragraphs_html = ""

    # 각 문단 데이터를 기반으로 문단 생성
    for i, data in df.iterrows():
        dict_data={}
        dict_data['id']=data['id']
        dict_data['answer']=data['answers']['text'][0]
        dict_data['question']=data['question']
        dict_data['paragraph']=data['context']
        paragraph_html = string.Template(paragraph_template).substitute(dict_data)
        paragraphs_html += paragraph_html

    return paragraphs_html

def df2html(
        df:pd.DataFrame,
        path: str = "./outputs/train_dataset/results.csv"
    ):
    # 최종 HTML 생성
    final_html = string.Template(TEMPLATE).substitute(paragraphs=df2paragraph(df,PARAGRAPH_TEMPLATE))

    # HTML 파일로 저장
    with open(path.replace(".csv","_prompt.html"), "w") as file:
        file.write(final_html)

def main():
    from datasets import Dataset,load_from_disk
    import pandas as pd
    data=load_from_disk('./data/train_dataset')
    train=data['train'].to_pandas()
    validation=data['validation'].to_pandas()
    df2html(train,"code/question_generation/train.csv")
    df2html(validation,"code/question_generation/validation.csv")




if __name__=="__main__":
    main()