import random
import numpy as np
import torch
import re
from transformers import is_torch_available


def set_seed(seed: int = 42):
    """
    seed 고정하는 함수 (random, numpy, torch)

    Args:
        seed (:obj:`int`): The seed to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    if is_torch_available():
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if use multi-GPU
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def context_cleaning(context):
    context = re.sub(r"\\+n", " ", context)
    context = re.sub(r"\s+", " ", context)
    return context


def preprocess(text):
    text = re.sub(r"\n", " ", text)
    text = re.sub(r"\\n", " ", text)
    text = re.sub(r"#", " ", text)
    text = re.sub(r"([^ㄱ-ㅎㅏ-ㅣ가-힣a-zA-Z\d\s\.\,\'\"\<\>\!\@\#\$\%\^\&\*\(\)\_\+\-])", "", text) # wontaek preprocessing code
    text = re.sub(r"[^A-Za-z0-9가-힣.?!,()~‘’“”"":%&《》〈〉''㈜·\-\'+\s一-龥]", "", text)
    text = re.sub(r"\s+", " ", text).strip()  # 두 개 이상의 연속된 공백을 하나로 치환
    
    return text


if __name__ == "__main__":
    import json

    with open("./data/wikipedia_documents.json", mode="r", encoding="utf-8") as f:
        wiki = json.load(f)

    print("*****original*****")
    print(wiki["1"]["text"])
    print("*****cleaned*****")
    print(context_cleaning(wiki["1"]["text"]))
