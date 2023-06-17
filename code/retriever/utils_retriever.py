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


class Preprocessor:
    def __init__(self, no_other_languages: bool = False, quoat_normalize: bool = False):
        """
        text를 전처리 하는 class입니다.

        Args:
            no_other_languages (bool, optional): 한글, 영어, 숫자, 특수문자만 남깁니다. Defaults to False.
            quoat_normalize (bool, optional): ‘’“”와 같은 따옴표를 '로 통일합니다. Defaults to False.
        """
        self.no_other_languaes = no_other_languages
        self.quoat_normalize = quoat_normalize

    def preprocess(self, text: str) -> str:
        text = re.sub(r"\\+n", " ", text)  # \\n 따위의 이상한 개행문자 제거
        if self.no_other_languaes:
            text = re.sub(
                r"([^ㄱ-ㅎㅏ-ㅣ가-힣a-zA-Z\d\s\.\,\'\"\<\>\!\@\#\$\%\^\&\*\(\)\[\]\_\+\-’‘“”《》〈〉~])",
                "",
                text,
            )
        if self.quoat_normalize:
            text = re.sub(r"[’‘“”]", "'", text)
        text = re.sub(r"(\(\)|\[\])", "", text)  # 빈 괄호 제거
        text = re.sub(r"\s+", " ", text).strip()  # 중복 공백 제거

        return text


if __name__ == "__main__":
    import json

    with open("./data/wikipedia_documents.json", mode="r", encoding="utf-8") as f:
        wiki = json.load(f)
    preprocessor = Preprocessor(True, True)
    index = "3"
    print("*****original*****")
    print(wiki[index]["text"])

    print("*****cleaned*****")
    print(preprocessor.preprocess(wiki[index]["text"]))
