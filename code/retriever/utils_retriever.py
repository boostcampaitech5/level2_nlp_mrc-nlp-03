import random
import numpy as np
import torch
import wandb
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


class WandbCallback:
    def __init__(self, eval_dataset):
        self.eval_dataset = eval_dataset

    def __call__(self, trainer, model, **kwargs):
        metrics = trainer.evaluate(eval_dataset=self.eval_dataset)
        wandb.log(metrics)
