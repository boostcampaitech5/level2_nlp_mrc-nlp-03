# coding=utf-8
# Copyright 2020 The HuggingFace Team All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Question-Answering task와 관련된 'Trainer'의 subclass 코드 입니다.
"""

from transformers import Trainer, is_datasets_available, is_torch_tpu_available
from transformers.trainer_utils import PredictionOutput
import torch
from tqdm.auto import tqdm
import pandas as pd

if is_datasets_available():
    import datasets

if is_torch_tpu_available():
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met

# Huggingface의 Trainer를 상속받아 QuestionAnswering을 위한 Trainer를 생성합니다.
class QuestionAnsweringTrainer(Trainer):
    def __init__(self, *args, eval_examples=None, post_process_function=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.eval_examples = eval_examples
        self.post_process_function = post_process_function

    def evaluate(self, eval_dataset=None, eval_examples=None, ignore_keys=None):
        eval_dataset = self.eval_dataset if eval_dataset is None else eval_dataset
        if 'token_type_ids' in eval_dataset.column_names:
            eval_dataset = eval_dataset.remove_columns('token_type_ids')
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        eval_examples = self.eval_examples if eval_examples is None else eval_examples

        # 일시적으로 metric computation를 불가능하게 한 상태이며, 해당 코드에서는 loop 내에서 metric 계산을 수행합니다.
        compute_metrics = self.compute_metrics
        self.compute_metrics = None
        try:
            # output = self.prediction_loop(
            #     eval_dataloader,
            #     description="Evaluation",
            #     # metric이 없으면 예측값을 모으는 이유가 없으므로 아래의 코드를 따르게 됩니다.
            #     # self.args.prediction_loss_only
            #     prediction_loss_only=True if compute_metrics is None else None,
            #     ignore_keys=ignore_keys,
            # )
            model = self.model
            model.eval()
            start_logits, end_logits = [], []
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            for i, samples in tqdm(enumerate(eval_dataloader), total=len(eval_dataloader), desc='evaluate'):
                input_ids = samples['input_ids'].to(device)
                attention_mask = samples['attention_mask'].to(device)
                output = model(input_ids=input_ids, attention_mask=attention_mask)
                start_logits.append(output['start_logits'].detach().cpu())
                end_logits.append(output['end_logits'].detach().cpu())
                torch.cuda.empty_cache()
            start_logits = torch.concat(start_logits)
            end_logits = torch.concat(end_logits)
            output = {'predictions':(start_logits.numpy(), end_logits.numpy())}
        finally:
            self.compute_metrics = compute_metrics

        if isinstance(eval_dataset, datasets.Dataset):
            eval_dataset.set_format(
                type=eval_dataset.format["type"],
                columns=list(eval_dataset.features.keys()),
            )

        if self.post_process_function is not None and self.compute_metrics is not None:
            eval_preds, prediction_start_pos, context = self.post_process_function(
                eval_examples, eval_dataset, output['predictions'], self.args
            )
            metrics = self.compute_metrics(eval_preds)

            self.log(metrics)
        else:
            metrics = {}
            eval_preds = None

        if self.args.tpu_metrics_debug or self.args.debug:
            # tpu-comment: PyTorch/XLA에 대한 Logging debug metrics (compile, execute times, ops, etc.)
            xm.master_print(met.metrics_report())

        self.control = self.callback_handler.on_evaluate(
            self.args, self.state, self.control, metrics
        )

        if eval_preds:
            prediction_text, answer = eval_preds
            prediction_text, answer = pd.DataFrame(prediction_text), pd.DataFrame(answer)
            prediction_start_pos=pd.DataFrame(prediction_start_pos.items(), columns=['id','prediction_start'])
            prediction=pd.merge(prediction_text, prediction_start_pos,on='id')
            answer = answer['answers'].apply(pd.Series)
            answer['answer_start']=answer['answer_start'].apply(lambda x : x[0]).astype('int32')
            answer['answer_text']=answer['text'].apply(lambda x : x[0])
            eval_preds = pd.concat([prediction, answer[['answer_start','answer_text']]], axis=1)
            eval_preds['context']=context
        return metrics, eval_preds

    def predict(self, test_dataset, test_examples, ignore_keys=None):
        if 'token_type_ids' in test_dataset.column_names:
            test_dataset = test_dataset.remove_columns('token_type_ids')
        test_dataloader = self.get_test_dataloader(test_dataset)


        # 일시적으로 metric computation를 불가능하게 한 상태이며, 해당 코드에서는 loop 내에서 metric 계산을 수행합니다.
        compute_metrics = self.compute_metrics
        self.compute_metrics = None
        try:
            # output = self.prediction_loop(
            #     test_dataloader,
            #     description="Evaluation",
            #     # metric이 없으면 예측값을 모으는 이유가 없으므로 아래의 코드를 따르게 됩니다.
            #     # self.args.prediction_loss_only
            #     prediction_loss_only=True if compute_metrics is None else None,
            #     ignore_keys=ignore_keys,
            # )
            model = self.model
            model.eval()
            start_logits, end_logits = [], []
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            for i, samples in tqdm(enumerate(test_dataloader), total=len(test_dataloader), desc='predict'):
                input_ids = samples['input_ids'].to(device)
                attention_mask = samples['attention_mask'].to(device)
                output = model(input_ids=input_ids, attention_mask=attention_mask)
                start_logits.append(output['start_logits'].detach().cpu())
                end_logits.append(output['end_logits'].detach().cpu())
                torch.cuda.empty_cache()
            start_logits = torch.concat(start_logits)
            end_logits = torch.concat(end_logits)
            output = {'predictions':(start_logits.numpy(), end_logits.numpy())}
        finally:
            self.compute_metrics = compute_metrics

        if self.post_process_function is None or self.compute_metrics is None:
            return output

        if isinstance(test_dataset, datasets.Dataset):
            test_dataset.set_format(
                type=test_dataset.format["type"],
                columns=list(test_dataset.features.keys()),
            )

        predictions = self.post_process_function(
            test_examples, test_dataset, output['predictions'], self.args
        )
        return predictions
