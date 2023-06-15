import torch
import torch.nn as nn
import numpy as np 
from typing import List, Optional, Tuple, Union
from torch.nn import CrossEntropyLoss
from transformers import (AutoModel, PreTrainedModel, BertModel, RobertaModel, ElectraModel, BigBirdForQuestionAnswering)

class ReadModel(PreTrainedModel):
    def __init__(self, config):
        super().__init__(config=config)
        self.num_labels = 2
        self.model_type = config.model_type
        print(f'model_type: {self.model_type}')
        if 'roberta' in config.model_type:
            self.roberta = RobertaModel(config,add_pooling_layer=False)
            self.qa_outputs = nn.Linear(config.hidden_size, self.num_labels)
        elif 'bert' in config.model_type:
            self.bert = BertModel(config, add_pooling_layer=False)
            self.qa_outputs = nn.Linear(config.hidden_size, self.num_labels)
        elif 'electra' in config.model_type:
            self.electra = ElectraModel(config, add_pooling_layer=False)
            self.qa_outputs = nn.Linear(config.hidden_size, self.num_labels)
        elif 'big_bird' in config.model_type:
            model = BigBirdForQuestionAnswering(config, add_pooling_layer=False)
            self.bert = model.bert
            try:
                self.classifier = model.classifier
            except:
                self.qa_classifier = model.qa_classifier
                self.classifier = None
        else:
            raise Exception('model_type error')
        

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def _set_gradient_checkpointing(self, module, value=False):
        module.gradient_checkpointing = value
        
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        start_positions: Optional[torch.LongTensor] = None,
        end_positions: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        **kwargs
    ):
        if self.model_type =='roberta':
            outputs = self.roberta(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )
        elif self.model_type in ['bert', 'big_bird']:
            outputs = self.bert(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )
        elif self.model_type == 'electra':
            outputs = self.electra(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )
        else:
            raise Exception('model type error')

        sequence_output = outputs['last_hidden_state'] # [batch, seq_len, h_dim]

        if self.model_type != 'big_bird':
            logits = self.qa_outputs(sequence_output)   # [batch, seq_len, 2]
        else:
            logits = self.classifier(sequence_output) if self.classifier is not None else self.qa_classifier(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)  
        start_logits = start_logits.squeeze(-1).contiguous()  # [batch, seq_len]
        end_logits = end_logits.squeeze(-1).contiguous() # [batch, seq_len]

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        result={
            'loss':total_loss,          # torch.Size([])
            'start_logits':start_logits,   # torch.Size([batch, seq_len])
            'end_logits':end_logits,     # torch.Size([batch, seq_len])
        }
        if 'hidden_states' in outputs.keys():
            result['hidden_states'] = outputs['hidden_state'] 
        if 'attentions' in outputs.keys():
            result['attentions'] = outputs['attentions']
            
        return result