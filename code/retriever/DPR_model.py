from transformers import AutoModel, AutoTokenizer
import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseEncoder(nn.Module):
    """
    Base Dense Passage Encoder Model
    """

    def __init__(self, model_checkpoint):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_checkpoint)

    def forward(self, inputs):
        outputs = self.encoder(**inputs)
        return outputs["pooler_output"]


class DPR(nn.Module):
    """
    Dense Passage Retriever
    """

    def __init__(self, model_checkpoint):
        super().__init__()
        self.q_encoder = BaseEncoder(model_checkpoint)
        self.p_encoder = BaseEncoder(model_checkpoint)

    def forward(self, **inputs):
        q_outputs = self.q_encoder(
            {
                "input_ids": inputs["q_input_ids"],
                "token_type_ids": inputs["q_token_type_ids"],
                "attention_mask": inputs["q_attention_mask"],
            }
        )  # (batch_size, emb_dim)
        p_outputs = self.p_encoder(
            {
                "input_ids": inputs["p_input_ids"],
                "token_type_ids": inputs["p_token_type_ids"],
                "attention_mask": inputs["p_attention_mask"],
            }
        )  # (batch_size, emb_dim)

        sim_scores = torch.matmul(
            q_outputs, torch.transpose(p_outputs, 0, 1)
        )  # (batch_size, batch_size)
        sim_scores = F.log_softmax(sim_scores, dim=1)
        targets = torch.arange(0, sim_scores.shape[0]).long().to(sim_scores.device)

        loss = F.nll_loss(sim_scores, targets)

        return {
            "q_outputs": q_outputs,
            "p_outputs": p_outputs,
            "sim_scores": sim_scores,
            "loss": loss,
        }

    def get_question_embedding(self, inputs):
        return self.q_encoder(inputs)

    def get_passage_embedding(self, inputs):
        return self.p_encoder(inputs)


if __name__ == "__main__":
    model = DPR("klue/bert-base")
    tokenizer = AutoTokenizer.from_pretrained("klue/bert-base")
    q = "이순신의 직업은 무엇인가?"
    p = "이순신은 조선의 장군이다."
    q_tok = tokenizer([q], return_tensors="pt")
    p_tok = tokenizer([p], return_tensors="pt")
    model_input = {
        "p_input_ids": p_tok["input_ids"],
        "p_token_type_ids": p_tok["token_type_ids"],
        "p_attention_mask": p_tok["attention_mask"],
        "q_input_ids": q_tok["input_ids"],
        "q_token_type_ids": q_tok["token_type_ids"],
        "q_attention_mask": q_tok["attention_mask"],
    }
    output = model(**model_input)
    print(output)
