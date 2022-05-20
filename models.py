import numpy as np

from torch import nn
from transformers import BertForMaskedLM


class BertForMaskedLMSoftmax(nn.Module):
    def __init__(self, model_name):
        super(BertForMaskedLMSoftmax, self).__init__()

        self.bert = BertForMaskedLM.from_pretrained(model_name)
        self.sm = nn.Softmax(dim=0)
        self.lsm = nn.LogSoftmax(dim=0)

    def forward(self, input_ids, attention_mask, mode=None):
        if mode in ['CLS']:
            # In this case, we do not need the masked copies (only interested in the 'CLS' token with full sentence)
            out = self.bert(input_ids=input_ids[:1], attention_mask=attention_mask, output_hidden_states=True)
            return [out[1][-1]]
        else:
            # Obtain contextual embeddings and masked probabilities for each token
            out = self.bert(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
            mask_ids = np.array(range(1, input_ids.shape[1]-1))
            logits = out[0][mask_ids, mask_ids]
            output_embs = out[1][-1][:1]
            lsm_out = self.lsm(logits)[mask_ids-1, input_ids[0, 1:-1]]

            # Normalize contextual embeddings
            output_embs /= output_embs.norm(dim=-1, keepdim=True)

            # Add information quantity to the embeddings
            output_embs[:, 1:-1, :] *= (-lsm_out).sqrt().unsqueeze(-1)

            return [output_embs]

    def get_input_embeddings(self):
        return self.bert.get_input_embeddings()
