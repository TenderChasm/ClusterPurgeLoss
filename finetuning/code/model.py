import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss


class RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size*2, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, 2)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = x.reshape(-1,x.size(-1)*2)
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x
        
class Model(nn.Module):   
    def __init__(self, encoder,config,tokenizer,args):
        super(Model, self).__init__()
        self.encoder = encoder
        self.config=config
        self.tokenizer=tokenizer
        self.classifier=RobertaClassificationHead(config)
        self.args=args
        self.query = 0
    
        
    def forward(self, inputs_ids_1, inputs_ids_2, labels): 
        bs,l=inputs_ids_1.size()
        input_ids=torch.cat((inputs_ids_1.unsqueeze(1),inputs_ids_2.unsqueeze(1)),1).view(bs*2,l)
        outputs = self.encoder(input_ids,attention_mask=input_ids.ne(1)).last_hidden_state
        outputs = (outputs * input_ids.ne(1)[:,:,None]).sum(1)/input_ids.ne(1).sum(1)[:,None]
        outputs = outputs.reshape(-1,2,outputs.size(-1))
        outputs = torch.nn.functional.normalize(outputs, p=2, dim=-1)
        cos_sim = (outputs[:,0]*outputs[:,1]).sum(-1)

        if labels is not None:
            loss = ((cos_sim-labels.float())**2).mean()
            return loss,cos_sim
        else:
            return cos_sim 
        
        '''outputs = self.encoder.roberta(inputs_embeds=inputs_embeddings,attention_mask=attn_mask,position_ids=position_idx)[0]
        logits = self.classifier(outputs)
        prob = F.softmax(logits)
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            return loss,prob, outputs[:, 0, :]
        else:
            return prob'''