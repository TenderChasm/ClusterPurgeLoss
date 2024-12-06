import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from robertaClassificationHead import RobertaClassificationHead

        
class pairModel(nn.Module):   
    def __init__(self, encoder,config,tokenizer,args):
        super().__init__()
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

        loss_dml = 0
        if self.args.lambd != 0.0:
            outputs_normalized = torch.nn.functional.normalize(outputs, p=2, dim=-1)
            cos_sim = (outputs_normalized[:,0]*outputs_normalized[:,1]).sum(-1)
            loss_dml = CrossEntropyLoss(cos_sim, labels)
            probs = cos_sim
            loss = loss_dml

        loss_classifier = 0
        if self.args.lambd != 1.0:
            logits = self.classifier(outputs)
            classifier_probs = F.softmax(logits)
            loss_classifier = CrossEntropyLoss(logits, labels)
            probs = classifier_probs
            loss = loss_classifier
        
        if self.args.lambd == 0.0 or self.args.lambd == 1.0:
            return loss, probs
        else:
            return loss_dml * self.args.lambd + loss_classifier * (1 - self.args.lambd), probs
        
        

                