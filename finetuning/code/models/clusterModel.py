import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from finetuning.code.models.robertaClassificationHead import RobertaClassificationHead
from torch.nn.functional import relu

        
class ClusterModel(nn.Module):   
    def __init__(self, encoder,config,tokenizer,args):
        super().__init__()
        self.encoder = encoder
        self.config=config
        self.tokenizer=tokenizer
        self.classifier=RobertaClassificationHead(config)
        self.lambd = args.lambd
        self.margin = args.margin
        self.number_of_classes = args.number_of_classes
        self.p = args.p
        verges = torch.zeros(args.number_of_classes, dtype=torch.float, device = args.device)
        self.register_buffer('verges', verges)
        self.args=args
        self.query = 0
    
        
    def forward(self, classes_numbers, centers_ids, mutants_ids, mutants_labels): 
        bs,l=centers_ids.size()

        centers_outputs = self.encoder(centers_ids,attention_mask=centers_ids.ne(1)).last_hidden_state
        mutants_outputs = self.encoder(mutants_ids,attention_mask=mutants_ids.ne(1)).last_hidden_state

        centers_outputs_means = (centers_outputs * centers_ids.ne(1)[:,:,None]).sum(1)/centers_ids.ne(1).sum(1)[:,None]
        mutants_outputs_means = (mutants_outputs * mutants_ids.ne(1)[:,:,None]).sum(1)/mutants_ids.ne(1).sum(1)[:,None]

        centers_outputs_means_normalized = torch.nn.functional.normalize(centers_outputs_means, p=2, dim=-1)
        mutants_outputs_means_normalized = torch.nn.functional.normalize(mutants_outputs_means, p=2, dim=-1)
        cos_sim = (centers_outputs_means_normalized * mutants_outputs_means_normalized).sum(-1)
        distances = 1 - (cos_sim + 1) / 2 #саш смотри

        loss_dml = 0
        if self.training:
            with torch.no_grad():
                encountered_classes = torch.unique(classes_numbers)
                for class_ in encountered_classes:
                    posdist_in_mb_for_class = torch.where((classes_numbers == class_) & (mutants_labels == 1), distances, -1)
                    posdist_in_mb_for_class = posdist_in_mb_for_class[posdist_in_mb_for_class != -1]
                    if posdist_in_mb_for_class.size(dim = 0) == 0:
                        continue

                    if self.verges[class_] == 0:
                        self.verges[class_] = posdist_in_mb_for_class[0]

                    a = 2 / (self.p + 1)
                    decreasing_powers = torch.arange(posdist_in_mb_for_class.size(dim = 0) - 1, -1, -1, dtype=torch.float,
                                                    device = self.args.device)
                    self.verges[class_] = self.verges[class_] * (1 - a) ** (decreasing_powers[0]+1) + \
                                            (posdist_in_mb_for_class * a * (1 - a) ** decreasing_powers).sum()
                
            loss_dml = relu(distances * mutants_labels + (self.verges[classes_numbers] + self.margin - distances) * (1 - mutants_labels)).sum()
        

        outputs_means_normalized = torch.cat((centers_outputs_means_normalized,mutants_outputs_means_normalized),1).reshape(-1,2,768)

        logits = self.classifier(outputs_means_normalized)
        classifier_probs = F.softmax(logits)
        loss_classifier =  torch.nn.functional.cross_entropy(logits, mutants_labels)

        probs = torch.where(classifier_probs[:,0] > classifier_probs[:,1],0,1)
        loss = loss_dml * self.lambd + loss_classifier * (1 - self.lambd)

        return loss, probs