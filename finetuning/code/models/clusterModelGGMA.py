import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from finetuning.code.models.robertaClassificationHead import RobertaClassificationHead
from torch.nn.functional import relu

        
class ClusterModelGGMA(nn.Module):   
    def __init__(self, encoder,config,tokenizer,args):
        super().__init__()
        self.encoder = encoder
        self.config=config
        self.tokenizer=tokenizer
        self.classifier=RobertaClassificationHead(config)
        self.lambd = args.lambd
        self.margin = args.margin
        self.coeff = args.coeff
        self.positives_coeff = args.positives_coeff
        self.number_of_classes = args.number_of_classes
        self.p = args.p
        verges = torch.zeros((2,args.number_of_classes), dtype=torch.float, device = args.device)
        self.register_buffer('verges', verges)
        self.args=args
        self.query = 0
    
        
    def forward(self, classes_numbers,inputs_ids_1,position_idx_1,attn_mask_1,inputs_ids_2,position_idx_2,attn_mask_2,url1,url2,labels): 
        bs,l=inputs_ids_1.size()

        inputs_ids=torch.cat((inputs_ids_1.unsqueeze(1),inputs_ids_2.unsqueeze(1)),1).view(bs*2,l)
        position_idx=torch.cat((position_idx_1.unsqueeze(1),position_idx_2.unsqueeze(1)),1).view(bs*2,l)
        attn_mask=torch.cat((attn_mask_1.unsqueeze(1),attn_mask_2.unsqueeze(1)),1).view(bs*2,l,l)

        nodes_mask=position_idx.eq(0)
        token_mask=position_idx.ge(2)        
        inputs_embeddings=self.encoder.embeddings.word_embeddings(inputs_ids)
        nodes_to_token_mask=nodes_mask[:,:,None]&token_mask[:,None,:]&attn_mask
        nodes_to_token_mask=nodes_to_token_mask/(nodes_to_token_mask.sum(-1)+1e-10)[:,:,None]
        avg_embeddings=torch.einsum("abc,acd->abd",nodes_to_token_mask,inputs_embeddings)
        inputs_embeddings=inputs_embeddings*(~nodes_mask)[:,:,None]+avg_embeddings*nodes_mask[:,:,None]    
        
        outputs = self.encoder(inputs_embeds=inputs_embeddings,attention_mask=attn_mask,position_ids=position_idx).last_hidden_state
        outputs_normalized = torch.nn.functional.normalize(outputs, p=2, dim=-1)

        mutants_labels = labels
        centers_outputs_normalized = outputs_normalized[0::2]
        mutants_outputs_normalized = outputs_normalized[1::2]
        centers_outputs_means_normalized = centers_outputs_normalized[:, 0, :].reshape(bs, centers_outputs_normalized.size(-1))
        mutants_outputs_means_normalized = mutants_outputs_normalized[:, 0, :].reshape(bs, mutants_outputs_normalized.size(-1))

        if self.args.do_test:
            self.record_embeddings(centers_outputs_means_normalized,url1,mutants_outputs_means_normalized,url2, labels)

        cos_sim = (centers_outputs_means_normalized * mutants_outputs_means_normalized).sum(-1)
        distances = 1 - (cos_sim + 1) / 2 #саш смотри
        #distances = (centers_outputs_means_normalized - mutants_outputs_means_normalized).pow(2).sum(1).sqrt()

        loss_dml = 0
        if self.training:
            with torch.no_grad():
                encountered_classes = torch.unique(classes_numbers)
                for class_ in encountered_classes:
                    for sign in range(0,2):
                        dist_in_mb_for_class = torch.where((classes_numbers == class_) & (mutants_labels == sign), distances, -1)
                        dist_in_mb_for_class = dist_in_mb_for_class[dist_in_mb_for_class != -1]
                        if dist_in_mb_for_class.size(dim = 0) == 0:
                            continue

                        if self.verges[sign][class_] == 0:
                            self.verges[sign][class_] = dist_in_mb_for_class[0]

                        a = 2 / (self.p + 1)
                        decreasing_powers = torch.arange(dist_in_mb_for_class.size(dim = 0) - 1, -1, -1, dtype=torch.float,
                                                        device = self.args.device)
                        self.verges[sign][class_] = self.verges[sign][class_] * (1 - a) ** (decreasing_powers[0]+1) + \
                                                (dist_in_mb_for_class * a * (1 - a) ** decreasing_powers).sum()
                
            loss_dml = (relu(distances - self.verges[0][classes_numbers] + self.margin)**2 * mutants_labels  + \
                            torch.sqrt(relu(self.verges[1][classes_numbers] + self.margin - distances)) * (1 - mutants_labels)).sum()
            loss_dml = loss_dml / bs * self.args.dml_amplification
            #loss_dml = loss_dml / ((1 - mutants_labels).sum() + 0.0001) * self.args.dml_amplification
        
        outputs_means_normalized = torch.cat((centers_outputs_means_normalized,mutants_outputs_means_normalized),1).reshape(-1,2,768)

        logits = self.classifier(outputs_means_normalized)
        classifier_probs = F.softmax(logits)
        loss_classifier =  torch.nn.functional.cross_entropy(logits, mutants_labels)

        probs = torch.where(classifier_probs[:,0] > classifier_probs[:,1],0,1)
        loss = loss_dml * self.lambd + loss_classifier * (1 - self.lambd)

        return loss, probs
    
    def record_embeddings(self, embedding1_tensor, url1_tensor, embedding2_tensor, url2_tensor, labels):
        for i in range(embedding1_tensor.size(0)):
            url1 = int(url1_tensor[i].cpu().numpy())
            embedding1 = embedding1_tensor[i].cpu().numpy()
            url2 = int(url2_tensor[i].cpu().numpy())
            embedding2 = embedding2_tensor[i].cpu().numpy()
            label = int(labels[i].cpu().numpy())

            if url1 not in self.args.embeddings:
                self.args.embeddings[url1] = ([],[])
                self.args.embeddings[url1][1].append((url1, embedding1))
            self.args.embeddings[url1][label].append((url2, embedding2))  