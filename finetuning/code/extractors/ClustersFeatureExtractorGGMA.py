import logging
import os
import numpy as np
import torch
import statistics
from torch.utils.data import Dataset
from tqdm import tqdm
import csv
import pandas

logger = logging.getLogger(__name__)

import finetuning.code.extractors.PairsFeatureExtractor as PairsFeatureExtractor
from finetuning.code.utils import *

class ClustersInputFeaturesGGMA(object):
    """A single training/test features for a example."""
    def __init__(self,
            class_number,
            input_tokens_1,
            input_ids_1,
            position_idx_1,
            dfg_to_code_1,
            dfg_to_dfg_1,
            input_tokens_2,
            input_ids_2,
            position_idx_2,
            dfg_to_code_2,
            dfg_to_dfg_2,
            label

    ):
        self.class_number = class_number

        #The first code function
        self.input_tokens_1 = input_tokens_1
        self.input_ids_1 = input_ids_1
        self.position_idx_1=position_idx_1
        self.dfg_to_code_1=dfg_to_code_1
        self.dfg_to_dfg_1=dfg_to_dfg_1
        
        #The second code function
        self.input_tokens_2 = input_tokens_2
        self.input_ids_2 = input_ids_2
        self.position_idx_2=position_idx_2
        self.dfg_to_code_2=dfg_to_code_2
        self.dfg_to_dfg_2=dfg_to_dfg_2
        
        #label
        self.label=label
        

def convert_examples_to_features(item):
    #source
    class_number,func1,func2,label,tokenizer,args=item
    parser=parsers['java']
    cache = {}
    for func_num,func in enumerate([func1,func2]):
            
            #extract data flow
            code_tokens,dfg=extract_dataflow(func,parser,'java')
            code_tokens=[tokenizer.tokenize('@ '+x)[1:] if idx!=0 else tokenizer.tokenize(x) for idx,x in enumerate(code_tokens)]
            ori2cur_pos={}
            ori2cur_pos[-1]=(0,0)
            for i in range(len(code_tokens)):
                ori2cur_pos[i]=(ori2cur_pos[i-1][1],ori2cur_pos[i-1][1]+len(code_tokens[i]))    
            code_tokens=[y for x in code_tokens for y in x]  
            
            #truncating
            code_tokens=code_tokens[:args.code_length+args.data_flow_length-3-min(len(dfg),args.data_flow_length)][:512-3]
            source_tokens =[tokenizer.cls_token]+code_tokens+[tokenizer.sep_token]
            source_ids =  tokenizer.convert_tokens_to_ids(source_tokens)
            position_idx = [i+tokenizer.pad_token_id + 1 for i in range(len(source_tokens))]
            dfg=dfg[:args.code_length+args.data_flow_length-len(source_tokens)]
            source_tokens+=[x[0] for x in dfg]
            position_idx+=[0 for x in dfg]
            source_ids+=[tokenizer.unk_token_id for x in dfg]
            padding_length=args.code_length+args.data_flow_length-len(source_ids)
            position_idx+=[tokenizer.pad_token_id]*padding_length
            source_ids+=[tokenizer.pad_token_id]*padding_length      
            
            #reindex
            reverse_index={}
            for idx,x in enumerate(dfg):
                reverse_index[x[1]]=idx
            for idx,x in enumerate(dfg):
                dfg[idx]=x[:-1]+([reverse_index[i] for i in x[-1] if i in reverse_index],)    
            dfg_to_dfg=[x[-1] for x in dfg]
            dfg_to_code=[ori2cur_pos[x[1]] for x in dfg]
            length=len([tokenizer.cls_token])
            dfg_to_code=[(x[0]+length,x[1]+length) for x in dfg_to_code]        
            cache[func_num]=source_tokens,source_ids,position_idx,dfg_to_code,dfg_to_dfg

        
    source_tokens_1,source_ids_1,position_idx_1,dfg_to_code_1,dfg_to_dfg_1=cache[0]   
    source_tokens_2,source_ids_2,position_idx_2,dfg_to_code_2,dfg_to_dfg_2=cache[1]   
    return ClustersInputFeaturesGGMA(class_number,source_tokens_1,source_ids_1,position_idx_1,dfg_to_code_1,dfg_to_dfg_1,
                   source_tokens_2,source_ids_2,position_idx_2,dfg_to_code_2,dfg_to_dfg_2,
                     label)


class ClustersTextDatasetGGMA(Dataset):
    def __init__(self, tokenizer, args, file_path, code_db_path):
        logger.info("Creating features from index file at %s ", file_path)

        name=file_path.split('/')[-1].split('.csv')[0]
        folder = 'finetuning/cached_files'
        cache_file_path = os.path.join(folder, f'cached_{name}_GGMA')

        url_to_code={}
        try:
            if not os.path.exists(folder):
                os.makedirs(folder)
            self.examples = torch.load(cache_file_path)
            logger.info("Loading features from cached file %s", cache_file_path)
        except:

            logger.info("Creating features from dataset file at %s", file_path)
            with open(code_db_path) as f:
                file_reader = csv.reader(f) 
                for line in file_reader:  
                    url_to_code[line[0]] = line[1]
                    
            data=[]

            with open(file_path) as f:
                file_reader = csv.reader(f)
                for line in file_reader:
                    class_number,center_url,mutant_url,mutant_label=line
                    if center_url not in url_to_code or mutant_url not in url_to_code:
                        continue
                    mutant_label = 0 if mutant_label == '0' else 1
                    data.append((int(class_number),center_url,mutant_url,mutant_label))

            
            '''data_df = pandas.DataFrame(data, colums = ['class_number','center_url','mutant_url','mutant_label'])
            max_class_number = int(data_df['class_number'].max())
            class_verges = [0]*max_class_number
            for i in range(0, len(class_verges)):
                class_verges[i] = data_df.loc[data_df['class_number'] == i & data_df['mutant_label'] == 1].mean()'''

            examples = []
            for line in tqdm(data,total=len(data)):
                class_number,center_url,mutant_url,mutant_label = line
                examples.append(convert_examples_to_features((class_number, url_to_code[center_url], url_to_code[mutant_url], mutant_label, tokenizer, args)))

            self.examples=examples
            torch.save(self.examples, cache_file_path)
        args.number_of_classes = max([example.class_number for example in self.examples])+1
        self.args = args
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, item):
        #calculate graph-guided masked function
        attn_mask_1= np.zeros((self.args.code_length+self.args.data_flow_length,
                        self.args.code_length+self.args.data_flow_length),dtype=np.bool_)
        #calculate begin index of node and max length of input
        node_index=sum([i>1 for i in self.examples[item].position_idx_1])
        max_length=sum([i!=1 for i in self.examples[item].position_idx_1])
        #sequence can attend to sequence
        attn_mask_1[:node_index,:node_index]=True
        #special tokens attend to all tokens
        for idx,i in enumerate(self.examples[item].input_ids_1):
            if i in [0,2]:
                attn_mask_1[idx,:max_length]=True
        #nodes attend to code tokens that are identified from
        for idx,(a,b) in enumerate(self.examples[item].dfg_to_code_1):
            if a<node_index and b<node_index:
                attn_mask_1[idx+node_index,a:b]=True
                attn_mask_1[a:b,idx+node_index]=True
        #nodes attend to adjacent nodes 
        for idx,nodes in enumerate(self.examples[item].dfg_to_dfg_1):
            for a in nodes:
                if a+node_index<len(self.examples[item].position_idx_1):
                    attn_mask_1[idx+node_index,a+node_index]=True  
                    
        #calculate graph-guided masked function
        attn_mask_2= np.zeros((self.args.code_length+self.args.data_flow_length,
                        self.args.code_length+self.args.data_flow_length),dtype=np.bool_)
        #calculate begin index of node and max length of input
        node_index=sum([i>1 for i in self.examples[item].position_idx_2])
        max_length=sum([i!=1 for i in self.examples[item].position_idx_2])
        #sequence can attend to sequence
        attn_mask_2[:node_index,:node_index]=True
        #special tokens attend to all tokens
        for idx,i in enumerate(self.examples[item].input_ids_2):
            if i in [0,2]:
                attn_mask_2[idx,:max_length]=True
        #nodes attend to code tokens that are identified from
        for idx,(a,b) in enumerate(self.examples[item].dfg_to_code_2):
            if a<node_index and b<node_index:
                attn_mask_2[idx+node_index,a:b]=True
                attn_mask_2[a:b,idx+node_index]=True
        #nodes attend to adjacent nodes 
        for idx,nodes in enumerate(self.examples[item].dfg_to_dfg_2):
            for a in nodes:
                if a+node_index<len(self.examples[item].position_idx_2):
                    attn_mask_2[idx+node_index,a+node_index]=True                      
                    
        return (torch.tensor(self.examples[item].class_number),
                torch.tensor(self.examples[item].input_ids_1),
                torch.tensor(self.examples[item].position_idx_1),
                torch.tensor(attn_mask_1), 
                torch.tensor(self.examples[item].input_ids_2),
                torch.tensor(self.examples[item].position_idx_2),
                torch.tensor(attn_mask_2),                 
                torch.tensor(self.examples[item].label))