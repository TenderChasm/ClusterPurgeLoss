from __future__ import absolute_import, division, print_function
import logging
import os
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import csv

cpu_cont = 16
logger = logging.getLogger(__name__)

from utils import remove_comments_and_docstrings_java


class PairsInputFeatures(object):
    """A single training/test features for a example."""
    def __init__(self,
             input_ids_1,
             input_ids_2,
             label

    ):
        self.input_ids_1 = input_ids_1
        self.input_ids_2 = input_ids_2    
        self.label=label
        

def convert_example_to_features(func, tokenizer, args):
    if args.delete_comments:
        func = remove_comments_and_docstrings_java(func)

    source_tokens = tokenizer.tokenize(func)
    source_tokens = source_tokens[:args.code_length-4]
    source_tokens = [tokenizer.cls_token,"<encoder-only>",tokenizer.sep_token]+source_tokens+[tokenizer.sep_token]
    source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
    padding_length = args.code_length - len(source_ids)
    source_ids += padding_length * [tokenizer.pad_token_id]

    return source_tokens, source_ids


class PairsTextDataset(Dataset):
    def __init__(self, tokenizer, args, file_path, code_db_path):
        logger.info("Creating features from index file at %s ", file_path)

        name=file_path.split('/')[-1].split('.csv')[0]
        folder = 'finetuning/cached_files'
        cache_file_path = os.path.join(folder, f'cached_{name}')
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
                    _,_,url1,url2,label=line
                    if url1 not in url_to_code or url2 not in url_to_code:
                        continue
                    label = 0 if label == '0' else 1
                    data.append((url1,url2,label))

            examples = []
            for line in tqdm(data,total=len(data)):
                url1,url2,label = line
                source_tokens_1, source_ids_1 = convert_example_to_features(url_to_code[url1],tokenizer,args)
                source_tokens_2, source_ids_2 = convert_example_to_features(url_to_code[url2],tokenizer,args)
                examples.append(PairsInputFeatures(source_ids_1, source_ids_2, label))

            self.examples=examples
            torch.save(self.examples, cache_file_path)

    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, item):
        return torch.tensor(self.examples[item].input_ids_1),torch.tensor(self.examples[item].input_ids_2), torch.tensor(self.examples[item].label)
