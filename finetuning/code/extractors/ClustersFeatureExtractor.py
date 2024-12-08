import logging
import os
import torch
import statistics
from torch.utils.data import Dataset
from tqdm import tqdm
import csv
import pandas

logger = logging.getLogger(__name__)

import finetuning.code.extractors.PairsFeatureExtractor as PairsFeatureExtractor


class ClustersInputFeatures(object):
    """A single training/test features for a example."""
    def __init__(self,
             class_number,
             center_ids,
             mutant_ids,
             mutant_label

    ):
        self.class_number = class_number
        self.center_ids = center_ids    
        self.mutant_ids = mutant_ids
        self.mutant_label = mutant_label


class ClustersTextDataset(Dataset):
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
                    class_number,center_url,mutant_url,mutant_label=line
                    if center_url not in url_to_code or mutant_url not in url_to_code:
                        continue
                    mutant_label = 0 if mutant_label == '0' else 1
                    data.append((class_number,center_url,mutant_url,mutant_label))

            
            '''data_df = pandas.DataFrame(data, colums = ['class_number','center_url','mutant_url','mutant_label'])
            max_class_number = int(data_df['class_number'].max())
            class_verges = [0]*max_class_number
            for i in range(0, len(class_verges)):
                class_verges[i] = data_df.loc[data_df['class_number'] == i & data_df['mutant_label'] == 1].mean()'''

            examples = []
            for line in tqdm(data,total=len(data)):
                class_number,center_url,mutant_url,mutant_label = line
                center_source_tokens, center_ids = PairsFeatureExtractor.convert_example_to_features(url_to_code[center_url],tokenizer,args)
                mutant_source_tokens, mutant_ids = PairsFeatureExtractor.convert_example_to_features(url_to_code[mutant_url],tokenizer,args)
                examples.append(ClustersInputFeatures(class_number,center_ids, mutant_ids, mutant_label))

            self.examples=examples
            torch.save(self.examples, cache_file_path)
        args.number_of_classes = max([example.class_number for example in self.examples])

    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, item):
        return torch.tensor(self.examples[item].class_number), \
               torch.tensor(self.examples[item].center_ids),  \
               torch.tensor(self.examples[item].mutant_ids),  \
               torch.tensor(self.examples[item].mutant_label)
