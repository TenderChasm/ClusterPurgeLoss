from __future__ import absolute_import, division, print_function
import argparse
import logging
import os
import pickle
import random
import numpy as np
from datetime import datetime
import torch
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
from transformers import (AdamW, get_linear_schedule_with_warmup,
                          RobertaConfig, RobertaModel, RobertaTokenizer)
from tqdm import tqdm
import shlex

from models.pairModel import PairModel
import extractors.PairsFeatureExtractor as PairsFeatureExtractor
from models.clusterModel import ClusterModel
import extractors.ClustersFeatureExtractor as ClustersFeatureExtractor
from models.clusterModelGGMA import ClusterModelGGMA
import extractors.ClustersFeatureExtractorGGMA as ClustersFeatureExtractorGGMA

cpu_cont = 16
logging.basicConfig(filename = 'log.txt', level=logging.DEBUG)
logger = logging.getLogger(__name__)

import warnings
warnings.filterwarnings("ignore", category=FutureWarning) 


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def train(args, train_dataset, model, tokenizer):
    """ Train the model """
    
    #build dataloader
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size,num_workers=args.workers_count)

    args.max_steps=args.epochs*len( train_dataloader)
    args.save_steps=len(train_dataloader)
    args.warmup_steps=args.max_steps//5
    model.to(args.device)
    
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                num_training_steps=args.max_steps)

    # multi-gpu training
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.train_batch_size//max(args.n_gpu, 1))
    logger.info("  Total train batch size = %d",args.train_batch_size*args.gradient_accumulation_steps)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", args.max_steps)
    
    global_step=0
    tr_loss, logging_loss,avg_loss,tr_nb,tr_num,train_loss = 0.0, 0.0,0.0,0,0,0
    best_f1=0

    model.zero_grad()
 
    for idx in range(args.epochs): 
        bar = tqdm(train_dataloader,total=len(train_dataloader))
        tr_num=0
        train_loss=0
        for step, batch in enumerate(bar):
            features=[x.to(args.device)  for x in batch]
            model.train()
            loss,logits = model(*features)

            if args.n_gpu > 1:
                loss = loss.mean()
                
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            tr_loss += loss.item()
            tr_num+=1
            train_loss+=loss.item()
            if avg_loss==0:
                avg_loss=tr_loss
                
            avg_loss=round(train_loss/tr_num,5)
            bar.set_description("epoch {} loss {}".format(idx,avg_loss))
              
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()  
                global_step += 1
                output_flag=True
                avg_loss=round(np.exp((tr_loss - logging_loss) /(global_step- tr_nb)),4)

                if global_step % args.save_steps == 0:
                    results = evaluate(args, model, tokenizer, 0.5, eval_when_training=True)   


                    
                    # Save model checkpoint
                    if results['eval_f1']>best_f1:
                        best_f1=results['eval_f1']
                        logger.info("  "+"*"*20)  
                        logger.info("  Best f1:%s",round(best_f1,4))
                        logger.info("  "+"*"*20)                          
                        
                        checkpoint_prefix = 'checkpoint-best-f1'
                        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))                        
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)                        
                        model_to_save = model.module if hasattr(model,'module') else model
                        output_dir = os.path.join(output_dir, '{}'.format('model.bin')) 
                        torch.save(model_to_save.state_dict(), output_dir)
                        logger.info("Saving model checkpoint to %s", output_dir)


def evaluate(args, model, tokenizer, best_threshold, eval_when_training=False, mode = 'eval'):
    
    data_file = args.eval_data_file if mode == 'eval' else args.eval_train_file

    dataset = args.extractor(tokenizer, args, file_path=data_file, code_db_path = args.code_db_file)
    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset, sampler=sampler,batch_size=args.eval_batch_size,num_workers=args.workers_count)

    # multi-gpu evaluate
    if args.n_gpu > 1 and eval_when_training is False:
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info(f"***** Running {mode} *****")
    logger.info("  Num examples = %d", len(dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    
    nb_eval_steps = 0
    model.eval()
    logits=[]  
    y_trues=[]
    for batch in tqdm(dataloader):
        features = [x.to(args.device)  for x in batch]
        labels = features[-1]
        with torch.no_grad():
            lm_loss,logit = model(*features)
            logits.append(logit.cpu().numpy())
            y_trues.append(labels.cpu().numpy())
        nb_eval_steps += 1

    #calculate scores
    logits=np.concatenate(logits,0)
    y_trues=np.concatenate(y_trues,0)
    y_preds=logits>best_threshold

    from sklearn.metrics import recall_score
    recall=recall_score(y_trues, y_preds, average='macro')
    from sklearn.metrics import precision_score
    precision=precision_score(y_trues, y_preds, average='macro')   
    from sklearn.metrics import f1_score
    f1=f1_score(y_trues, y_preds, average='macro')             
    result = {
        f"{mode}_recall": float(recall),
        f"{mode}_precision": float(precision),
        f"{mode}_f1": float(f1),
        f"{mode}_threshold":best_threshold,
        
    }

    logger.info(f"***** {mode} results *****")
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(round(result[key],4)))

    return result

def test(args, model, tokenizer, best_threshold=0):
    return eval(args, model, tokenizer, best_threshold, mode = 'test')


def initiate(arg_string):
    arguments_tokens = shlex.split(arg_string)
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--requires_grad", default=None, type=int, required=True)
    parser.add_argument("--code_db_file", default=None, type=str, required=True)
    parser.add_argument("--train_data_file", default=None, type=str, required=True,
                        help="The input training data file (a text file).")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--eval_data_file", default=None, type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")
    parser.add_argument("--test_data_file", default=None, type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")
                    
    parser.add_argument("--model_name_or_path", default=None, type=str,
                        help="The model checkpoint for weights initialization.")

    parser.add_argument("--config_name", default="", type=str,
                        help="Optional pretrained config name or path if not the same as model_name_or_path")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path")

    parser.add_argument("--code_length", default=256, type=int,
                        help="Optional code input sequence length after tokenization.") 
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run eval on the dev set.")    
    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Run evaluation during training at each logging step.")

    parser.add_argument("--train_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=2e-6, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")

    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--epochs', type=int, default=30,
                        help="training epochs")
    
    parser.add_argument('--delete_comments', action='store_true',
                        help="comments in the source code is deleted")
    parser.add_argument('--workers_count', default=4, type=int,
                        help='number of workers to load data with')
    parser.add_argument('--best_threshold', default=0.5, type=float,
                        help='treshold of probability for the classification to flip')
    parser.add_argument('--specimen', type=str, default = "pair",
                        help='1 of the 3 finetuning options')
    parser.add_argument('--lambd', type=float, default = 1.0,
                        help='weight of the DML loss function in loss combining expression' )
    parser.add_argument('--margin', type=float, default = 0.1,
                        help='margin for CPL and trilet loss' )
    parser.add_argument('--p', type=float, default = 3,
                        help='margin for CPL and trilet loss' )
    parser.add_argument("--data_flow_length", default=64, type=int,
                        help="Optional Data Flow input sequence length after tokenization.") 

    args = parser.parse_args(arguments_tokens)
    print(datetime.now())

    # Setup CUDA, GPU
    #device = "cpu"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',datefmt='%m/%d/%Y %H:%M:%S',level=logging.INFO)
    logger.warning("device: %s, n_gpu: %s",device, args.n_gpu)

    match args.specimen:
        case 'pair':
            args.extractor = PairsFeatureExtractor.PairsTextDataset
            args.model = PairModel
        case 'cluster':
            args.extractor = ClustersFeatureExtractor.ClustersTextDataset
            args.model = ClusterModel
        case 'clusterGGMA':
            args.extractor = ClustersFeatureExtractorGGMA.ClustersTextDatasetGGMA
            args.model = ClusterModelGGMA

    # Set seed
    set_seed(args)
    config = RobertaConfig.from_pretrained(args.config_name if args.config_name else args.model_name_or_path)
    config.num_labels=1
    tokenizer = RobertaTokenizer.from_pretrained(args.tokenizer_name)
    if args.do_train:
        train_dataset = args.extractor(tokenizer, args, file_path=args.train_data_file, code_db_path= args.code_db_file)
    model = RobertaModel.from_pretrained(args.model_name_or_path,config=config)    
    model = args.model(model,config,tokenizer,args)

    for name, param in model.named_parameters():
        if 'classifier' not in name:
            param.requires_grad = False if args.requires_grad==0 else True

    logger.info("Training/evaluation parameters %s", args)
    # Training
    if args.do_train:
        train(args, train_dataset, model, tokenizer)

    # Evaluation
    results = {}
    if args.do_eval:
        checkpoint_prefix = 'checkpoint-best-f1/model.bin'
        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))  
        model.load_state_dict(torch.load(output_dir))
        model.to(args.device)
        results = evaluate(args, model, tokenizer, best_treshold = args.best_threshold)
        
    if args.do_test:
        checkpoint_prefix = 'checkpoint-best-f1/model.bin'
        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))  
        model.load_state_dict(torch.load(output_dir))
        model.to(args.device)
        results = test(args, model, tokenizer,best_treshold = args.best_threshold)

    print(datetime.now())
    return results

