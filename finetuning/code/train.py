import math
import numpy as np
from runner import initiate
import matplotlib.pyplot as plt

#this file is a hot mess,i didn't have time to turn it in a proper experimentation facility, self-automatic stuff

"""initiate('''
        --output_dir saved_models/MutantEq \
        --config_name unixcoder-base \
        --model_name_or_path unixcoder-base \
        --tokenizer_name unixcoder-base \
        --requires_grad 1 \
        --do_train \
        --code_db_file dataset/MutantBench_code_db_java.csv \
        --train_data_file dataset/train_pairs.csv \
        --eval_data_file dataset/test_pairs.csv \
        --test_data_file dataset/test_pairs.csv \
        --delete_comments
        --workers_count 0
        --specimen pair''')"""

datum = []
print('1-----------------------------------------')
datum.append(initiate('''
        --output_dir saved_models/MutantEq \
        --config_name unixcoder-base \
        --model_name_or_path unixcoder-base \
        --tokenizer_name unixcoder-base \
        --requires_grad 1 \
        --do_train \
        --code_db_file dataset/MutantBench_code_db_java.csv \
        --train_data_file dataset/train_clusters.csv \
        --eval_data_file dataset/test_clusters.csv \
        --test_data_file dataset/test_clusters.csv \
        --delete_comments
        --train_batch_size 4
        --best_threshold 0.5
        --workers_count 0
        --specimen clusterGGMA
        --lambd 0.1
        --margin -0.015
        --dml_amplification 10              
        --p 12'''))

a = 2


def show(data):
        f1 = [i["eval_f1"] for i in data]
        precision = [i["eval_precision"] for i in data]
        recall = [i["eval_recal"] for i in data]
        threshold = [i["eval_threshold"] for i in data]
        # X-axis values
        x = list(range(len(f1)))  # Assuming all arrays have the same length
        # Plot each line
        plt.plot(x, f1, label='F1 Score')
        plt.plot(x, precision, label='Precision')
        plt.plot(x, recall, label='Recall')
        plt.axhline(y = 0.943, color='darkgoldenrod', label='Paper precision',linestyle = (0, (5, 10)))
        plt.axhline(y = 0.8181, color='darkgreen', label='Paper recall', linestyle = (0, (5, 10)))
        plt.axhline(y = 0.8658, color='navy', label='Paper F1', linestyle = (0, (5, 10)))
        #plt.scatter(x = 9,y = 0.943, color='darkgoldenrod', label='Paper precision',s = 25)
        #plt.scatter(x = 9,y = 0.8181, color='darkgreen', label='Paper recall', s = 25)
        #plt.scatter(x = 9,y = 0.8658, color='navy', label='Paper F1', s = 25)
        #plt.plot(x, loss, label='Loss')

        # Add labels, title, and legend
        plt.xlabel('Epoch')
        plt.ylabel('Values')
        plt.title('Evaluation Metrics')
        plt.legend()
        y_min, y_max = min(min(f1), min(precision), min(recall)), max(max(f1), max(precision), max(recall))
        plt.yticks(np.arange(math.floor(y_min*100)/100, y_max + 0.01, 0.01))
        plt.xticks(ticks=x)

        # Add grid for better visualization
        plt.grid(True)
        plt.show()