from runner import initiate


initiate('''
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
        --specimen pair''')