import os

base_path = './log/bert_small_dataset_DA'
run_time = 0
percents = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
for p in percents:
    real_path = f'{base_path}_{p}_{run_time}'
    print(real_path)

