from Config import args
import os
import torch
from utils import load_dataset, load_dataset_hf
from model import BILSTM_classi
from trainer import NativePtTrainer, MyHfTrainer, NativePtHfTrainer
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer


def get_model_path(name):
    match_dict = {
        'bert': './model/bert_base_chinese',
        'bert-wwm': './model/chinese-bert-wwm-ext',
    }
    if name in match_dict:
        return match_dict[name]
    else:
        raise RuntimeError('unknown model type!')


def main(source=None):
    """
    :param source: the name of all the ablation groups
    :return:
    """
    # 检测一下，如果save_model_path路径存在，就raise Runtimeerror，不然会有override风险
    save_model_path = os.path.join('./log', args['save_model_path'])
    if os.path.exists(save_model_path) and args['save_model_path'] != 'temp':
        raise RuntimeError("Model saving path already exist! Please change a path (args['save_model_path'])")

    # 记录都哪些类别的预训练数据参与了训练
    if args['ablation']:
        os.makedirs(save_model_path)
        if source is not None:
            with open(os.path.join(save_model_path, 'desc.txt'), 'w', encoding='utf-8') as f:
                f.write('ablation study:\n\n')
                for s in source:
                    f.write(f'{s}\n')

    if args['model']['name'] == 'bilstm':
        train_set, dev_set, test_set = load_dataset(args)
        model = BILSTM_classi(args)
        trainer = NativePtTrainer(args, model, [train_set, dev_set, test_set])
    else:
        args['hf_download_path'] = get_model_path(args['model']['name'])
        train_set, dev_set, test_set = load_dataset_hf(args)
        bert_model = AutoModelForSequenceClassification.from_pretrained(args['hf_download_path'],
                                                                        num_labels=args['dataset']['num_label'])
        trainer = NativePtHfTrainer(args, bert_model, [train_set, dev_set, test_set])

    trainer.train(args)


def ablation_study_pretrain(ablation_group):
    assert args['dataset']['name'] == 'CHIP-CDN_DA_v2.0'
    args['finetune'] = False

    for i in range(len(ablation_group)):
        args['dataset']['source'] = ablation_group[:i + 1]
        args['save_model_path'] = f'ablation_pretrain_{i + 1}'
        args['batch_size'] = 256
        args['epoch'] = 200

        main(args['dataset']['source'])


def ablation_study_finetune(ablation_group):
    for i in range(len(ablation_group)):
        args['dataset']['name'] = 'CHIP-CDN'
        args['dataset']['source'] = []
        args['fine_tune'] = 'best'  # False, best, newest
        args['pretrain_model_path'] = f'ablation_pretrain{i + 1}'
        args['save_model_path'] = f'ablation_finetune{i + 1}'
        args['batch_size'] = 64
        args['epoch'] = 100

        main(args['dataset']['source'])


# def train_small_dataset():
#     save_model_path_base = args['save_model_path']
#     percents = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
#     for percent in percents:
#         for i in range(10):
#             args['percent'] = percent  # args里面本身不设置percent，在这里设置，这样就保证只有小数据集模式才有这个参数
#             args['save_model_path'] = f'{save_model_path_base}_{percent}_{i}'
#             main()


def train_small_dataset():
    train_times = 1  # 从0到9
    save_model_path_base = args['save_model_path']
    for train_time in range(train_times):
        percents = [0.05]  # [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        for percent in percents:
            args['percent'] = percent  # args里面本身不设置percent，在这里设置，这样就保证只有小数据集模式才有这个参数
            args['save_model_path'] = f'{save_model_path_base}_{percent}_{train_time}'
            main()


def checkpoint_finetune():
    save_model_path_base = args['save_model_path']
    percents = [0.05]
    for percent in percents:
        for checkpoint in range(1, 3):
            args['percent'] = percent  # args里面本身不设置percent，在这里设置，这样就保证只有小数据集模式才有这个参数
            args['fine_tune'] = f'checkpoint{checkpoint}'
            args['save_model_path'] = f'{save_model_path_base}_checkpoint{checkpoint}_{percent}'
            main()


if __name__ == '__main__':
    ablation_group = [
        'icd4to6',
        'icd4to6_from_cdn',
        'upper_or_lower_position_from_icd',
        'upper_or_lower_position_from_cdn',
        'label_aug_upper_or_lower_position',
        'label_aug_position_all',
        'label_aug_center',
        'CDN_training_set',
        'same_disease',
        'replace_fake_organ_from_icd',
        'replace_fake_center_from_icd',
        'replace_fake_quality_from_icd',
    ]

    assert args['ablation'] != True or args['small_dataset'] != True
    if args['ablation']:
        ablation_study_pretrain(ablation_group)
        ablation_study_finetune(ablation_group)
    elif args['small_dataset']:
        train_small_dataset()
    elif args['checkpoint_finetune']:
        checkpoint_finetune()
    else:
        main()
