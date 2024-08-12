import torch

# nli_for_simcse.csv中共有约270000条数据
args = {
    # change from here in remote server
    'ablation': False,  # perform ablation study or not
    'small_dataset': False,  # perform small dataset study or not
    'checkpoint_finetune': False,  # different from fine_tune. This is for another set of experiments

    ################ATTENTION################
    'dataset': {
        'name': 'CHIP-CDN',  # 'yidu-n7k', 'CHIP-CDN', 'CHIP-CDN_pretrain_icd', 'CHIP-CDN_DA_v3.0'
        'truncate': 32,  # different for each dataset (SimCSE官方代码中的default是32) disease: 64
        'num_label': 37645,  # CHIP-CDN: 37645
        'devset_len': 2000,  # 为了修正CHIP-CDN等任务的acc值 CHIP-CDN: 37645
        'source': [],  # 'icd4to6', 'same_disease', 'upper_or_lower_position', 'label_aug_upper_or_lower_position', 'label_aug_position_all', 'label_aug_center'
    },
    ################ATTENTION################

    # 'dataset': {
    #     'name': 'yidu-n7k_DA',  # 'yidu-n7k', 'CHIP-CDN', 'CHIP-CDN_pretrain_icd'
    #     'truncate': 32,  # different for each dataset (SimCSE官方代码中的default是32) disease: 64
    #     'num_label': 9468,  # CHIP-CDN: 37645
    #     'devset_len': 1000,  # 为了修正CHIP-CDN等任务的acc值 CHIP-CDN: 37645
    #     'source': [],
    # },

    # 'dataset': {
    #     'name': 'similarity_v1.1_trial',  # 'yidu-n7k', 'CHIP-CDN', 'CHIP-CDN_pretrain_icd'
    #     'truncate': 64,  # different for each dataset (SimCSE官方代码中的default是32) disease: 64
    #     'num_label': 2,  # CHIP-CDN: 37645
    #     'source': [],
    # },

    'model': {
        'name': 'bilstm',  # 'bilstm': 1e-4 or 'bert': 5e-5, 'bert-wwm'
        'lr': 1e-3,
    },
    'batch_size': 256,
    'round2eval': 500,  # 实现的时候用min(config.round2eval, len(train_dataloader))  # hf模型目前默认每个epoch测试一次
    'save_model_path': 'bilstm_baseline_cuhk_1e-3_trial3',
    'enable_writer': False,
    'save_checkpoint': True,
    'fine_tune': False,  # False, best, newest, checkpoints(e.g. checkpoint9)
    'pretrain_model_path': 'asdfasdf',  # specify the pretrained model path (enter a random value if not used)
    # end of change
    # 'model_type': 'bert-base-uncased',  # 'bert-base-uncased' or 'bert_base_chinese'
    'epoch': 200,
    'early_stop_round': 30
}

device_id = 0  # modify this on remote server
if torch.cuda.is_available():
    torch.torch.cuda.set_device(device_id)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
args['device'] = device

if __name__ == '__main__':
    print(5e-5)
