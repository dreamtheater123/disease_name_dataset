import json
import pickle as pkl
import random

import pandas as pd
import numpy as np
import os

import torch
from torch.utils.data import DataLoader, TensorDataset
from datasets import Dataset, ClassLabel, Features, Value
from transformers import AutoTokenizer


def read_json(filepath):
    """
    this method reads the content in the json file.
    :return: json contents
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def write_json_format(filepath, data):
    """
    this method writes the python dict into readable json format (format that shows the structure instead of putting all the content into a single line)
    :return:
    """
    with open(filepath, 'w', encoding='utf-8') as f:
        content = json.dumps(data, ensure_ascii=False, indent=1)
        f.write(content)


def pickle_load(filepath):
    """
    load pickle binary document into python object
    :return:
    """
    with open(filepath, 'rb') as f:
        data = pkl.load(f)
    return data


def pickle_dump(filepath, data):
    """
    save python object into a pikcle binary document
    :return:
    """
    with open(filepath, 'wb') as f:
        pkl.dump(data, f)


def filter_label_pretrain_dataset(data, args):
    """
    筛选以伪标签为目标的预训练数据集
    手动调整通过哪些规则生成的数据可以被放进去训练
    :return:
    """
    sources = args['dataset']['source']
    if sources == []:  # 所有方法构建的样本都参与训练
        print('no need for filtering data.')
        return data

    output_list = []
    for item in data:
        if item['source'] in sources:
            output_list.append(item)
    print(f'data amount after filtering: {len(output_list)}')
    return output_list


def load_yidun7k_from_excel(filepath):
    # filepath = './data/yidu-n7k'
    filenames = ['train.xlsx', 'val.xlsx', 'answer.xlsx']
    output_list = []
    for filename in filenames:
        print(filename)
        file = os.path.join(filepath, filename)
        dataset = pd.read_excel(file)
        data_dict = []
        for index, row in dataset.iterrows():
            data_dict.append({
                'text': row['原始词'],
                'normalized_result': row['标准词']
            })
        print(len(data_dict))
        output_list.append(data_dict)

    return output_list


def parse_CDN_dataset(args, dataset, label_dict, vocab):
    dataset = filter_label_pretrain_dataset(dataset, args)

    texts, labels = [], []
    count = 0
    for data in dataset:
        normalized_result = data['normalized_result'].replace('"', '').replace('\\', '')
        normalized_result = normalized_result.split('##')
        for label in normalized_result:
            try:
                temp = []
                labels.append(label_dict[label])
                # truncate
                text = data['text'][:args['dataset']['truncate']]
                for char in text:
                    if char in vocab.keys():
                        temp.append(vocab[char])
                    else:  # unk情况
                        temp.append(vocab['<unk>'])
                assert len(temp) <= args['dataset']['truncate']
                # padding
                if len(temp) < args['dataset']['truncate']:
                    remain_len = args['dataset']['truncate'] - len(temp)
                    pads = [vocab['<pad>'] for i in range(remain_len)]
                    temp = temp + pads
                assert len(temp) == args['dataset']['truncate']
                texts.append(temp)
            except:  # 有些标签不在label_dict里面，这种情况直接pass
                pass
                count += 1
                # print(data['text'])
                # print(data['normalized_result'])
                # print(normalized_result)
                # print(label)
                # print('============================================')
                # raise RuntimeError('unknown label type!')
    assert len(texts) == len(labels)
    print(len(dataset), count)

    # convert data into pytorch dataset
    texts = np.array(texts)
    labels = np.array(labels)
    tensordataset = TensorDataset(torch.tensor(texts).to(args['device']), torch.LongTensor(labels).to(args['device']))

    return tensordataset


def parse_CDN_dataset_hf(args, dataset, label_dict, tokenizer):
    def tokenize_function(examples):
        return tokenizer(examples['text'], padding='max_length', truncation=True,
                         max_length=args['dataset']['truncate'])

    dataset = filter_label_pretrain_dataset(dataset, args)

    # HF dataset settings
    class_names = [c for c in label_dict]
    feature_dict = {'text': Value(dtype='string'), 'label': ClassLabel(num_classes=len(label_dict))}
    features = Features(feature_dict)

    data_dict = {
        'text': [],
        'label': []
    }
    count = 0
    for data in dataset:
        normalized_result = data['normalized_result'].replace('"', '').replace('\\', '')
        normalized_result = normalized_result.split('##')
        for label in normalized_result:
            if label in class_names:
                data_dict['text'].append(data['text'])
                data_dict['label'].append(label_dict[label])
            else:
                count += 1

    assert len(data_dict['text']) == len(data_dict['label'])
    print(len(dataset), count)
    dataset = Dataset.from_dict(data_dict, features=features)
    dataset = dataset.map(tokenize_function, batched=True)
    print(dataset)

    return dataset


def text2id(text, args, vocab):
    temp = []
    # truncate
    text = text[:args['dataset']['truncate']]
    for char in text:
        if char in vocab.keys():
            temp.append(vocab[char])
        else:  # unk情况
            temp.append(vocab['<unk>'])
    assert len(temp) <= args['dataset']['truncate']
    # padding
    if len(temp) < args['dataset']['truncate']:
        remain_len = args['dataset']['truncate'] - len(temp)
        pads = [vocab['<pad>'] for i in range(remain_len)]
        temp = temp + pads
    assert len(temp) == args['dataset']['truncate']

    return temp


def parse_similarity_dataset(args, dataset, vocab):
    """
    label 0 for similar, 1 for unsimilar
    :return:
    """
    # dataset = filter_label_pretrain_dataset(dataset, args)

    texts_1, texts_2, labels = [], [], []
    count = 0
    for data in dataset:  # data['text1'], data['text2'], data['label']
        text1_id = text2id(data['text1'], args, vocab)
        text2_id = text2id(data['text2'], args, vocab)
        texts_1.append(text1_id)
        texts_2.append(text2_id)
        if data['label'] == 'similar':
            labels.append(0)
        elif data['label'] == 'unsimilar':
            labels.append(1)
        else:
            raise RuntimeError('unknown label type!')

    # convert data into pytorch dataset
    texts_1 = np.array(texts_1)
    texts_2 = np.array(texts_2)
    labels = np.array(labels)
    tensordataset = TensorDataset(
        torch.tensor(texts_1).to(args['device']),
        torch.tensor(texts_2).to(args['device']),
        torch.LongTensor(labels).to(args['device'])
    )

    return tensordataset


def parse_similarity_dataset_hf(args, dataset, tokenizer):
    """
    其实只需要关注hf的similarity数据集，因为CBLUE就是那hf做的模型，lstm模型没必要跑了就
    """
    def tokenize_function(examples):
        return tokenizer(examples['text1'], examples['text2'], padding='max_length', truncation=True,
                         max_length=args['dataset']['truncate'])

    # dataset = filter_label_pretrain_dataset(dataset, args)

    # HF dataset settings
    class_names = ['similar', 'unsimilar']
    feature_dict = {
        'text1': Value(dtype='string'),
        'text2': Value(dtype='string'),
        'label': ClassLabel(num_classes=len(class_names))
    }
    features = Features(feature_dict)

    data_dict = {
        'text1': [],
        'text2': [],
        'label': []
    }
    for data in dataset:  # data['text1'], data['text2'], data['label']
        data_dict['text1'].append(data['text1'])
        data_dict['text2'].append(data['text2'])

        if data['label'] == 'similar':
            data_dict['label'].append(0)
        elif data['label'] == 'unsimilar':
            data_dict['label'].append(1)
        else:
            raise RuntimeError('unknown label type!')

    assert len(data_dict['text1']) == len(data_dict['label'])
    print(len(dataset))
    dataset = Dataset.from_dict(data_dict, features=features)
    dataset = dataset.map(tokenize_function, batched=True)

    return dataset


def sample_CDN_dataset(dataset, percent):
    random.shuffle(dataset)
    left_len = round(len(dataset) * percent)
    dataset = dataset[:left_len]

    return dataset


def load_dataset(args):
    with open('./vocab/vocab2id.pkl', 'rb') as file:
        vocab = pkl.load(file)
    file_path = './data/' + args['dataset']['name'] + '/'
    print(file_path)
    if 'yidu-n7k' in args['dataset']['name']:
        # file_list = os.listdir(file_path)
        if args['dataset']['name'] == 'yidu-n7k':
            datasets = load_yidun7k_from_excel(file_path)
        else:
            datasets = [
                read_json(file_path + 'yidu_train.json'),
                read_json(file_path + 'yidu_dev.json'),
                None
            ]
        label_dict = read_json(file_path + 'label_list.json')

        trainingset = parse_CDN_dataset(args, datasets[0], label_dict, vocab)
        devset = parse_CDN_dataset(args, datasets[1], label_dict, vocab)
        testset = parse_CDN_dataset(args, datasets[2], label_dict, vocab) if datasets[2] is not None else None
    elif 'CHIP-CDN' in args['dataset']['name']:
        trainingset = read_json(file_path + 'CHIP-CDN_train.json')
        devset = read_json(file_path + 'CHIP-CDN_dev.json')
        if args['small_dataset'] or args['checkpoint_finetune']:
            trainingset = sample_CDN_dataset(trainingset, args['percent'])
        # testset = read_json(file_path + 'CHIP-CDN_test.json')
        label_dict = read_json(file_path + 'label_list.json')
        # label_file = pd.read_excel(file_path + '国际疾病分类 ICD-10北京临床版v601.xlsx', header=None, names=['text', 'label_text'])
        # label_text = list(set(list(label_file['label_text'])))
        # print(len(label_text))
        # label_dict = {}
        # for i, label in enumerate(label_text):
        #     label_dict[label] = i
        # write_json_format('./data/CHIP-CDN/label_list.json', label_dict)

        trainingset = parse_CDN_dataset(args, trainingset, label_dict, vocab)
        devset = parse_CDN_dataset(args, devset, label_dict, vocab)
        testset = None
        # testset = parse_CDN_dataset(testset, label_dict)
    elif 'similarity' in args['dataset']['name']:
        raise RuntimeError('similarity data should be loaded in HuggingFace way and run by Bert model!')
    else:
        raise RuntimeError('unknown dataset!')

    return trainingset, devset, testset


def load_dataset_hf(args):
    file_path = './data/' + args['dataset']['name'] + '/'
    print(file_path)
    tokenizer = AutoTokenizer.from_pretrained(args['hf_download_path'])

    if 'yidu-n7k' in args['dataset']['name']:
        if args['dataset']['name'] == 'yidu-n7k':
            datasets = load_yidun7k_from_excel(file_path)
        else:
            datasets = [
                read_json(file_path + 'yidu_train.json'),
                read_json(file_path + 'yidu_dev.json'),
                None
            ]
        label_dict = read_json(file_path + 'label_list.json')

        trainingset = parse_CDN_dataset_hf(args, datasets[0], label_dict, tokenizer)
        devset = parse_CDN_dataset_hf(args, datasets[1], label_dict, tokenizer)
        testset = parse_CDN_dataset_hf(args, datasets[2], label_dict, tokenizer) if datasets[2] is not None else None
    elif 'CHIP-CDN' in args['dataset']['name']:
        trainingset = read_json(file_path + 'CHIP-CDN_train.json')
        devset = read_json(file_path + 'CHIP-CDN_dev.json')
        if args['small_dataset'] or args['checkpoint_finetune']:
            trainingset = sample_CDN_dataset(trainingset, args['percent'])
        # testset = read_json(file_path + 'CHIP-CDN_test.json')
        label_dict = read_json(file_path + 'label_list.json')

        trainingset = parse_CDN_dataset_hf(args, trainingset, label_dict, tokenizer)
        devset = parse_CDN_dataset_hf(args, devset, label_dict, tokenizer)
        testset = None
        # testset = parse_CDN_dataset(testset, label_dict)
    elif 'similarity' in args['dataset']['name']:
        trainingset = read_json(file_path + 'train.json')
        devset = read_json(file_path + 'dev.json')
        # testset = read_json(file_path + 'CHIP-CDN_test.json')

        trainingset = parse_similarity_dataset_hf(args, trainingset, tokenizer)
        devset = parse_similarity_dataset_hf(args, devset, tokenizer)
        testset = None
    else:
        raise RuntimeError('unknown dataset!')

    return trainingset, devset, testset


if __name__ == '__main__':
    # load_yidun7k_from_excel()

    data = {
        'text1': '句子一',
        'text2': '句子二！',
    }
    tokenizer = AutoTokenizer.from_pretrained("./model/bert_base_chinese")
    temp1 = tokenizer(data['text1'], padding='max_length', truncation=True, max_length=8)
    temp2 = tokenizer(data['text2'], padding='max_length', truncation=True, max_length=8)
    temp_all = tokenizer(data['text1'], data['text2'], padding='max_length', truncation=True, max_length=8)
    pass
