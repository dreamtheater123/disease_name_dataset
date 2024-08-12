import pandas as pd
import os
import json
import sys
import time
import pickle as pkl
import numpy as np
import random

import setuptools
from sklearn.metrics.pairwise import cosine_similarity


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


class Embedding:
    def __init__(self):
        with open('./vocab/vocab2id.pkl', 'rb') as f:
            self.vocab = pkl.load(f)
        with open('./vocab/pure_weights.pkl', 'rb') as f:
            self.weights = pkl.load(f)

    def word2id(self, sentence):
        id_list = []
        for d in sentence:
            if d in self.vocab.keys():
                id_list.append(self.vocab[d])
            else:  # unk情况
                id_list.append(self.vocab['<unk>'])

        return id_list

    def embedding_lookup(self, i):
        embedding = self.weights[i]
        assert len(embedding) == 200
        return embedding

    def gen_sentence_embedding(self, sentence):
        id_list = self.word2id(sentence)
        word_embeddings = []
        for i in id_list:
            word_embeddings.append(self.embedding_lookup(i))
        word_embeddings = np.array(word_embeddings)
        return np.mean(word_embeddings, axis=0)


def filter_clinical_term(text, mode):
    """
    数据集的临床名称中可能一个临床名称包含多个真实疾病，这种情况对于数据增强来说无法识别，于是把这些去除掉
    如果疾病可以保留，返回True
    如果疾病需要被剔除掉，返回False
    :param text:
    :return:
    """
    if mode == 'disease':
        banned_char = ['?', '？', ';', '；', '.', '/', ',', '，', '。', '\\']
        if '1' in text and '2' in text:
            return False
    elif mode == 'operation':
        banned_char = ['+']
    else:
        raise RuntimeError('filter_clinical_term setting error!')

    for char in banned_char:
        if char in text:
            return False

    return True


def filter_icd_disease(key, value):
    """
    北京版icd中没有800和900，都是.8或.9的形式
    .8, .9, 其他, 未特指
    以上所有疾病全部被剔除掉
    如果疾病可以保留，返回True
    如果疾病需要被剔除掉，返回False
    :param key:
    :param value:
    :return:
    """
    if len(key) == 5 and key[-2:] in ['.8', '.9']:
        return False
    elif '其他' in value or '未特指' in value or '并发' in value or '糖尿病性' in value:
        return False
    else:
        return True


def filter_icd_operation(key, value):
    """
    去除 其他, 未特指 的手术
    去除 手术A伴手术B的情况，直接把含有伴的手术去掉
    :param key:
    :param value:
    :return:
    """
    if '其他' in value or '未特指' in value or '伴' in value:
        return False
    else:
        return True


def is_upper_or_lower_position(position_tree, p1, p2):
    """
    judge whether p1 and p2 is upper-position or lower-position relation.
    :param position_tree:
    :param p1: position1
    :param p2: position2
    :return: True or False
    """
    if 'father' in position_tree[p1].keys():
        p1_father_and_children = position_tree[p1]['father'] + position_tree[p1]['children']
    else:
        p1_father_and_children = position_tree[p1]['children']
    if 'father' in position_tree[p2].keys():
        p2_father_and_children = position_tree[p2]['father'] + position_tree[p2]['children']
    else:
        p2_father_and_children = position_tree[p2]['children']

    if p1 in p2_father_and_children or p2 in p1_father_and_children:
        return True
    else:
        return False


def already_exist(output_list, candidate):
    for output in output_list:
        check_list = [output['text'], output['normalized_result']]
        if candidate['text'] in check_list and candidate['normalized_result'] in check_list:
            return True
    return False


def global_init(config):
    with open('./data/CHIP-CDN_DA/raw_data/positions1.json', 'r', encoding='utf-8') as f:
        positions = json.load(f)

    if config['type'] == 'disease':
        target_train_set = read_json('./data/CHIP-CDN_DA/raw_data/CHIP-CDN_train.json')
        label_file = pd.read_excel('./data/CHIP-CDN_DA/' + '国际疾病分类 ICD-10北京临床版v601.xlsx', header=None, names=['icd_code', 'text'])
        icd_dict = {}
        disease2icd = {}
        for index, row in label_file.iterrows():
            icd_dict[row['icd_code']] = row['text']
        for index, row in label_file.iterrows():
            if row['text'] not in disease2icd:
                disease2icd[row['text']] = row['icd_code']
            else:  # 一个疾病名称对应多个icd编码的情况（有时候，四位码和六位码的名称是一样的）
                if len(row['icd_code']) < len(disease2icd[row['text']]):
                    disease2icd[row['text']] = row['icd_code']
    elif config['type'] == 'operation':
        target_train_set = []
        icd_dict = {}
        disease2icd = {}

        dataset = pd.read_excel('./data/yidu-n7k/train.xlsx')
        for index, row in dataset.iterrows():
            target_train_set.append({
                'text': row['原始词'],
                'normalized_result': row['标准词']
            })

        count = 0
        with open('./data/yidu-n7k/code.txt', 'r', encoding='utf-8') as f:
            label_file = f.readlines()
            label_file = [l.strip().split('\t') for l in label_file]
        for item in label_file:  # item[0]是icd编码，item[1]是手术名称
            icd_dict[item[0]] = item[1]
        for item in label_file:
            if item[1] not in disease2icd:
                disease2icd[item[1]] = item[0]
            else:  # 一个手术名称对应多个icd编码的情况（有时候，四位码和六位码的名称是一样的）
                if len(item[0]) < len(disease2icd[item[1]]):
                    disease2icd[item[1]] = item[0]
                count += 1
        print(f'{count} operations with multiple icd codes!')
    else:
        raise RuntimeError('unknown data augmentation type!')

    config['target_train_set'] = target_train_set
    config['positions'] = positions
    config['icd_dict'] = icd_dict   # code to disease
    config['disease2icd'] = disease2icd  # disease to code

    return config


def gen_icd4to6_disease(config):
    """
    把六位码疾病的标签变成四位码
    :return:
    """
    current_icd = None
    icd4_to_6 = {}
    for key in config['icd_dict']:
        if len(key) == 5 and key[3] == '.':
            current_icd = key
            icd4_to_6[key] = [config['icd_dict'][key]]
        else:
            replaced_key = key.replace('xx', '.x')
            if replaced_key[:5] == current_icd:
                icd4_to_6[current_icd].append(config['icd_dict'][key])
    for key in icd4_to_6:
        print(key, icd4_to_6[key])

    # 列表中第一个是标签
    output_list = []
    for key in icd4_to_6:
        label = icd4_to_6[key][0]
        for disease in icd4_to_6[key][1:]:
            output_list.append({
                'text': disease,
                'normalized_result': label,
                'source': 'icd4to6'
            })

    return output_list


def gen_icd3to4_operate(config):
    """
    把六位码疾病的标签变成四位码
    :return:
    """
    current_icd = None
    icd3_to_4 = {}
    for key in config['icd_dict']:
        if len(key) == 4 and key[2] == '.':
            current_icd = key
            icd3_to_4[key] = [config['icd_dict'][key]]
        elif len(key) == 7:
            if key[:4] == current_icd:
                icd3_to_4[current_icd].append(config['icd_dict'][key])
    for key in icd3_to_4:
        print(key, icd3_to_4[key])

    # 列表中第一个是标签
    output_list = []
    for key in icd3_to_4:
        label = icd3_to_4[key][0]
        for disease in icd3_to_4[key][1:]:
            output_list.append({
                'text': disease,
                'normalized_result': label,
                'source': 'icd3to4'
            })

    return output_list


def gen_icd3_from_yidu(config):
    """
    把四位码疾病的标签变成三位码
    注意这里不能用devset，因为这样相当于变相用到了验证集中的标签信息
    icd_dict
    disease2icd
    :return:
    """
    train_text, train_label_text = [], []
    for item in config['target_train_set']:
        if filter_clinical_term(item['text'], mode='operation'):  # 过滤掉临床名称中包含多个疾病的
            labels = item['normalized_result'].replace('"', '').replace('\\', '').split("##")
            for label in labels:
                train_text.append(item['text'])
                train_label_text.append(label)

    for i, label in enumerate(train_label_text):
        if label in config['disease2icd']:
            icd = config['disease2icd'][label]
            if len(icd) == 7 and icd[2] == '.':  # 六位码
                replaced_icd = icd[:4]
                if replaced_icd in config['icd_dict']:
                    train_label_text[i] = config['icd_dict'][replaced_icd]
    assert len(train_text) == len(train_label_text)

    output_list = []
    for i in range(len(train_text)):
        output_list.append({
            'text': train_text[i],
            'normalized_result': train_label_text[i],
            'source': 'icd3to4_from_yidu'
        })

    return output_list


def gen_icd4_from_cdn(config):
    """
    把六位码疾病的标签变成四位码
    注意这里不能用devset，因为这样相当于变相用到了验证集中的标签信息
    icd_dict
    disease2icd
    :return:
    """
    train_text, train_label_text = [], []
    for item in config['target_train_set']:
        if filter_clinical_term(item['text'], mode='disease'):  # 过滤掉临床名称中包含多个疾病的
            labels = item['normalized_result'].replace('"', '').replace('\\', '').split("##")
            for label in labels:
                train_text.append(item['text'])
                train_label_text.append(label)

    for i, label in enumerate(train_label_text):
        if label in config['disease2icd']:
            icd = config['disease2icd'][label]
            if len(icd) > 5 and icd[3] == '.':  # 六位码
                replaced_icd = icd.replace('xx', '.x')  # 替换成四位码
                replaced_icd = replaced_icd[:5]
                if replaced_icd in config['icd_dict']:
                    train_label_text[i] = config['icd_dict'][replaced_icd]
    assert len(train_text) == len(train_label_text)

    output_list = []
    for i in range(len(train_text)):
        output_list.append({
            'text': train_text[i],
            'normalized_result': train_label_text[i],
            'source': 'icd4to6_from_cdn'
        })

    return output_list


def gen_disease_another_name(config):
    filenames = ['same_disease_expand.txt', 'another_name.json', 'data_as_anothername.json']
    output_list = []

    for filename in filenames:
        if '.txt' in filename:
            with open(os.path.join('data/CHIP-CDN_DA/raw_data', filename), 'r', encoding='utf-8') as f:
                data = f.readlines()
            # 经查证，所有数据项里面要么是' '分割，要么是'  '分割，且空格只出现在分割处
            # 如果第一个疾病在icd列表中，那么把第二个归成第一个的类，如果没有再考虑归到第二个的类，如果全没有就pass
            for line in data:
                flag = False
                disease_pair = line.strip().replace('  ', '\t').replace(' ', '\t').split('\t')
                assert len(disease_pair) == 2
                if disease_pair[0] in config['disease2icd']:
                    output_list.append({
                        'text': disease_pair[1],
                        'normalized_result': disease_pair[0],
                        'source': 'same_disease'
                    })
                elif disease_pair[1] in config['disease2icd']:
                    output_list.append({
                        'text': disease_pair[0],
                        'normalized_result': disease_pair[1],
                        'source': 'same_disease'
                    })
        elif '.json' in filename:
            with open(os.path.join('data/CHIP-CDN_DA/raw_data', filename), 'r', encoding='utf-8') as f:
                data = json.load(f)
            for key in data.keys():
                disease_pair = [key.replace('\t', '').replace('\n', ''), data[key].replace('\t', '').replace('\n', '')]
                if disease_pair[1] in config['disease2icd']:
                    output_list.append({
                        'text': disease_pair[0],
                        'normalized_result': disease_pair[1],
                        'source': 'same_disease'
                    })
        else:
            raise RuntimeError('no such file!')

    print(len(output_list))
    return output_list


def gen_disease_from_icd(config):
    """
    another_name1.json中的所有别名疾病的标签都是以icd编码的形式体现的
    所以这个函数其实也是构造的别名疾病
    :return:
    """
    with open('./data/CHIP-CDN_DA/raw_data/ICD10_北京版6.01_医保版.txt', 'r', encoding='utf-8') as f:
        lines = f.readlines()
        print(lines[:10])
        lines = [l.strip().split('\t') for l in lines]
        print(lines[:10])
        icd_yibao2bj = {l[2]: l[0] for l in lines}
        # print(icd_yibao2bj)
        print(len(icd_yibao2bj))

    data = read_json('./data/CHIP-CDN_DA/raw_data/another_name1.json')
    output_list = []
    for key in data:
        if data[key] in icd_yibao2bj:
            output_list.append({
                'text': key,
                'normalized_result': config['icd_dict'][icd_yibao2bj[data[key]]],
                'source': 'same_disease'
            })

    return output_list


def gen_operation_from_icd(config):
    """
    another_name1.json中的所有别名疾病的标签都是以icd编码的形式体现的
    所以这个函数其实也是构造的别名疾病
    :return:
    """
    data = read_json('./data/yidu-n7k_DA/raw_data/bieming.json')
    output_list = []
    for key in data:
        for icd in data[key]:
            if icd in config['icd_dict']:
                output_list.append({
                    'text': key,
                    'normalized_result': config['icd_dict'][icd],
                    'source': 'same_operation'
                })

    return output_list


def judge_organ_center_in_ner_result(ner_result, name, mode):
    """
    判断术语的ner结果中是否既包含主导词（术式）又包含部位
    :param mode:
    :return:
    """
    if mode == 'disease':
        if 'organ' in ner_result[name] and 'center' in ner_result[name]:  # 既有部位又有主导词
            return True
        else:
            return False
    else:
        if 'BW' in ner_result and 'SHS' in ner_result:
            return True
        else:
            return False


def fix_operation_ner_result(result_list, name_list):
    """
    把手术ner的结果变成跟疾病ner一样的
    :return:
    """
    for i in range(len(result_list)):
        if 'BW' in result_list[i]:
            result_list[i]['organ'] = result_list[i].pop('BW')
        if 'SHS' in result_list[i]:
            result_list[i]['center'] = result_list[i].pop('SHS')
        result_list[i] = {name_list[i]: result_list[i]}

    return result_list


def replace_position_real_pos(config, ner):
    """
    在标准疾病中找真实的替换疾病，主导词相同，部位是上下位
    :param icd_dict:
    :param positions:
    :param ner:
    :return:
    """
    embed = Embedding()
    count = 0
    input_list = []
    position_list = [p for p in config['positions'].keys()]
    print(position_list)

    if config['type'] == 'disease':
        for icd_key in config['icd_dict'].keys():
            if (icd_key < 'O' or icd_key[0] in ['Q', 'R', 'S', 'T']) and filter_icd_disease(icd_key, config['icd_dict'][icd_key]) == True:  # 开头从E到N的所有疾病都被作为候选疾病进行筛选
                input_list.append(config['icd_dict'][icd_key])
    else:
        for icd_key in config['icd_dict'].keys():
            if filter_icd_operation(icd_key, config['icd_dict'][icd_key]):  # 开头从E到N的所有疾病都被作为候选疾病进行筛选
                input_list.append(config['icd_dict'][icd_key])
    tik = time.time()
    result_list = ner.parser_list(input_list)
    if config['type'] == 'operation':
        result_list = fix_operation_ner_result(result_list, input_list)
    tok = time.time()
    assert len(input_list) == len(result_list)

    same_center_dict = {}  # 相同主导词的疾病进行聚合
    for i in range(len(result_list)):
        if 'organ' in result_list[i][input_list[i]] and 'center' in result_list[i][input_list[i]]:  # 既有部位又有主导词
            count += 1

            # 把相同主导词的疾病进行聚合，记录他们的index
            center = input_list[i][
                    result_list[i][input_list[i]]['center'][0][0]: result_list[i][input_list[i]]['center'][0][1]]
            organ = input_list[i][
                    result_list[i][input_list[i]]['organ'][0][0]: result_list[i][input_list[i]]['organ'][0][1]]
            organ_embed = embed.gen_sentence_embedding(organ)
            if center not in same_center_dict.keys():
                same_center_dict[center] = {}
                same_center_dict[center]['name'] = [input_list[i]]
                same_center_dict[center]['organ'] = [organ]
                same_center_dict[center]['organ_embed'] = [organ_embed]
            else:
                same_center_dict[center]['name'].append(input_list[i])
                same_center_dict[center]['organ'].append(organ)
                same_center_dict[center]['organ_embed'].append(organ_embed)
    print(count)
    position_embeddings = [embed.gen_sentence_embedding(pos) for pos in config['positions']]
    position_embeddings = np.array(position_embeddings)

    # 卡阈值，匹配标准部位时，余弦相似度得分最高的部位，这个分数必须大于0.75才保留，所以会有那一堆带left的变量
    output_list = []
    for center in same_center_dict.keys():
        name_list_left, organ_list_left, map_positions_left = [], [], []  # 筛选剩下的name，organ和positions里map到的部位
        organ_embeddings = np.array(same_center_dict[center]['organ_embed'])
        cos_scores = cosine_similarity(organ_embeddings, position_embeddings)
        max_coses = np.max(cos_scores, axis=1)
        max_indexes = np.argmax(cos_scores, axis=1)
        assert len(max_indexes == len(organ_embeddings))
        for i in range(len(same_center_dict[center]['name'])):
            if max_coses[i] > 0.75:  # threshold
                name_list_left.append(same_center_dict[center]['name'][i])
                organ_list_left.append(same_center_dict[center]['organ'][i])
                map_positions_left.append(position_list[max_indexes[i]])

        output_local = []  # 为了降低去重的查找复杂度
        for i in range(len(name_list_left)):  # name_list_left, organ_list_left, map_positions_left
            for j in range(50):
                pair_index = random.randint(0, len(name_list_left)-1)
                pair_disease = name_list_left[pair_index]
                pair_position = map_positions_left[pair_index]
                if is_upper_or_lower_position(config['positions'], map_positions_left[i], map_positions_left[pair_index]):
                    candidate = {
                        'text': pair_disease,
                        'normalized_result': name_list_left[i],
                        'source': 'upper_or_lower_position_from_icd'
                    }
                    if not already_exist(output_local, candidate) and name_list_left[i] != pair_disease:  # 北京版icd中有重复的，所以筛选的时候要去重
                        output_local.append(candidate)
                        break
        output_list = output_list + output_local

    print(output_list)
    print('len:', len(output_list))
    return output_list


def replace_position_real_pos_from_trainset(config, ner):
    """
    在CDN训练集中找真实的替换疾病，主导词相同，部位是上下位
    :param icd_dict:
    :param positions:
    :param ner:
    :return:
    """
    input_list = []
    temp = [item['text'] for item in config['target_train_set']]
    clinical2icd = {item['text']: item['normalized_result'] for item in config['target_train_set']}  # 训练集中临床名称作为key，icd名称作为value
    for disease in temp:
        if filter_clinical_term(disease, mode=config['type']):
            input_list.append(disease)

    embed = Embedding()
    count = 0
    position_list = [p for p in config['positions'].keys()]

    tik = time.time()
    result_list = ner.parser_list(input_list)
    if config['type'] == 'operation':
        result_list = fix_operation_ner_result(result_list, input_list)
    tok = time.time()
    assert len(input_list) == len(result_list)

    input_has_all, organ_list, organ_embeddings = [], [], []
    same_center_dict = {}  # 相同主导词的疾病进行聚合
    for i in range(len(result_list)):
        if 'organ' in result_list[i][input_list[i]] and 'center' in result_list[i][input_list[i]]:  # 既有部位又有主导词
            count += 1
            # input_has_all.append(input_list[i])
            # organ = input_list[i][
            #         result_list[i][input_list[i]]['organ'][0][0]: result_list[i][input_list[i]]['organ'][0][1]]
            # organ_list.append(organ)
            # organ_embed = embed.gen_sentence_embedding(organ)
            # organ_embeddings.append(organ_embed)

            # 把相同主导词的疾病进行聚合，记录他们的index
            center = input_list[i][
                    result_list[i][input_list[i]]['center'][0][0]: result_list[i][input_list[i]]['center'][0][1]]
            organ = input_list[i][
                    result_list[i][input_list[i]]['organ'][0][0]: result_list[i][input_list[i]]['organ'][0][1]]
            organ_embed = embed.gen_sentence_embedding(organ)
            if center not in same_center_dict.keys():
                same_center_dict[center] = {}
                same_center_dict[center]['name'] = [input_list[i]]
                same_center_dict[center]['organ'] = [organ]
                same_center_dict[center]['organ_embed'] = [organ_embed]
            else:
                same_center_dict[center]['name'].append(input_list[i])
                same_center_dict[center]['organ'].append(organ)
                same_center_dict[center]['organ_embed'].append(organ_embed)
    print(count)
    position_embeddings = [embed.gen_sentence_embedding(pos) for pos in config['positions']]
    position_embeddings = np.array(position_embeddings)

    # 卡阈值，匹配标准部位时，余弦相似度得分最高的部位，这个分数必须大于0.75才保留，所以会有那一堆带left的变量
    output_list = []
    for center in same_center_dict.keys():
        name_list_left, organ_list_left, map_positions_left = [], [], []  # 筛选剩下的name，organ和positions里map到的部位
        organ_embeddings = np.array(same_center_dict[center]['organ_embed'])
        cos_scores = cosine_similarity(organ_embeddings, position_embeddings)
        max_coses = np.max(cos_scores, axis=1)
        max_indexes = np.argmax(cos_scores, axis=1)
        assert len(max_indexes == len(organ_embeddings))
        for i in range(len(same_center_dict[center]['name'])):
            if max_coses[i] > 0.75:  # threshold
                name_list_left.append(same_center_dict[center]['name'][i])
                organ_list_left.append(same_center_dict[center]['organ'][i])
                map_positions_left.append(position_list[max_indexes[i]])

        output_local = []  # 为了降低去重的查找复杂度
        source_name = 'cdn' if config['type'] == 'disease' else 'yidu'
        for i in range(len(name_list_left)):  # name_list_left, organ_list_left, map_positions_left
            for j in range(50):
                pair_index = random.randint(0, len(name_list_left)-1)
                pair_disease = name_list_left[pair_index]
                pair_position = map_positions_left[pair_index]
                if is_upper_or_lower_position(config['positions'], map_positions_left[i], map_positions_left[pair_index]):
                    candidate = {
                        'text': pair_disease,
                        'normalized_result': clinical2icd[name_list_left[i]],
                        'source': f'upper_or_lower_position_from_{source_name}'
                    }
                    if not already_exist(output_local, candidate) and name_list_left[i] != pair_disease:  # 北京版icd中有重复的，所以筛选的时候要去重
                        output_local.append(candidate)
                        break
        output_list = output_list + output_local

    print(output_list)
    print('len:', len(output_list))
    return output_list


def replace_axis_fake_pos(config, ner, same_item, replace_item):
    """
    在标准疾病中替换轴心词
    举例：固定主导词，替换部位
        在标准疾病中找主导词相同，部位不同的疾病A和B，把A的部位替换成B的，然后拿B作为标签
    轴心：
        主导词；center
        部位：organ
        疾病性质：quality
    :param icd_dict: 标准icd列表
    :param ner: 疾病ner模块
    :param same_item: 不变的轴心
    :param replace_item: 进行暴力替换的轴心
    :return:
    """
    print(f'axis to freeze: {same_item}\naxis to replace: {replace_item}')
    count = 0
    input_list = []
    if config['type'] == 'disease':
        for icd_key in config['icd_dict'].keys():
            if (icd_key < 'O' or icd_key[0] in ['Q', 'R', 'S', 'T']) \
                    and filter_icd_disease(icd_key, config['icd_dict'][icd_key]) == True\
                    and '(' not in config['icd_dict'][icd_key] and ')' not in config['icd_dict'][icd_key]\
                    and '[' not in config['icd_dict'][icd_key] and ']' not in config['icd_dict'][icd_key]:  # 开头从E到N的所有疾病都被作为候选疾病进行筛选
                input_list.append(config['icd_dict'][icd_key])
    else:
        for icd_key in config['icd_dict'].keys():
            if filter_icd_operation(icd_key, config['icd_dict'][icd_key]):  # 开头从E到N的所有疾病都被作为候选疾病进行筛选
                input_list.append(config['icd_dict'][icd_key])
    tik = time.time()
    result_list = ner.parser_list(input_list)
    if config['type'] == 'operation':
        result_list = fix_operation_ner_result(result_list, input_list)
    tok = time.time()
    assert len(input_list) == len(result_list)

    same_item_dict = {}  # 相同主导词的疾病进行聚合
    for i in range(len(result_list)):
        # if same_item in result_list[i][input_list[i]] and replace_item in result_list[i][input_list[i]]:
        if same_item in result_list[i][input_list[i]] and replace_item in result_list[i][input_list[i]]\
                and len(result_list[i][input_list[i]][same_item]) == 1 and len(result_list[i][input_list[i]][replace_item]) == 1:  # 既有部位又有主导词
            count += 1
            # input_has_all.append(input_list[i])
            # organ = input_list[i][
            #         result_list[i][input_list[i]]['organ'][0][0]: result_list[i][input_list[i]]['organ'][0][1]]
            # organ_list.append(organ)
            # organ_embed = embed.gen_sentence_embedding(organ)
            # organ_embeddings.append(organ_embed)

            # 把相同主导词的疾病进行聚合，记录他们的index
            item_same = input_list[i][
                    result_list[i][input_list[i]][same_item][0][0]: result_list[i][input_list[i]][same_item][0][1]]
            item_replace = input_list[i][
                    result_list[i][input_list[i]][replace_item][0][0]: result_list[i][input_list[i]][replace_item][0][1]]
            # 这里单独记录organ是因为如果替换疾病性质的话，要求主导词和部位均相同
            if 'organ' in result_list[i][input_list[i]]:
                organ = input_list[i][
                        result_list[i][input_list[i]]['organ'][0][0]: result_list[i][input_list[i]]['organ'][0][1]]
            else:
                organ = None
            if item_same not in same_item_dict.keys():
                same_item_dict[item_same] = {}  # 主导词相同部位相同，才能替换性质
                same_item_dict[item_same]['name'] = [input_list[i]]
                same_item_dict[item_same]['replace_item'] = [item_replace]
                same_item_dict[item_same]['organ'] = [organ]
            else:
                same_item_dict[item_same]['name'].append(input_list[i])
                same_item_dict[item_same]['replace_item'].append(item_replace)
                same_item_dict[item_same]['organ'].append(organ)
    print(count)

    count = 0
    output_list = []
    for item_same in same_item_dict.keys():
        for i, label in enumerate(same_item_dict[item_same]['name']):
            text_list = []
            for j, disease in enumerate(same_item_dict[item_same]['name']):
                if i != j and same_item_dict[item_same]['replace_item'][j] != same_item_dict[item_same]['replace_item'][i]:
                    origin = same_item_dict[item_same]['replace_item'][j]
                    replace = same_item_dict[item_same]['replace_item'][i]
                    # 有时候会把 肾癌 替换成 结石，但其实是肾结石，所以要纠正一下
                    if same_item == 'organ' and replace_item == 'center':
                        if item_same in origin and item_same not in replace:
                            if config['type'] == 'disease':
                                replace = item_same + replace
                    # 替换疾病性质的话，除了主导词相同外，部位也要求相同
                    if same_item == 'center' and replace_item == 'quality':
                        if same_item_dict[item_same]['organ'][j] != None \
                                and same_item_dict[item_same]['organ'][i] != None\
                                and same_item_dict[item_same]['organ'][j] != same_item_dict[item_same]['organ'][i]:
                            continue  # 如果两个疾病中都存在部位，且部位还不相同，则跳过

                    text = disease.replace(origin, replace)
                    # 避免重复
                    if text not in text_list:
                        text_list.append(text)
                        output_list.append({
                            'text': text,
                            'normalized_result': label,
                            'source': f'replace_fake_{replace_item}_from_icd'
                        })
                        count += 1

    print(len(output_list))
    return output_list


def label_aug_position_shangxiawei(config, ner):
    """
    标签增强，从标准疾病中找到主导词相同，部位是上下位的疾病，并把临床疾病的部位替换
    :param icd_dict:
    :param positions:
    :param ner:
    :return:
    """
    # TODO: 改成上下位疾病：把上下部位的限制条件加上，删掉主导词部位都必须只有一个的限制，分别对应399，452，519行
    # 之所以要把它们分出来，是因为要过ner那个系统，所以要成一个disease list的形式
    train_text, train_label_text = [], []
    for item in config['target_train_set']:
        if filter_clinical_term(item['text'], mode=config['type']):  # 过滤掉临床名称中包含多个疾病的
            labels = item['normalized_result'].replace('"', '').replace('\\', '').split("##")
            for label in labels:
                train_text.append(item['text'])
                train_label_text.append(label)

    embed = Embedding()
    count = 0
    input_list = []
    position_list = [p for p in config['positions'].keys()]
    print(position_list)

    if config['type'] == 'disease':
        for icd_key in config['icd_dict'].keys():
            if (icd_key < 'O' or icd_key[0] in ['Q', 'R', 'S', 'T']) and filter_icd_disease(icd_key, config['icd_dict'][icd_key]) == True:  # 开头从E到N的所有疾病都被作为候选疾病进行筛选
                input_list.append(config['icd_dict'][icd_key])
    else:
        for icd_key in config['icd_dict'].keys():
            if filter_icd_operation(icd_key, config['icd_dict'][icd_key]):  # 开头从E到N的所有疾病都被作为候选疾病进行筛选
                input_list.append(config['icd_dict'][icd_key])
    tik = time.time()
    result_list = ner.parser_list(input_list)  # icd列表疾病的ner结果
    if config['type'] == 'operation':
        result_list = fix_operation_ner_result(result_list, input_list)
    train_text_ner = ner.parser_list(train_text)
    if config['type'] == 'operation':
        train_text_ner = fix_operation_ner_result(train_text_ner, train_text)
    train_label_text_ner = ner.parser_list(train_label_text)
    if config['type'] == 'operation':
        train_label_text_ner = fix_operation_ner_result(train_label_text_ner, train_label_text)
    tok = time.time()
    assert len(input_list) == len(result_list)
    assert len(train_text_ner) == len(train_label_text_ner)

    # position_embeddings = [embed.gen_sentence_embedding(pos) for pos in config['positions']]
    # position_embeddings = np.array(position_embeddings)

    # 找到训练集中label既有主导词又有部位，且text和label的部位名称相同，这样通过label替换完之后，就可以把text的部位也替换掉
    count = 0
    train_data_left = []  # 把position embedding一起记录下
    for i in range(len(train_text_ner)):
        # 判断label中是否既有主导词又有部位，且主导词和部位的数量都只有1
        # if 'organ' in train_label_text_ner[i][train_label_text[i]] and 'center' in train_label_text_ner[i][train_label_text[i]] \
        #         and len(train_label_text_ner[i][train_label_text[i]]['center']) == 1 and len(train_label_text_ner[i][train_label_text[i]]['organ']) == 1 \
        #         and 'organ' in train_text_ner[i][train_text[i]] and 'center' in train_text_ner[i][train_text[i]] \
        #         and len(train_text_ner[i][train_text[i]]['center']) == 1 and len(train_text_ner[i][train_text[i]]['organ']) == 1:
        if 'organ' in train_label_text_ner[i][train_label_text[i]] and 'center' in train_label_text_ner[i][train_label_text[i]]:
            # text中也有部位，且部位跟label中的一样
            label_center = train_label_text[i][
                          train_label_text_ner[i][train_label_text[i]]['center'][0][0]:
                          train_label_text_ner[i][train_label_text[i]]['center'][0][1]]
            label_organ = train_label_text[i][
                          train_label_text_ner[i][train_label_text[i]]['organ'][0][0]:
                          train_label_text_ner[i][train_label_text[i]]['organ'][0][1]]
            if 'organ' in train_text_ner[i][train_text[i]]:  # text中只需要有部位就行
                text_organ = train_text[i][
                              train_text_ner[i][train_text[i]]['organ'][0][0]:
                              train_text_ner[i][train_text[i]]['organ'][0][1]]
                # 这里不能通过简单的字符相等来判断，比如（乳，乳腺）（骨，肋骨）等这些情况就判断不出来
                # TODO: 用cosine similarity取最大的
                if label_organ == text_organ:
                    count += 1
                    print(train_text[i], train_label_text[i], label_center)
                    # 判断label这个疾病在不在上面筛选出的标准icd疾病中
                    if train_label_text[i] in input_list:
                        # train_text, train_label_text, train_text_ner, train_label_text_ner
                        train_data_left.append({
                            'text': train_text[i],
                            'label_text': train_label_text[i],
                            'text_organ': text_organ,
                            'label_organ': label_organ,
                            'label_center': label_center,
                            'text_organ_embed': embed.gen_sentence_embedding(text_organ),
                            'label_organ_embed': embed.gen_sentence_embedding(label_organ)  # 之后会拿label的embedding矩阵匹配标准部位
                        })  # 把position embedding一起记录下
    print(count)
    position_embeddings = [embed.gen_sentence_embedding(pos) for pos in config['positions']]
    position_embeddings = np.array(position_embeddings)
    # 给train_data_left中的每一个label_organ都匹配上标准部位
    organ_embeddings = np.array([item['label_organ_embed'] for item in train_data_left])
    cos_scores = cosine_similarity(organ_embeddings, position_embeddings)
    max_coses = np.max(cos_scores, axis=1)
    max_indexes = np.argmax(cos_scores, axis=1)
    assert len(max_indexes) == len(organ_embeddings)
    temp_list = []
    for i in range(len(train_data_left)):
        if max_coses[i] > 0.75:  # threshold
            train_data_left[i]['label_organ_map'] = position_list[max_indexes[i]]
            temp_list.append(train_data_left[i])
    train_data_left = temp_list

    count = 0
    input_has_all, organ_list, organ_embeddings = [], [], []
    same_icd_center_dict = {}  # 相同主导词的疾病进行聚合（标准icd列表）
    for i in range(len(result_list)):  # result_list: icd列表疾病的ner结果
        # if 'organ' in result_list[i][input_list[i]] and 'center' in result_list[i][input_list[i]] \
        #         and len(result_list[i][input_list[i]]['center']) == 1 and len(result_list[i][input_list[i]]['organ']) == 1:  # 既有部位又有主导词
        if 'organ' in result_list[i][input_list[i]] and 'center' in result_list[i][input_list[i]]:
            count += 1
            # input_has_all.append(input_list[i])
            # organ = input_list[i][
            #         result_list[i][input_list[i]]['organ'][0][0]: result_list[i][input_list[i]]['organ'][0][1]]
            # organ_list.append(organ)
            # organ_embed = embed.gen_sentence_embedding(organ)
            # organ_embeddings.append(organ_embed)

            # 把相同主导词的疾病进行聚合，记录他们的index
            center = input_list[i][
                    result_list[i][input_list[i]]['center'][0][0]: result_list[i][input_list[i]]['center'][0][1]]
            organ = input_list[i][
                    result_list[i][input_list[i]]['organ'][0][0]: result_list[i][input_list[i]]['organ'][0][1]]
            organ_embed = embed.gen_sentence_embedding(organ)
            if center not in same_icd_center_dict.keys():
                same_icd_center_dict[center] = {}
                same_icd_center_dict[center]['name'] = [input_list[i]]
                same_icd_center_dict[center]['organ'] = [organ]
                same_icd_center_dict[center]['organ_embed'] = [organ_embed]
            else:
                same_icd_center_dict[center]['name'].append(input_list[i])
                same_icd_center_dict[center]['organ'].append(organ)
                same_icd_center_dict[center]['organ_embed'].append(organ_embed)
    print(count)

    # icd标准疾病匹配标准部位，并且把最大分数小于0.75的给筛掉
    icd_same_center_left = {}
    for center in same_icd_center_dict.keys():
        icd_same_center_left[center] = {
            'name': [],
            'organ': [],
            'map_position': []
        }
        organ_embeddings = np.array(same_icd_center_dict[center]['organ_embed'])
        cos_scores = cosine_similarity(organ_embeddings, position_embeddings)
        max_coses = np.max(cos_scores, axis=1)
        max_indexes = np.argmax(cos_scores, axis=1)
        assert len(max_indexes) == len(organ_embeddings)
        for i in range(len(same_icd_center_dict[center]['name'])):
            if max_coses[i] > 0.75:  # threshold
                icd_same_center_left[center]['name'].append(same_icd_center_dict[center]['name'][i])
                icd_same_center_left[center]['organ'].append(same_icd_center_dict[center]['organ'][i])
                icd_same_center_left[center]['map_position'].append(position_list[max_indexes[i]])

    # 对于每一个训练集数据，找找看有没有可替换的上下位部位
    print('--------------------------------------------------')
    output_list = []
    analysis_dict = []
    for item in train_data_left:
        original_organ = item['label_organ']
        center = item['label_center']
        if center in icd_same_center_left:  # 因为上面标准icd通过organ和center数量等，被筛掉了一波，所以可能某些主导词不在key中
            for i in range(len(icd_same_center_left[center]['name'])):  # organ是疾病中的真实部位，map_position是匹配到的标准部位
                icd_current_organ = icd_same_center_left[center]['map_position'][i]
                if is_upper_or_lower_position(config['positions'], item['label_organ_map'], icd_current_organ):
                # if True:
                    replaced_organ = icd_same_center_left[center]['organ'][i]
                    output_list.append({
                        'text': item['text'].replace(original_organ, replaced_organ),
                        'normalized_result': icd_same_center_left[center]['name'][i],
                        'source': 'label_aug_upper_or_lower_position'
                    })
    #                 print(item['text'], item['label_text'], item['text'].replace(original_organ, replaced_organ), icd_same_center_left[center]['name'][i])
    #                 analysis_dict.append({
    #                     '原临床名称': item['text'],
    #                     '原标准名称': item['label_text'],
    #                     '替换后的临床名称': item['text'].replace(original_organ, replaced_organ),
    #                     '替换后的标准名称': icd_same_center_left[center]['name'][i],
    #                 })
    # df = pd.DataFrame(analysis_dict)
    # df.to_excel('替换部位.xls')
    print(len(output_list))
    return output_list


def label_aug_position_all(config, ner):
    """
    标签增强，从标准疾病中找到主导词相同，部位不同的疾病，并把临床疾病的部位替换
    :param icd_dict:
    :param positions:
    :param ner:
    :return:
    """
    # TODO: 改成上下位疾病：把上下部位的限制条件加上，删掉主导词部位都必须只有一个的限制，分别对应399，452，519行
    # 之所以要把它们分出来，是因为要过ner那个系统，所以要成一个disease list的形式
    train_text, train_label_text = [], []
    for item in config['target_train_set']:
        if filter_clinical_term(item['text'], mode=config['type']):  # 过滤掉临床名称中包含多个疾病的
            labels = item['normalized_result'].replace('"', '').replace('\\', '').split("##")
            for label in labels:
                train_text.append(item['text'])
                train_label_text.append(label)

    embed = Embedding()
    count = 0
    input_list = []
    position_list = [p for p in config['positions'].keys()]

    if config['type'] == 'disease':
        for icd_key in config['icd_dict'].keys():
            if (icd_key < 'O' or icd_key[0] in ['Q', 'R', 'S', 'T']) and filter_icd_disease(icd_key, config['icd_dict'][
                icd_key]) == True:  # 开头从E到N的所有疾病都被作为候选疾病进行筛选
                input_list.append(config['icd_dict'][icd_key])
    else:
        for icd_key in config['icd_dict'].keys():
            if filter_icd_operation(icd_key, config['icd_dict'][icd_key]):  # 开头从E到N的所有疾病都被作为候选疾病进行筛选
                input_list.append(config['icd_dict'][icd_key])
    tik = time.time()
    result_list = ner.parser_list(input_list)  # icd列表疾病的ner结果
    if config['type'] == 'operation':
        result_list = fix_operation_ner_result(result_list, input_list)
    train_text_ner = ner.parser_list(train_text)
    if config['type'] == 'operation':
        train_text_ner = fix_operation_ner_result(train_text_ner, train_text)
    train_label_text_ner = ner.parser_list(train_label_text)
    if config['type'] == 'operation':
        train_label_text_ner = fix_operation_ner_result(train_label_text_ner, train_label_text)
    tok = time.time()
    assert len(input_list) == len(result_list)
    assert len(train_text_ner) == len(train_label_text_ner)

    # position_embeddings = [embed.gen_sentence_embedding(pos) for pos in config['positions']]
    # position_embeddings = np.array(position_embeddings)

    # 找到训练集中label既有主导词又有部位，且text和label的部位名称相同，这样通过label替换完之后，就可以把text的部位也替换掉
    count = 0
    train_data_left = []  # 把position embedding一起记录下
    for i in range(len(train_text_ner)):
        # 判断label中是否既有主导词又有部位，且主导词和部位的数量都只有1
        if 'organ' in train_label_text_ner[i][train_label_text[i]] and 'center' in train_label_text_ner[i][train_label_text[i]] \
                and len(train_label_text_ner[i][train_label_text[i]]['center']) == 1 and len(train_label_text_ner[i][train_label_text[i]]['organ']) == 1 \
                and 'organ' in train_text_ner[i][train_text[i]] and 'center' in train_text_ner[i][train_text[i]] \
                and len(train_text_ner[i][train_text[i]]['center']) == 1 and len(train_text_ner[i][train_text[i]]['organ']) == 1:
        # if 'organ' in train_label_text_ner[i][train_label_text[i]] and 'center' in train_label_text_ner[i][train_label_text[i]]:
            # text中也有部位，且部位跟label中的一样
            label_center = train_label_text[i][
                          train_label_text_ner[i][train_label_text[i]]['center'][0][0]:
                          train_label_text_ner[i][train_label_text[i]]['center'][0][1]]
            label_organ = train_label_text[i][
                          train_label_text_ner[i][train_label_text[i]]['organ'][0][0]:
                          train_label_text_ner[i][train_label_text[i]]['organ'][0][1]]
            if 'organ' in train_text_ner[i][train_text[i]]:  # text中只需要有部位就行
                text_organ = train_text[i][
                              train_text_ner[i][train_text[i]]['organ'][0][0]:
                              train_text_ner[i][train_text[i]]['organ'][0][1]]
                # 这里不能通过简单的字符相等来判断，比如（乳，乳腺）（骨，肋骨）等这些情况就判断不出来
                # TODO: 用cosine similarity取最大的
                if label_organ == text_organ:
                    count += 1
                    # print(train_text[i], train_label_text[i], label_center)
                    # 判断label这个疾病在不在上面筛选出的标准icd疾病中
                    if train_label_text[i] in input_list:
                        # train_text, train_label_text, train_text_ner, train_label_text_ner
                        train_data_left.append({
                            'text': train_text[i],
                            'label_text': train_label_text[i],
                            'text_organ': text_organ,
                            'label_organ': label_organ,
                            'label_center': label_center,
                            'text_organ_embed': embed.gen_sentence_embedding(text_organ),
                            'label_organ_embed': embed.gen_sentence_embedding(label_organ)  # 之后会拿label的embedding矩阵匹配标准部位
                        })  # 把position embedding一起记录下
    print(count)
    position_embeddings = [embed.gen_sentence_embedding(pos) for pos in config['positions']]
    position_embeddings = np.array(position_embeddings)
    # 给train_data_left中的每一个label_organ都匹配上标准部位
    organ_embeddings = np.array([item['label_organ_embed'] for item in train_data_left])
    cos_scores = cosine_similarity(organ_embeddings, position_embeddings)
    max_coses = np.max(cos_scores, axis=1)
    max_indexes = np.argmax(cos_scores, axis=1)
    assert len(max_indexes) == len(organ_embeddings)
    temp_list = []
    for i in range(len(train_data_left)):
        if max_coses[i] > 0.75:  # threshold
            train_data_left[i]['label_organ_map'] = position_list[max_indexes[i]]
            temp_list.append(train_data_left[i])
    train_data_left = temp_list

    count = 0
    input_has_all, organ_list, organ_embeddings = [], [], []
    same_icd_center_dict = {}  # 相同主导词的疾病进行聚合（标准icd列表）
    for i in range(len(result_list)):  # result_list: icd列表疾病的ner结果
        if 'organ' in result_list[i][input_list[i]] and 'center' in result_list[i][input_list[i]] \
                and len(result_list[i][input_list[i]]['center']) == 1 and len(result_list[i][input_list[i]]['organ']) == 1:  # 既有部位又有主导词
        # if 'organ' in result_list[i][input_list[i]] and 'center' in result_list[i][input_list[i]]:
            count += 1
            # input_has_all.append(input_list[i])
            # organ = input_list[i][
            #         result_list[i][input_list[i]]['organ'][0][0]: result_list[i][input_list[i]]['organ'][0][1]]
            # organ_list.append(organ)
            # organ_embed = embed.gen_sentence_embedding(organ)
            # organ_embeddings.append(organ_embed)

            # 把相同主导词的疾病进行聚合，记录他们的index
            center = input_list[i][
                    result_list[i][input_list[i]]['center'][0][0]: result_list[i][input_list[i]]['center'][0][1]]
            organ = input_list[i][
                    result_list[i][input_list[i]]['organ'][0][0]: result_list[i][input_list[i]]['organ'][0][1]]
            organ_embed = embed.gen_sentence_embedding(organ)
            if center not in same_icd_center_dict.keys():
                same_icd_center_dict[center] = {}
                same_icd_center_dict[center]['name'] = [input_list[i]]
                same_icd_center_dict[center]['organ'] = [organ]
                same_icd_center_dict[center]['organ_embed'] = [organ_embed]
            else:
                same_icd_center_dict[center]['name'].append(input_list[i])
                same_icd_center_dict[center]['organ'].append(organ)
                same_icd_center_dict[center]['organ_embed'].append(organ_embed)
    print(count)

    # icd标准疾病匹配标准部位，并且把最大分数小于0.75的给筛掉
    icd_same_center_left = {}
    for center in same_icd_center_dict.keys():
        icd_same_center_left[center] = {
            'name': [],
            'organ': [],
            'map_position': []
        }
        organ_embeddings = np.array(same_icd_center_dict[center]['organ_embed'])
        cos_scores = cosine_similarity(organ_embeddings, position_embeddings)
        max_coses = np.max(cos_scores, axis=1)
        max_indexes = np.argmax(cos_scores, axis=1)
        assert len(max_indexes) == len(organ_embeddings)
        for i in range(len(same_icd_center_dict[center]['name'])):
            if max_coses[i] > 0.75:  # threshold
                icd_same_center_left[center]['name'].append(same_icd_center_dict[center]['name'][i])
                icd_same_center_left[center]['organ'].append(same_icd_center_dict[center]['organ'][i])
                icd_same_center_left[center]['map_position'].append(position_list[max_indexes[i]])

    # 对于每一个训练集数据，找找看有没有可替换的上下位部位
    print('--------------------------------------------------')
    output_list = []
    analysis_dict = []
    for item in train_data_left:
        original_organ = item['label_organ']
        center = item['label_center']
        if center in icd_same_center_left:  # 因为上面标准icd通过organ和center数量等，被筛掉了一波，所以可能某些主导词不在key中
            for i in range(len(icd_same_center_left[center]['name'])):  # organ是疾病中的真实部位，map_position是匹配到的标准部位
                icd_current_organ = icd_same_center_left[center]['map_position'][i]
                # if is_upper_or_lower_position(config['positions'], item['label_organ_map'], icd_current_organ):
                if True:
                    replaced_organ = icd_same_center_left[center]['organ'][i]
                    output_list.append({
                        'text': item['text'].replace(original_organ, replaced_organ),
                        'normalized_result': icd_same_center_left[center]['name'][i],
                        'source': 'label_aug_position_all'
                    })
    #                 print(item['text'], item['label_text'], item['text'].replace(original_organ, replaced_organ), icd_same_center_left[center]['name'][i])
    #                 analysis_dict.append({
    #                     '原临床名称': item['text'],
    #                     '原标准名称': item['label_text'],
    #                     '替换后的临床名称': item['text'].replace(original_organ, replaced_organ),
    #                     '替换后的标准名称': icd_same_center_left[center]['name'][i],
    #                 })
    # df = pd.DataFrame(analysis_dict)
    # df.to_excel('替换部位.xls')
    print(len(output_list))
    return output_list


def label_aug_center(config, ner):
    """
    对于主导词的标签增强
    在标准疾病中找部位相同，主导词不同的疾病，并把对应标签进行替换
    :param icd_dict:
    :param positions:
    :param ner:
    :return:
    """
    # 之所以要把它们分出来，是因为要过ner那个系统，所以要成一个disease list的形式
    train_text, train_label_text = [], []
    for item in config['target_train_set']:
        if filter_clinical_term(item['text'], mode=config['type']):  # 过滤掉临床名称中包含多个疾病的
            labels = item['normalized_result'].replace('"', '').replace('\\', '').split("##")
            for label in labels:
                train_text.append(item['text'])
                train_label_text.append(label)

    embed = Embedding()
    count = 0
    input_list = []
    position_list = [p for p in config['positions'].keys()]
    print(position_list)

    if config['type'] == 'disease':
        for icd_key in config['icd_dict'].keys():
            if (icd_key < 'O' or icd_key[0] in ['Q', 'R', 'S', 'T']) and filter_icd_disease(icd_key, config['icd_dict'][
                icd_key]) == True:  # 开头从E到N的所有疾病都被作为候选疾病进行筛选
                input_list.append(config['icd_dict'][icd_key])
    else:
        for icd_key in config['icd_dict'].keys():
            if filter_icd_operation(icd_key, config['icd_dict'][icd_key]):  # 开头从E到N的所有疾病都被作为候选疾病进行筛选
                input_list.append(config['icd_dict'][icd_key])
    tik = time.time()
    result_list = ner.parser_list(input_list)  # icd列表疾病的ner结果
    if config['type'] == 'operation':
        result_list = fix_operation_ner_result(result_list, input_list)
    train_text_ner = ner.parser_list(train_text)
    if config['type'] == 'operation':
        train_text_ner = fix_operation_ner_result(train_text_ner, train_text)
    train_label_text_ner = ner.parser_list(train_label_text)
    if config['type'] == 'operation':
        train_label_text_ner = fix_operation_ner_result(train_label_text_ner, train_label_text)
    tok = time.time()
    assert len(input_list) == len(result_list)
    assert len(train_text_ner) == len(train_label_text_ner)

    # 找到训练集中label既有主导词又有部位，且text和label的部位名称相同，这样通过label替换完之后，就可以把text的部位也替换掉
    count = 0
    train_data_left = []  # 把position embedding一起记录下
    for i in range(len(train_text_ner)):
        # 判断label中是否既有主导词又有部位，且主导词和部位的数量都只有1
        # 这里把主导词和部位都控制在了一个，因为发现不控制的话，生成的数据很不靠谱
        if 'organ' in train_label_text_ner[i][train_label_text[i]] and 'center' in train_label_text_ner[i][train_label_text[i]] \
                and len(train_label_text_ner[i][train_label_text[i]]['center']) == 1 and len(train_label_text_ner[i][train_label_text[i]]['organ']) == 1 \
                and 'organ' in train_text_ner[i][train_text[i]] and 'center' in train_text_ner[i][train_text[i]] \
                        and len(train_text_ner[i][train_text[i]]['center']) == 1 and len(train_text_ner[i][train_text[i]]['organ']) == 1:  # 既有部位又有主导词
        # if 'organ' in train_label_text_ner[i][train_label_text[i]] and 'center' in train_label_text_ner[i][train_label_text[i]]:
            # text中也有部位，且部位跟label中的一样
            label_center = train_label_text[i][
                          train_label_text_ner[i][train_label_text[i]]['center'][0][0]:
                          train_label_text_ner[i][train_label_text[i]]['center'][0][1]]
            label_organ = train_label_text[i][
                          train_label_text_ner[i][train_label_text[i]]['organ'][0][0]:
                          train_label_text_ner[i][train_label_text[i]]['organ'][0][1]]
            if 'center' in train_text_ner[i][train_text[i]]:  # text中只需要有主导词就行，因为最终替换的是主导词
                text_center = train_text[i][
                              train_text_ner[i][train_text[i]]['center'][0][0]:
                              train_text_ner[i][train_text[i]]['center'][0][1]]
                if label_center == text_center:
                    count += 1
                    print(train_text[i], train_label_text[i], label_center)
                    # 判断label这个疾病在不在上面筛选出的标准icd疾病中
                    if train_label_text[i] in input_list:
                        # train_text, train_label_text, train_text_ner, train_label_text_ner
                        train_data_left.append({
                            'text': train_text[i],
                            'label_text': train_label_text[i],
                            'text_center': text_center,
                            'label_organ': label_organ,
                            'label_center': label_center,
                            # 'text_organ_embed': embed.gen_sentence_embedding(text_organ),
                            # 'label_organ_embed': embed.gen_sentence_embedding(label_organ)  # 之后会拿label的embedding矩阵匹配标准部位
                        })  # 把position embedding一起记录下
    print(count)
    position_embeddings = [embed.gen_sentence_embedding(pos) for pos in config['positions']]
    position_embeddings = np.array(position_embeddings)
    # # 给train_data_left中的每一个label_organ都匹配上标准部位
    # organ_embeddings = np.array([item['label_organ_embed'] for item in train_data_left])
    # cos_scores = cosine_similarity(organ_embeddings, position_embeddings)
    # max_coses = np.max(cos_scores, axis=1)
    # max_indexes = np.argmax(cos_scores, axis=1)
    # assert len(max_indexes) == len(organ_embeddings)
    # temp_list = []
    # for i in range(len(train_data_left)):
    #     if max_coses[i] > 0.75:  # threshold
    #         train_data_left[i]['label_organ_map'] = position_list[max_indexes[i]]
    #         temp_list.append(train_data_left[i])
    # train_data_left = temp_list

    count = 0
    input_has_all, organ_list, organ_embeddings = [], [], []
    same_icd_organ_dict = {}  # 相同部位的疾病进行聚合（标准icd列表）
    for i in range(len(result_list)):  # result_list: icd列表疾病的ner结果
        if 'organ' in result_list[i][input_list[i]] and 'center' in result_list[i][input_list[i]] \
                and len(result_list[i][input_list[i]]['center']) == 1 and len(result_list[i][input_list[i]]['organ']) == 1:  # 既有部位又有主导词
        # if 'organ' in result_list[i][input_list[i]] and 'center' in result_list[i][input_list[i]]:
            count += 1
            # input_has_all.append(input_list[i])
            # organ = input_list[i][
            #         result_list[i][input_list[i]]['organ'][0][0]: result_list[i][input_list[i]]['organ'][0][1]]
            # organ_list.append(organ)
            # organ_embed = embed.gen_sentence_embedding(organ)
            # organ_embeddings.append(organ_embed)

            # 把相同主导词的疾病进行聚合，记录他们的index
            center = input_list[i][
                    result_list[i][input_list[i]]['center'][0][0]: result_list[i][input_list[i]]['center'][0][1]]
            organ = input_list[i][
                    result_list[i][input_list[i]]['organ'][0][0]: result_list[i][input_list[i]]['organ'][0][1]]
            # organ_embed = embed.gen_sentence_embedding(organ)
            if organ not in same_icd_organ_dict.keys():
                same_icd_organ_dict[organ] = {}
                same_icd_organ_dict[organ]['name'] = [input_list[i]]
                same_icd_organ_dict[organ]['center'] = [center]
                # same_icd_center_dict[center]['organ_embed'] = [organ_embed]
            else:
                same_icd_organ_dict[organ]['name'].append(input_list[i])
                same_icd_organ_dict[organ]['center'].append(center)
                # same_icd_center_dict[center]['organ_embed'].append(organ_embed)
    print(count)

    # # icd标准疾病匹配标准部位，并且把最大分数小于0.75的给筛掉
    # icd_same_center_left = {}
    # for center in same_icd_center_dict.keys():
    #     icd_same_center_left[center] = {
    #         'name': [],
    #         'organ': [],
    #         'map_position': []
    #     }
    #     organ_embeddings = np.array(same_icd_center_dict[center]['organ_embed'])
    #     cos_scores = cosine_similarity(organ_embeddings, position_embeddings)
    #     max_coses = np.max(cos_scores, axis=1)
    #     max_indexes = np.argmax(cos_scores, axis=1)
    #     assert len(max_indexes) == len(organ_embeddings)
    #     for i in range(len(same_icd_center_dict[center]['name'])):
    #         if max_coses[i] > 0.75:  # threshold
    #             icd_same_center_left[center]['name'].append(same_icd_center_dict[center]['name'][i])
    #             icd_same_center_left[center]['organ'].append(same_icd_center_dict[center]['organ'][i])
    #             icd_same_center_left[center]['map_position'].append(position_list[max_indexes[i]])

    # 对于每一个训练集数据，找找看有没有可替换的上下位部位
    print('--------------------------------------------------')
    output_list = []
    analysis_dict = []
    for item in train_data_left:
        organ = item['label_organ']
        original_center = item['label_center']
        if organ in same_icd_organ_dict:  # 因为上面标准icd通过organ和center数量等，被筛掉了一波，所以可能某些主导词不在key中
            for i in range(len(same_icd_organ_dict[organ]['name'])):  # organ是疾病中的真实部位，map_position是匹配到的标准部位
                if original_center != same_icd_organ_dict[organ]['center'][i]:
                    replaced_center = same_icd_organ_dict[organ]['center'][i]
                    assert item['text_center'] == item['label_center']
                    output_list.append({
                        'text': item['text'].replace(original_center, replaced_center),
                        'normalized_result': same_icd_organ_dict[organ]['name'][i],
                        'source': 'label_aug_center'
                    })
    #                 print(item['text'], item['label_text'], item['text'].replace(original_center, replaced_center), same_icd_organ_dict[organ]['name'][i])
    #                 analysis_dict.append({
    #                     '原临床名称': item['text'],
    #                     '原标准名称': item['label_text'],
    #                     '替换后的临床名称': item['text'].replace(original_center, replaced_center),
    #                     '替换后的标准名称': same_icd_organ_dict[organ]['name'][i],
    #                 })
    # df = pd.DataFrame(analysis_dict)
    # df.to_excel('替换主导词.xls')
    print(len(output_list))
    return output_list


def add_cdn_trainset(config):
    output_list = []
    for item in config['target_train_set']:
        item['source'] = 'CDN_training_set'
        output_list.append(item)

    return output_list
