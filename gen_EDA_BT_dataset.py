from textda.data_expansion import *
from textda.youdao_translate import *
import json
from tqdm import tqdm


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


def EDA():
    train_set = read_json('./data/CHIP-CDN/CHIP-CDN_train.json')
    print(len(train_set))
    output_list = []
    for item in tqdm(train_set):
        expan_list = data_expansion(item['text'])
        expan_list.append(item['text'])
        expan_list = list(set(expan_list))

        for disease in expan_list:
            output_list.append({
                'text': disease,
                'normalized_result': item['normalized_result']
            })
    write_json_format('./data/CHIP-CDN_EDA/CHIP-CDN_train.json', output_list)


def BT_prepare():
    train_set = read_json('./data/CHIP-CDN/CHIP-CDN_train.json')
    output_list = []
    for item in train_set:
        output_list.append(f'{item["normalized_result"]}\t{item["text"]}\n')
    print(len(output_list))

    with open('CHIP-CDN_huiyi.txt', 'w', encoding='utf-8') as f:
        f.writelines(output_list)


def BT():
    translate_batch('CHIP-CDN_huiyi.txt', batch_num=30)


def BT_after():
    with open('CHIP-CDN_huiyi.txt_youdao', 'r', encoding='utf-8') as f:
        lines = f.readlines()
    output_list = []
    for line in lines:
        seg = line.strip().split('\t')
        output_list.append({
            'text': seg[1],
            'normalized_result': seg[0]
        })
    write_json_format('./data/CHIP-CDN_BT/CHIP-CDN_train.json', output_list)


if __name__ == '__main__':
    BT_after()
    # print(translate_batch('test_huiyi.txt', batch_num=30))
    # TODO: 把trainer重新上传至服务器
