import os
import sys
import random
from data_aug_methods import *

# modify work path to load disease_ner module
temp_path = os.getcwd()
# os.chdir(temp_path + '/disease_ner')
sys.path = ['./disease_ner'] + sys.path
print(os.getcwd())
from disease_ner.models_new import model_andBE_muilemb_bilstm_res_crf

# os.chdir(temp_path)
print(os.getcwd())


def main():
    write = False
    config = {
        'type': 'disease',
    }
    config = global_init(config)
    print(config.keys())

    # output1 = gen_icd4to6_disease(config)  # MGA
    # output1_1 = gen_icd4_from_cdn(config)  # MGA
    #
    # output2 = gen_disease_another_name(config)  # 疾病别称，弃之不用
    # output3 = gen_disease_from_icd(config)  # 同样也是疾病别称，弃之不用

    ner = model_andBE_muilemb_bilstm_res_crf()
    ner.restore("./disease_ner/model_l3")

    # Real disease names, blur label
    output4 = replace_position_real_pos(config, ner)  # blur position
    output4_1 = replace_position_real_pos_from_trainset(config, ner)  # 这一类新增了一个限制条件，可能数量级会有区别
    # Fake disease names, real label
    output4_2 = replace_axis_fake_pos(config, ner, same_item='center', replace_item='organ')
    output4_3 = replace_axis_fake_pos(config, ner, same_item='organ', replace_item='center')
    output4_4 = replace_axis_fake_pos(config, ner, same_item='center', replace_item='quality')

    # label augmentation
    output5 = label_aug_position_shangxiawei(config, ner)  # TODO: check!上下位标签增强，输入输出同时增强，说明标签是对的上的？
    output6 = label_aug_position_all(config, ner)
    output7 = label_aug_center(config, ner)

    output8 = add_cdn_trainset(config)  # cdn训练集

    if write:
        output = output1 + output1_1 + \
                 output2 + output3 + \
                 output4 + output4_1 + output4_2 + output4_3 + output4_4 + \
                 output5 + output6 + output7 + output8

        # 计算一下每个类别的数量大小
        count_dict = {}
        for data in output:
            if data['source'] not in count_dict:
                count_dict[data['source']] = 1
            else:
                count_dict[data['source']] += 1
        for key in count_dict:
            print(key, count_dict[key])
        exit()

        write_json_format('CHIP-CDN_all.json', output)
        # split training set and testset
        random.shuffle(output)
        train_len = round(len(output) * 0.95)
        output_train = output[:train_len]
        output_dev = output[train_len:]
        write_json_format('CHIP-CDN_train.json', output_train)
        write_json_format('CHIP-CDN_dev.json', output_dev)
        print('len_train:', len(output_train), 'len_dev', len(output_dev))


if __name__ == '__main__':
    main()

