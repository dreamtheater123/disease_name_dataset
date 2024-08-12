from data_aug_methods import *
from OperateNER.models import model_andBE_muilemb_bilstm_res_crf


def main():
    config = {
        'type': 'operation',
    }
    config = global_init(config)
    write = True

    output1 = gen_icd3to4_operate(config)
    output1_1 = gen_icd3_from_yidu(config)

    # # 以下两行都是疾病别称，手术用不上
    output3 = gen_operation_from_icd(config)

    ner = model_andBE_muilemb_bilstm_res_crf()
    ner.restore("./OperateNER/model_v1.1")

    output4 = replace_position_real_pos(config, ner)
    output4_1 = replace_position_real_pos_from_trainset(config, ner)
    output4_2 = replace_axis_fake_pos(config, ner, same_item='center', replace_item='organ')
    output4_3 = replace_axis_fake_pos(config, ner, same_item='organ', replace_item='center')  # TODO: debug一下里面的if
    # output4_4 = replace_axis_fake_pos(config, ner, same_item='center', replace_item='quality')

    output5 = label_aug_position_shangxiawei(config, ner)
    output6 = label_aug_position_all(config, ner)
    output7 = label_aug_center(config, ner)
    output8 = add_cdn_trainset(config)

    if write:
        output = output1 + output1_1 + \
                 output3 + \
                 output4 + output4_1 + output4_2 + output4_3 + \
                 output5 + output6 + output7 + output8

        # 计算一下每个类别的数量大小
        count_all = 0
        count_dict = {}
        for data in output:
            if data['source'] not in count_dict:
                count_dict[data['source']] = 1
                count_all += 1
            else:
                count_dict[data['source']] += 1
                count_all += 1
        for key in count_dict:
            print(key, count_dict[key])
        print(count_all)

        write_json_format('yidu_all.json', output)
        # split training set and testset
        random.shuffle(output)
        train_len = round(len(output) * 0.95)
        output_train = output[:train_len]
        output_dev = output[train_len:]
        write_json_format('yidu_train.json', output_train)
        write_json_format('yidu_dev.json', output_dev)
        print('len_train:', len(output_train), 'len_dev', len(output_dev))

        # df = pd.DataFrame(output)
        # df.rename(columns={'text': '原始词', 'normalized_result': '标准词'}, inplace=True)
        # order = ['原始词', '标准词', 'source']
        # df = df[order]
        # df.to_excel('all.xlsx')
        # print(df)
        #
        # # split training set and testset
        # random.shuffle(output)
        # train_len = round(len(output) * 0.95)
        # output_train = output[:train_len]
        # output_dev = output[train_len:]
        # print('len_train:', len(output_train), 'len_dev', len(output_dev))
        #
        # df_train = pd.DataFrame(output_train)
        # df_train.rename(columns={'text': '原始词', 'normalized_result': '标准词'}, inplace=True)
        # order = ['原始词', '标准词', 'source']
        # df_train = df_train[order]
        # df_train.to_excel('train.xlsx')
        #
        # df_dev = pd.DataFrame(output_dev)
        # df_dev.rename(columns={'text': '原始词', 'normalized_result': '标准词'}, inplace=True)
        # order = ['原始词', '标准词', 'source']
        # df_dev = df_dev[order]
        # df_dev.to_excel('val.xlsx')


if __name__ == '__main__':
    main()

