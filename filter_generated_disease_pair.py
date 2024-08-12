"""
This file contains the code to filter out the low-quality disease pairs.
It mainly leverages the following two techniques:
    n-gram matching
    semantic (cosine) similarity

Important: This file only filters data from AR methods.
"""

import json
import re
import random
from tqdm import tqdm

from data_aug_methods import Embedding


def ngram_match(s1, s2):
    common_ngrams_count = 0
    short_str_ngrams_count = 0

    # find the shorter string
    if len(s1) > len(s2):
        short_str, long_str = s2, s1
    else:
        short_str, long_str = s1, s2

    def clean_text(text):
        text = re.sub(r'\s+', '', text)  # 去除空格和换行符
        return text

    def generate_ngrams(text, n):
        text = clean_text(text)
        n_grams = [text[i:i + n] for i in range(len(text) - n + 1)]
        return n_grams

    for i in range(1, len(short_str)+1):
        ngrams_short = generate_ngrams(short_str, i)
        ngrams_long = generate_ngrams(long_str, i)
        common_ngrams = set(ngrams_short).intersection(ngrams_long)

        common_ngrams_count += len(common_ngrams)
        short_str_ngrams_count += len(ngrams_short)

    return common_ngrams_count / short_str_ngrams_count


def cosine_similarity_match_sentence_transformer(s1, s2):
    """
    Calculate the cosine similarity between two Chinese sentences.
    :param s1:
    :param s2:
    :return:
    """
    from sentence_transformers import SentenceTransformer
    # choose a model that is suitable for Chinese and have good performance
    model = SentenceTransformer('./model/bert_base_chinese')

    def clean_text(text):
        text = re.sub(r'\s+', '', text)  # 去除空格和换行符
        return text

    s1 = clean_text(s1)
    s2 = clean_text(s2)

    embeddings = model.encode([s1, s2])
    from sklearn.metrics.pairwise import cosine_similarity
    cosine_score = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]

    return cosine_score


def cosine_similarity_match_vocab(s1, s2):
    embed = Embedding()

    def clean_text(text):
        text = re.sub(r'\s+', '', text)  # 去除空格和换行符
        return text

    s1 = clean_text(s1)
    s2 = clean_text(s2)

    s1_embed = embed.gen_sentence_embedding(s1)
    s2_embed = embed.gen_sentence_embedding(s2)

    from sklearn.metrics.pairwise import cosine_similarity
    cosine_score = cosine_similarity([s1_embed], [s2_embed])[0][0]

    return cosine_score


if __name__ == '__main__':
    # # testing
    # cosine_similarity_match('今天天气不错', '今天天气很好')

    ngram_threshold = 0.7
    cosine_threshold = 0.8
    filtered_data = []

    allowed_category = [
        'replace_fake_organ_from_icd',
        'replace_fake_center_from_icd',
        'replace_fake_quality_from_icd',
        'label_aug_position_all',
        'label_aug_center',
        'label_aug_upper_or_lower_position'
    ]

    # label_aug_upper_or_lower_position类别不做变换，因为这个类别的数据本身质量较高
    changed_category = [
        'replace_fake_organ_from_icd',
        'replace_fake_center_from_icd',
        'replace_fake_quality_from_icd',
        'label_aug_position_all',
        'label_aug_center'
    ]

    # # 对于所有从icd进行变换的疾病，把文字多的变成unnormalized disease，字数少的变成normalized disease
    # # 这样符合直觉：一般医生写的疾病都会比较长，与训练集相符
    # swaped_category = [
    #     'replace_fake_organ_from_icd',
    #     'replace_fake_center_from_icd',
    #     'replace_fake_quality_from_icd'
    # ]

    # load the data
    # load json file ./data/CHIP-CDN_DA_v3.0/CHIP-CDN_all.json
    with open('./data/CHIP-CDN_DA_v3.0/CHIP-CDN_all.json', 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    # count = 0
    # random.shuffle(raw_data)
    for data_pair in tqdm(raw_data):  # unnormalized-normalized disease pair
        # # for testing purposes only: process the first 100k samples
        # count += 1
        # if count > 100000:
        #     break

        if data_pair['source'] in allowed_category:
            if data_pair['source'] in changed_category:
                # process
                ngram_score = ngram_match(data_pair['text'], data_pair['normalized_result'])
                cosine_score = cosine_similarity_match_vocab(data_pair['text'], data_pair['normalized_result'])
                # satisfy at least one condition
                if (ngram_score >= ngram_threshold or cosine_score >= cosine_threshold) and len(data_pair['text']) > len(data_pair['normalized_result']):
                    data_pair['ngram_score'] = ngram_score
                    data_pair['cosine_score'] = cosine_score
                    filtered_data.append(data_pair)
            else:
                if len(data_pair['text']) > len(data_pair['normalized_result']):
                    filtered_data.append(data_pair)

    # save filtered_data as json format
    with open('./test_filtered_data.json', 'w', encoding='utf-8') as f:
        json.dump(filtered_data, f, ensure_ascii=False, indent=4)
