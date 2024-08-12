import json
from tqdm import tqdm

######################ATTENTION######################
zero_shot_data = False
data_output_folder = 'CHIP-CDN_DA_v3.0_filtered_all'
######################ATTENTION######################
zero_shot_AR_sources = [
    'replace_fake_organ_from_icd',
    'replace_fake_center_from_icd',
    'replace_fake_quality_from_icd',
]
AR_sources = [
    'replace_fake_organ_from_icd',
    'replace_fake_center_from_icd',
    'replace_fake_quality_from_icd',
    'label_aug_position_all',
    'label_aug_center',
    'label_aug_upper_or_lower_position'
]
zero_shot_MGA_sources = [
    'icd4to6',
    'upper_or_lower_position_from_icd',
]
MGA_sources = [
    'icd4to6',
    'icd4to6_from_cdn',
    'upper_or_lower_position_from_icd',
    'upper_or_lower_position_from_cdn'
]

with open('./data/CHIP-CDN_DA_v3.0_filtered_AR/augmented_filtered_data_all.json', 'r', encoding='utf-8') as f:
    AR_data = json.load(f)
with open('./data/CHIP-CDN_DA_v3.0/CHIP-CDN_all.json', 'r', encoding='utf-8') as f:
    MGA_data = json.load(f)

grouped_data = []

# add AR data into grouped_data
if zero_shot_data:
    for data in AR_data:
        if data['source'] in zero_shot_AR_sources:
            grouped_data.append(data)
else:
    grouped_data = AR_data

# add MGA data into grouped_data
if zero_shot_data:
    for data in MGA_data:
        if data['source'] in zero_shot_MGA_sources:
            grouped_data.append(data)
else:
    for data in MGA_data:
        if data['source'] in MGA_sources:
            grouped_data.append(data)

# check resulted sources
if zero_shot_data:
    for data in grouped_data:
        assert data['source'] in zero_shot_AR_sources + zero_shot_MGA_sources, f'{data["source"]} not in zero-shot sources list.'
    print('All sources are in the zero-shot sources list.')
else:
    for data in grouped_data:
        assert data['source'] in AR_sources + MGA_sources, f'{data["source"]} not in filtered AR + MGA sources list.'
    print('All sources are in the MGA sources list.')
print(f'grouped data count: {len(grouped_data)}')
print(f'first ten samples in grouped data: {grouped_data[:10]}')

# save grouped_data as json format in './data/CHIP-CDN_DA_v3.0_filtered_all/CHIP-CDN_train.json'
with open(f'./data/{data_output_folder}/CHIP-CDN_train.json', 'w', encoding='utf-8') as f:
    json.dump(grouped_data, f, ensure_ascii=False, indent=4)
