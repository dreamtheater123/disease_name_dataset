import json

with open('./data/CHIP-CDN_DA_v3.0/CHIP-CDN_all.json', 'r', encoding='utf-8') as f:
    raw_data = json.load(f)

allowed_category = [
    'icd4to6',
    'icd4to6_from_cdn',
    'upper_or_lower_position_from_icd',
    'upper_or_lower_position_from_cdn'
]

filtered_data = []
for data_pair in raw_data:
    if data_pair['source'] in allowed_category:
        filtered_data.append(data_pair)

# save filtered_data as json format
with open('./data/CHIP-CDN_DA_v3.0_MGA/v3.0_MGA_all.json', 'w', encoding='utf-8') as f:
    json.dump(filtered_data, f, ensure_ascii=False, indent=4)
