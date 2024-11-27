import csv

# 读取txt文件
with open('/raid/ligq/wkm/sdsa_v2_hardvs/output_dir/hardvs_8bit_firing/19M_HARDVS_8bit_int_firing_rate.txt', 'r') as txt_file:
    lines = txt_file.readlines()

# 处理txt文件内容，提取字典数据
csv_data = []
current_main_key = None
current_sub_dict = {}
all_keys = []

for line in lines:
    line = line.strip()
    if line.endswith(':'):
        current_main_key = line[:-1]
        current_sub_dict = {}
        all_keys.append(current_main_key)
    elif line:
        key, value = map(str.strip, line.split(':'))
        current_sub_dict[key] = value
    else:
        csv_data.append({'main_key': current_main_key, **current_sub_dict})

# 动态构建csv_columns
unique_keys = [key for sub_dict in csv_data for key in sub_dict.keys()]
unique_keys = sorted(set(unique_keys), key=lambda x: unique_keys.index(x))  # 按照顺序排序
csv_columns = ['main_key'] + list(unique_keys)

# 写入csv文件
csv_file_path = '/raid/ligq/wkm/sdsa_v2_hardvs/output_dir/hardvs_8bit_firing/19M_HARDVS_8bit_int_firing_rate.csv'

with open(csv_file_path, 'w', newline='') as csv_file:
    writer = csv.DictWriter(csv_file, fieldnames=csv_columns)
    
    # 写入表头
    writer.writeheader()

    # 写入数据
    for data in csv_data:
        writer.writerow(data)

print(f'CSV文件已生成: {csv_file_path}')
