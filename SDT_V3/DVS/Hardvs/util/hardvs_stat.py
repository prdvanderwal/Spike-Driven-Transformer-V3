# 导入需要的模块
import os
import collections
import matplotlib.pyplot as plt

# 定义第0层文件夹的路径
root_dir = "/home/ligq/wkm/HARDVS/rawframes" # 你可以根据实际情况修改这个路径

image_extensions = [".png"]

# 创建一个空的字典，用来存储第二层文件夹的文件个数和出现次数
picture_count_dict = {}
numm = 0
num = 0
total_size = 0
for first_dir in os.listdir(root_dir):
    # 拼接第1层文件夹的完整路径
    first_dir_path = os.path.join(root_dir, first_dir)
    # 遍历第1层文件夹下的所有第2层文件夹
    for second_dir in os.listdir(first_dir_path):
        # 拼接第2层文件夹的完整路径
        second_dir_path = os.path.join(first_dir_path, second_dir)
        numm += 1
        # 遍历第2层文件夹下的所有文件和子文件夹
        for third_dir in os.listdir(second_dir_path):
            picture_count = 0
            # 拼接完整路径
            third_dir_path =  os.path.join(second_dir_path, third_dir)
            # print(third_dir_path)
            num += 1
            # 初始化文件个数为0
            # print(third_dir_path)
            for file_or_dir in os.listdir(third_dir_path):
                file_or_dir_path = os.path.join(third_dir_path, file_or_dir)
                # print(file_or_dir_path)
                # print(file_or_dir_path)
                # 判断是否是文件，如果是，文件个数加1
                if os.path.isfile(file_or_dir_path):
                    picture_count += 1
                    total_size += os.path.getsize(file_or_dir_path)
                    # 把文件个数作为键，出现次数作为值，存入字典中
                    # 如果字典中已经有这个键，就在原来的值上加1；如果没有，就设为1
            if picture_count in picture_count_dict:
                picture_count_dict[picture_count] += 1
            else:
                picture_count_dict[picture_count] = 1
                    # picture_count_dict[picture_count] = picture_count_dict.get(picture_count, 0) + 1

# 打印字典中的内容，查看结果
print(picture_count_dict)
print(numm)
print(num)
total_sum = sum(picture_count_dict.values())

print("Total sum of values:", total_sum)


sorted_dict = dict(sorted(picture_count_dict.items(), key=lambda item: item[1], reverse=True))

print(sorted_dict)

# 使用 sorted() 对字典的 items 进行排序，key 参数指定按键排序
key_sorted_dict = dict(sorted(picture_count_dict.items(), key=lambda item: item[0], reverse=False))

print(key_sorted_dict)

# 使用matplotlib模块绘制柱状图，可视化结果
# 创建一个新的图形对象
plt.figure()
# 设置图形标题
plt.title("The distribution of picture counts in each sample")
# 设置x轴标签
plt.xlabel("Picture count")
# 设置y轴标签
plt.ylabel("Frequency")
# 使用字典中的键和值作为x轴和y轴的数据，绘制柱状图
plt.bar(picture_count_dict.keys(), picture_count_dict.values())
# 保存图像为文件
plt.savefig('/home/ligq/wkm/SDSA_v2/hardvs_stat.png')
# 显示图形
plt.show()


import os
print(f"Folder size: {total_size} bytes")
