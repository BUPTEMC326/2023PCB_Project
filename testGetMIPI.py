import re

input_filename = r"C:\Users\11204\Desktop\code\Knowledge_Map\FPC_Kmap\FPCData\RCAMFPC-cline.txt"
output_filename = r"C:\Users\11204\Desktop\code\Knowledge_Map\FPC_Kmap\extractedData\output1.json"

# 初始化一个空字典来存储结果
results = {}

# 编译正则表达式模式
pattern = re.compile(r'Part of net:       CSI\d+_[ABC]\d+_LN\d+_P_P\d+')
subclass_pattern = re.compile(r'subclass\s+([^ \n]+)')
width_pattern = re.compile(r'width \(([\d\.]+)\)')

try:
    with open(input_filename, 'r') as file:
        current_item_content = ''
        current_item_number = ''
        item_started = False

        for line in file:
            # 检查是否是新项的开始
            if line.startswith('Item'):
                if item_started:
                    # 提取Part of net信息
                    part_of_net_match = pattern.search(current_item_content)
                    if part_of_net_match:
                        # 提取subclass信息
                        subclass_match = subclass_pattern.search(current_item_content)
                        subclass = subclass_match.group(1) if subclass_match else 'Not found'

                        # 提取宽度信息并去重
                        widths = width_pattern.findall(current_item_content)
                        unique_widths = set(widths)

                        # 将结果添加到字典中
                        results[current_item_number] = {
                            'part_of_net': part_of_net_match.group(),
                            'subclass': subclass,
                            'widths': ', '.join(unique_widths)
                        }

                # 开始新的Item
                current_item_number = line.split(' ')[1]
                current_item_content = line
                item_started = True
            else:
                # 继续构建当前Item的内容
                current_item_content += line

        # 处理最后一个Item
        if item_started:
            part_of_net_match = pattern.search(current_item_content)
            if part_of_net_match:
                subclass_match = subclass_pattern.search(current_item_content)
                subclass = subclass_match.group(1) if subclass_match else 'Not found'

                widths = width_pattern.findall(current_item_content)
                unique_widths = set(widths)

                results[current_item_number] = {
                    'part_of_net': part_of_net_match.group(),
                    'subclass': subclass,
                    'widths': ', '.join(unique_widths)
                }

    # 将结果写入TXT文件
    with open(output_filename, 'w') as txt_file:
        for item_number, result in results.items():
            txt_file.write(f"Item Number: {item_number}\n")
            txt_file.write(f"Part of net: {result['part_of_net']}\n")
            txt_file.write(f"Subclass: {result['subclass']}\n")
            txt_file.write(f"Widths: {result['widths']}\n\n")

    print(f'Results have been written to {output_filename}')

except FileNotFoundError:
    print(f"The file {input_filename} does not exist. Please check the file path.")
print("==========================================")
import re

# 使用原始字符串来避免无效的转义序列警告
input_filename = r"C:\Users\11204\Desktop\code\Knowledge_Map\FPC_Kmap\FPCData\RCAMFPC-cline.txt"
output_filename = r"C:\Users\11204\Desktop\code\Knowledge_Map\FPC_Kmap\extractedData\output2.json"

# 初始化一个空字典来存储结果
results = {}

# 编译正则表达式模式
pattern = re.compile(r'Part of net:       CSI\d+_[ABC]\d+_LN\d+_M_P\d+|Part of net:       CSI\d+_[ABC]\d+_CLK_M_P\d+')
subclass_pattern = re.compile(r'subclass\s+([^ \n]+)')
width_pattern = re.compile(r'width \(([\d\.]+)\)')

try:
    with open(input_filename, 'r') as file:
        current_item_content = ''
        current_item_number = ''
        item_started = False

        for line in file:
            # 检查是否是新项的开始
            if line.startswith('Item'):
                if item_started:
                    # 提取Part of net信息
                    part_of_net_match = pattern.search(current_item_content)
                    if part_of_net_match:
                        # 提取subclass信息
                        subclass_match = subclass_pattern.search(current_item_content)
                        subclass = subclass_match.group(1) if subclass_match else 'Not found'

                        # 提取宽度信息并去重
                        widths = width_pattern.findall(current_item_content)
                        unique_widths = set(widths)

                        # 将结果添加到字典中
                        results[current_item_number] = {
                            'part_of_net': part_of_net_match.group(),
                            'subclass': subclass,
                            'widths': ', '.join(unique_widths)
                        }

                # 开始新的Item
                current_item_number = line.split(' ')[1]
                current_item_content = line
                item_started = True
            else:
                # 继续构建当前Item的内容
                current_item_content += line

        # 处理最后一个Item
        if item_started:
            part_of_net_match = pattern.search(current_item_content)
            if part_of_net_match:
                subclass_match = subclass_pattern.search(current_item_content)
                subclass = subclass_match.group(1) if subclass_match else 'Not found'

                widths = width_pattern.findall(current_item_content)
                unique_widths = set(widths)

                results[current_item_number] = {
                    'part_of_net': part_of_net_match.group(),
                    'subclass': subclass,
                    'widths': ', '.join(unique_widths)
                }

    # 将结果写入TXT文件
    with open(output_filename, 'w') as txt_file:
        for item_number, result in results.items():
            txt_file.write(f"Item Number: {item_number}\n")
            txt_file.write(f"Part of net: {result['part_of_net']}\n")
            txt_file.write(f"Subclass: {result['subclass']}\n")
            txt_file.write(f"Widths: {result['widths']}\n\n")

    print(f'Results have been written to {output_filename}')

except FileNotFoundError:
    print(f"The file {input_filename} does not exist. Please check the file path.")

import json

# 使用原始字符串来避免无效的转义序列警告
input_filename = r"D:\Hejint\File\output-PCB\output_tro.txt"
output_filename = r"D:\Hejint\File\output-PCB\MIPI_D_PHY_WIDTH.json"

# 初始化一个字典来存储结果
results = {}

try:
    with open(input_filename, 'r') as file:
        current_item_number = None
        current_data = {}

        for line in file:
            line = line.strip()

            if line.startswith('Item Number:'):
                # 如果当前项目不为空，保存它
                if current_item_number is not None:
                    results[current_item_number] = current_data
                # 重置当前项目
                current_item_number = line.split(': ')[1].strip()
                current_data = {}
            elif line.startswith('Part of net:'):
                # 提取 "Part of net:" 后的字符串
                current_data["Part of net"] = line.split(': ', 1)[1].strip()
            elif line.startswith('Subclass:'):
                # 提取并添加子类信息
                current_data["subclass"] = line.split(': ')[1].strip()
            elif line.startswith('Widths:'):
                # 提取并添加宽度信息
                widths = line.split(': ')[1].strip()
                if "Widths" not in current_data:
                    current_data["Widths"] = []
                current_data["Widths"].append(widths)

        # 保存最后一个项目
        if current_item_number is not None:
            results[current_item_number] = current_data

    # 将结果写入JSON文件
    with open(output_filename, 'w') as json_file:
        json.dump(results, json_file, indent=4)

    print(f'Results have been written to {output_filename}')

except FileNotFoundError:
    print(f"The file {input_filename} does not exist. Please check the file path.")