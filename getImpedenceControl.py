import os
import re
from collections import Counter
import json


class MIPI:
    def __init__(self, file_path):
        self.file_path = file_path
        self.text_data, self.exact_data = self.read_file()
        self.data_format = self.format_data()

    def read_file(self):
        netname_lines = []
        extracted_data = []

        with open(self.file_path, 'r') as file:
            lines = file.readlines()
            for line in lines:
                if line.replace(" ", "").strip().startswith("NetName:MIPI"):
                    netname_lines.append(line.strip().split(":")[-1].strip())
                    extracted_data.append(line.strip().split(":")[-1].strip())
                if "cline" in line:
                    parts = line.split("cline", 1)[1].strip().split()
                    if parts:
                        extracted_data.append(parts[0])
            return netname_lines, extracted_data

    def format_data(self):
        groups = {}
        for item in self.text_data:
            key = "lane" + re.search(r'\d+', item.split('_')[2]).group()
            if key not in groups:
                groups[key] = []
            groups[key].append(item)

        result = {}
        for lane, components in groups.items():
            result[lane] = {}
            for component in components:
                result[lane][component] = ''

        for lane in result:
            for key in result[lane]:
                if key in self.exact_data:
                    index = self.exact_data.index(key)
                    values = []
                    for next_index in range(index + 1, len(self.exact_data)):
                        if self.exact_data[next_index] in result['lane2'] or self.exact_data[next_index] in result[
                            'lane1'] or self.exact_data[next_index] in result['lane0']:
                            break
                        values.append(self.exact_data[next_index])
                    counts = Counter(values)
                    if any(count > 1 for count in counts.values()):
                        result[lane][key] = [item for item, count in counts.items() if count > 1]
                    else:
                        result[lane][key] = values

        dimension = [len(groups)]
        for key, value in groups.items():
            dimension.append(len(value))

        for lane in result:
            for key in result[lane]:
                result[lane][key] = {'net-layer': result[lane][key]}
        result['dimension'] = dimension
        return result

class MIPI_C_PHY:
    def __init__(self, file_path):
        self.file_path = file_path
        self.results = self.get_width()

    def get_width(self):
        # pattern = re.compile(r'Part of net:       CSI\d+_[ABC]\d+_LN\d+_P_P\d+')
        pattern = re.compile(r'Part of net:\s+(CSI\d+_[ABC]\d+_LN\d+_P_P\d+)')
        subclass_pattern = re.compile(r'subclass\s+(\S+)')
        width_pattern = re.compile(r'width \(([\d\.]+)\)')

        # 初始化空字典存储结果
        results = {}
        try:
            with open(self.file_path, 'r') as file:
                current_item_content = ''
                current_item_number = ''

                for line in file:
                    # 检查是否为新项开始
                    if line.startswith('Item'):
                        # 处理上一个Item内容
                        if current_item_content:
                            part_of_net_match = pattern.search(current_item_content)
                            if part_of_net_match:
                                subclass = subclass_pattern.search(current_item_content)
                                widths = set(width_pattern.findall(current_item_content))
                                # print("check: ", part_of_net_match.group())
                                results[current_item_number] = {
                                    'part_of_net': part_of_net_match.group(1),
                                    'subclass': subclass.group(1) if subclass else 'Not found',
                                    'widths': list(widths)
                                }

                        # 初始化新项
                        current_item_number = line.split(' ')[1].strip()
                        current_item_content = line
                    else:
                        current_item_content += line

                # 处理最后一个Item
                if current_item_content:
                    part_of_net_match = pattern.search(current_item_content)
                    if part_of_net_match:
                        subclass = subclass_pattern.search(current_item_content)
                        widths = set(width_pattern.findall(current_item_content))
                        results[current_item_number] = {
                            'part_of_net': part_of_net_match.group(1),
                            'subclass': subclass.group(1) if subclass else 'Not found',
                            'widths': list(widths)
                        }
            return results
        except FileNotFoundError:
            print(f"The file {self.file_path} does not exist. Please check the file path.")

class MIPI_D_PHY:
    def __init__(self, file_path):
        self.file_path = file_path
        self.results = self.get_width()

    def get_width(self):
        # pattern = re.compile(r'Part of net:       CSI\d+_[ABC]\d+_LN\d+_M_P\d+|Part of net:       CSI\d+_[ABC]\d+_CLK_M_P\d+')
        pattern = re.compile(r'Part of net:\s+(CSI\d+_[ABC]\d+_LN\d+_M_P\d+)|Part of net:\s+(CSI\d+_[ABC]\d+_CLK_M_P\d+)')
        subclass_pattern = re.compile(r'subclass\s+(\S+)')
        width_pattern = re.compile(r'width \(([\d\.]+)\)')
    results = {}


net_path = r"C:\Users\11204\Desktop\yan\xiaomi\dataText\RCAMFPC-net.txt"
cline_path = r"C:\Users\11204\Desktop\code\Knowledge_Map\FPC_Kmap\FPCData\RCAMFPC-cline.txt"
output_dir = r"C:\Users\11204\Desktop\code\Knowledge_Map\FPC_Kmap\extractedData"

# impedence control
mipi = MIPI(net_path)
result_data = mipi.data_format
output_Impedence = os.path.join(output_dir, 'impedenceControl.json')
os.makedirs(output_dir, exist_ok=True)
with open(output_Impedence, 'w', encoding='utf-8') as json_file:
    json.dump(result_data, json_file, indent=4, ensure_ascii=False)
print(f"已导出到 {output_Impedence}。")

# MIPI_C_PHY
mipi_c_phy = MIPI_C_PHY(cline_path)
output_MIPICPHY = os.path.join(output_dir, 'MIPI_C_PHY.json')
with open(output_MIPICPHY, 'w') as json_file:
    json.dump(mipi_c_phy.results, json_file, indent=4)
print(f"已导出到 {output_MIPICPHY}'")

# MIPI_D_PHY