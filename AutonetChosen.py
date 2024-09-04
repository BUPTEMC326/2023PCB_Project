import os
import re
from openpyxl import Workbook

"""关于pcb板网络信息的.ncf文件，里面的数据是一行一行存储的，每一行代表该pcb板上的一条网络，
存储格式为NET "32KOUT_WIFI" IMPORT=0 CREATE_PORT=0 REFNET="GND"，其中NET表明这是这是一条网络，
net后面的"32KOUT_WIFI"是这条网络的名字，IMPORT代表该网络是否被选中，IMPORT=1是代表该网络被选中，IMPORT=0代表该网络未被选中；
CREATE_PORT代表是否为该条网络设置PORT，CREATE_PORT=0时代表为该条网络设置PORT，CREATE_PORT=1时代表不为该条网络设置PORT。"""

# 修改成自己网络信息存储的路径
input_file = "C:/Users/11204/Desktop/yan/LXsimulation/recordNet/netChosen_RK3588H1R0/1.ncf"
net_chosen_folder = "C:/Users/11204/Desktop/yan/LXsimulation/recordNet/netChosen_RK3588H1R0"
output_folder = "C:/Users/11204/Desktop/yan/LXsimulation/recordNet/importNET_RK3588H1R0"

os.makedirs(net_chosen_folder, exist_ok=True)
os.makedirs(output_folder, exist_ok=True)

print("-----reading file-----")
with open(input_file, 'r') as file:
    lines = file.readlines()
print("-----reading done-----")

# 按照50条网络为一组import
group_size = 50
num_groups = (len(lines) + group_size - 1) // group_size

for group_num in range(num_groups):
    modified_lines = []

    for i, line in enumerate(lines):
        if group_num * group_size <= i < (group_num + 1) * group_size:
            # 设置当前组的50条网络的IMPORT和CREATE_PORT为1
            modified_line = re.sub(r'IMPORT=\d', 'IMPORT=1', line)
            modified_line = re.sub(r'CREATE_PORT=\d', 'CREATE_PORT=1', modified_line)
        else:
            # 其他为0
            modified_line = re.sub(r'IMPORT=\d', 'IMPORT=0', line)
            modified_line = re.sub(r'CREATE_PORT=\d', 'CREATE_PORT=0', modified_line)

        modified_lines.append(modified_line)
    print("number of nets import: ", len(modified_lines))
    # 保存修改后的组到新的.ncf文件(只保存import的50条)
    output_ncf_path = os.path.join(net_chosen_folder, f'{group_num + 1}.ncf')
    with open(output_ncf_path, 'w') as output_ncf_file:
        output_ncf_file.writelines(modified_lines)

print("-----netChosen done-----")

pattern = r'NET "(.*?)" IMPORT=1'

# 遍历netChosen文件夹中的所有.ncf文件
for ncf_filename in sorted(os.listdir(net_chosen_folder)):
    if ncf_filename.endswith('.ncf'):
        ncf_path = os.path.join(net_chosen_folder, ncf_filename)

        wb = Workbook()
        ws = wb.active

        with open(ncf_path, 'r') as ncf_file:
            ncf_lines = ncf_file.readlines()

        row_index = 1
        for ncf_line in ncf_lines:
            match = re.search(pattern, ncf_line)
            if match:
                network_name = match.group(1)
                ws.cell(row=row_index, column=1, value=network_name)
                row_index += 1

        excel_filename = f'{os.path.splitext(ncf_filename)[0]}.xlsx'
        excel_path = os.path.join(output_folder, excel_filename)
        wb.save(excel_path)

        print(f"Excel '{excel_path}' saved.")

print("-----all done-----")

