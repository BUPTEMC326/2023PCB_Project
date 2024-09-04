import os
import re
from openpyxl import Workbook

recordFile = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'importNET')
os.makedirs(recordFile, exist_ok=True)
pattern = r'NET "(.*?)" IMPORT=1'

print("------start------")

for filename in os.listdir('netChosen_odbjob'):
    if filename.endswith('.ncf'):
        wb = Workbook()
        ws = wb.active

        with open(os.path.join('netChosen_odbjob', filename), 'r') as f:
            lines = f.readlines()

        row_index = 1
        for line in lines:
            match = re.search(pattern, line)
            if match:
                network_name = match.group(1)
                ws.cell(row=row_index, column=1, value=network_name)
                row_index += 1

        output_filename = os.path.join(recordFile, f'{os.path.splitext(filename)[0]}.xlsx')
        wb.save(output_filename)

print("------done------")

