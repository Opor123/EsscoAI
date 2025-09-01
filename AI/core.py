import os
import sys
import json
try:
    module_dir=os.path.dirname(__file__)
    file_path=os.path.join(module_dir,'..','Data','dataAI.json')
    with open(file_path,'r') as f:
        data=json.load(f)

except FileNotFoundError:
    print(f'Error Occurred at File not exists: {file_path}')
except json.JSONDecodeError:
    print("Error: could not decode JSON data")

for i in data:
    for j,k in i.items():
        if j=='question':
            print(f'{j}: {k}\n')