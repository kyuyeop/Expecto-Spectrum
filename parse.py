import pandas as pd
from tqdm import tqdm
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

romanDict2 = {1: 'I', 4: 'IX', 5: 'V', 9: 'IX', 10: 'X', 40: 'XL',
              50: 'L', 90: 'XC', 100: 'C', 400: 'CD', 500: 'D', 900: 'CM', 1000: 'M'}
keyValues = [1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1]

def toRomanNumber(n):
    str = ""
    for i in range(len(keyValues)):
        while n >= keyValues[i]:
            str += romanDict2[keyValues[i]]
            n -= keyValues[i]
    return str

d = pd.read_csv('lines.csv')

lines = {}
elements = []
for i in tqdm(range(len(d))):
    name = f"{d['element'][i]} {toRomanNumber(d['sp_num'][i])}"
    if d['Acc'][i] == 'AAA' or d['Acc'][i] == 'AA':
        if name in elements:
            lines[name].append(d['obs_wl_vac(A)'][i])
        else:
            lines[name] = [d['obs_wl_vac(A)'][i]]
            elements.append(name)

for i in lines.keys():
    lines[i] = list(set(lines[i]))

with open('lines.txt', 'w+') as f:
    f.write(str(lines))
