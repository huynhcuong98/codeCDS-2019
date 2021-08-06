import xml.etree.ElementTree as ET
import glob
import numpy as np 
import csv
import os

root = 'straight'
files = os.listdir(root)
def get_index(n):
    index = ['0','0','0','0','0','0','0','0']
    n = str(n)
    for i in range(len(n)):
        index.pop(0)
        index.append(n[i])
    # print('index', index)
    s=''
    for it in index:
        s += str(it)
    return s
# for it in f:
#     print(it)
#     os.rename(f'{root}/{it}', f'{root}/{it}.jpg')
for i in range(len(files)):
    new_name = get_index(i)

    os.rename(f'{root}/{files[i][:-4]}.png', f'{root}/{root}_{new_name}.jpg')
    # os.rename(f'../labels/{files[i]}', f'../labels/{new_name}.xml')