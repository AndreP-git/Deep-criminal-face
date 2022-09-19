import pandas as pd
import os
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

df = pd.DataFrame(columns=['name', 'face', 'confidence'])
dir_path = "./dataset/aligned/criminal/"

for file_name in os.listdir(dir_path):
    if not file_name.endswith(".csv"):
        continue

    with open(dir_path + file_name) as f:
        lines = f.readlines()[1:]
        for line in lines:
            line = [el.strip() for el in line.split(sep=', ')]
            to_add = [file_name.split(sep='.')[0]]
            to_add.extend(line)
            dict_entry = {'name': to_add[0], 'face': to_add[1], 'confidence': to_add[2]}
            df = df.append(dict_entry, ignore_index=True)


df.to_csv('./dataset/aligned/criminal.csv')
print(df.head())
print(len(df))

