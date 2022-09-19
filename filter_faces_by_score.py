import pandas as pd
import os

path_dir = './dataset/aligned/criminal/'

csv_file_names = ['./dataset/aligned/criminal.csv', './dataset/aligned/non-criminal.csv']

for csv_file_name in csv_file_names:
    csv_file = pd.read_csv(csv_file_name)

    csv_file2 = csv_file[csv_file['confidence'] >= 0.9]
    # csv_file3 = pd.concat([csv_file, csv_file2]).drop_duplicates(keep=False).reset_index()
    csv_file3 = csv_file2['name']


    for crim in csv_file3:
        if os.path.exists(path_dir + crim + '.bmp'):
            os.remove(path_dir + crim + '.bmp')


