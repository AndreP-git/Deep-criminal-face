import os
import shutil

dir_path = "./dataset/aligned/criminal/"
for dir in os.listdir(dir_path):
    if not os.path.isdir(dir_path + dir):
        continue

    for file in os.listdir(dir_path + dir):
        shutil.copyfile(dir_path + dir + '/' + file, './dataset/aligned/prova/' + dir.replace("_aligned", "") + '.bmp')
        
