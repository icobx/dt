import os
import shutil

from definitions import *

exp_path = EXP_DIR_PATH
current_path = os.getcwd()
print(os.listdir(exp_path))

for model in os.listdir(exp_path):
    path = os.path.join(exp_path, model, 'training')
    
    for d in os.listdir(path):
        op = os.path.join(path, d)
        np = os.path.join(current_path, f'{model}-training', d)

        files_tcp = [f for f in os.listdir(op) if 'metrics.pt' in f]
        if not len(files_tcp):
            continue

        os.mkdir(np)
        for f in files_tcp:
            shutil.copy(os.path.join(op, f), np)
            
        print(os.listdir(np))