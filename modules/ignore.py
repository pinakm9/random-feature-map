import glob
import os
glob.glob('../data/*/batch_data.csv') 

def unignore(pattern):
    paths = glob.glob(pattern, recursive=True)
    file = os.path.basename(pattern)
    for p in paths:
        print('working on ' + f'{os.path.dirname(p)}/.gitignore')
        with open(f'{os.path.dirname(p)}/.gitignore', 'w+') as f:
            f.write('/**\n!batch_data.csv\n!config.json\n')
    
