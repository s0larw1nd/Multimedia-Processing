import yaml
import os

path_to_classes_txt = './dataset/classes.txt'
path_to_data_yaml = './data.yaml'

with open(path_to_classes_txt, 'r') as f:
    classes = []
    for line in f.readlines():
        if len(line.strip()) == 0: continue
        classes.append(line.strip())
number_of_classes = len(classes)

data = {
    'path': '/content/data',
    'train': 'train/images',
    'val': 'validation/images',
    'nc': number_of_classes,
    'names': classes
}

with open(path_to_data_yaml, 'w') as f:
    yaml.dump(data, f, sort_keys=False)