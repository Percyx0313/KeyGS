import yaml
import os


yaml_file = 'outputs/TANK_KEY5_BEST/key_gaussian/Ballroom/config.yml'
import yaml

# 讀取 YAML 文件
with open(yaml_file, 'r', encoding='utf-8') as file:
    data = yaml.safe_load(file)

print(data)