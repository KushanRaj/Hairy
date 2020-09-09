import yaml
import os



def read_yaml(config_path):
    if not os.path.exists(config_path):
        raise ValueError(f"{config_path} does not exist.")
    config = yaml.safe_load(open(config_path, "r"))
    
    return config

