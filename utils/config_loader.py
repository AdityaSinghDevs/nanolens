import yaml
from pathlib import Path 


def load_configs(filename:str):

    PROJECT_ROOT = Path(__file__).resolve().parent.parent

    if not filename:
        raise ValueError("configs missing or file not specified")
    
    config_path = PROJECT_ROOT / "config" / f"{filename}.yaml"

    with open (config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    return cfg 
