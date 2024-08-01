import yaml, os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
def load_config(config_path = 'config.yaml'):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def get_test_name():
    configs = load_config()

    print(configs)

    config_name = '_'.join(str(value) for value in configs.values())

    # remove the . from the name
    if '.' in config_name: config_name = config_name.replace('.', '')

    return config_name

if __name__ == "__main__":
    print(get_test_name())
   
