import yaml

def load_config():
    with open("../config.yaml", "r") as f:
        return yaml.safe_load(f)

def get_test_name():
    configs = load_config()

    config_name = '_'.join(str(value) for value in configs.values())

    # remove the . from the name
    if '.' in config_name: config_name = config_name.replace('.', '')

    return config_name

if __name__ == "__main__":
    print(get_test_name())
   
