import yaml


def load_config(config_path):
    """Helper function to load config file

    Args:
        config_path (_type_): _description_

    Returns:
        _type_: _description_
    """
    with open(config_path, 'r') as config_file:
        return yaml.safe_load(config_file)