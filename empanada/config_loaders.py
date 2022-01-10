import os
import yaml

__all__ = [
    'load_config',
    'load_train_config',
    'load_inference_config'
]

def load_config(url):
    with open(url, mode='r') as handle:
        config = yaml.load(handle, Loader=yaml.FullLoader)

    return config

def load_train_config(config_file):
    config = load_config(config_file)
    base_configs = [config]

    # inherit recursively
    has_base = 'BASE' in config
    while has_base:
        # get the location of the BASE file
        config_file_dir = os.path.dirname(config_file)
        base_path = config['BASE']

        base_config_path = os.path.join(os.path.abspath(config_file_dir), base_path)
        base_config_file = os.path.relpath(base_config_path)
        base_config = load_config(base_config_file)

        has_base = 'BASE' in base_config
        base_configs.append(base_config)
        config_file = base_config_file
        config = base_config


    # reverse the order of the configs so that the root
    # config file is first in the list
    base_configs = base_configs[::-1]

    # update keys with children overwriting parents
    inherited_config = base_configs[0]

    for config in base_configs[1:]:
        # we only go 2 keys deep
        # keys at first level are:
        check_keys = ['DATASET', 'MODEL', 'TRAIN', 'EVAL']

        for ckey in check_keys:
            config_value = config.get(ckey)
            if config_value != inherited_config[ckey] and config_value is not None:
                # then assign any parameters to inherited config
                for pname, pvalue in config_value.items():
                    inherited_config[ckey][pname] = pvalue

    return inherited_config

def load_inference_config(config_file):
    config = load_config(config_file)
    base_configs = [config]

    # inherit recursively
    has_base = 'BASE' in config
    while has_base:
        # get the location of the BASE file
        config_file_dir = os.path.dirname(config_file)
        base_path = config['BASE']

        base_config_path = os.path.join(os.path.abspath(config_file_dir), base_path)
        base_config_file = os.path.relpath(base_config_path)
        base_config = load_config(base_config_file)

        has_base = 'BASE' in base_config
        base_configs.append(base_config)

    # reverse the order of the configs so that the root
    # config file is first in the list
    base_configs = base_configs[::-1]

    # update keys with children overwriting parents
    inherited_config = base_configs[0]

    for config in base_configs[1:]:
        # we only go 2 keys deep
        # keys at first level are:
        check_keys = ['engine_params', 'matcher_params']

        for ckey in check_keys:
            config_value = config.get(ckey)
            if config_value != inherited_config[ckey] and config_value is not None:
                # then assign any parameters to inherited config
                for pname, pvalue in config_value.items():
                    inherited_config[ckey][pname] = pvalue

        # now the 1 level deep parameters
        check_keys = ['axes', 'labels', 'filters']
        for ckey in check_keys:
            config_value = config.get(ckey)
            if config_value != inherited_config[ckey] and config_value is not None:
                inherited_config[ckey] = config_value

    return inherited_config
