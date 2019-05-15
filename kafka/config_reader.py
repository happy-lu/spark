import configparser


class ConfigReader(object):
    def __init__(self, conf_file):
        self.config = configparser.ConfigParser()
        self.config.read(conf_file)

    def get_all_confs(self):
        return self.config

    def get_conf_str(self, name, category="default"):
        return remove_quotes(self.config[category][name])


def remove_quotes(value):
    value = value.strip()
    if value.startswith('"') or value.startswith("'"):
        return value[1:-1]
    else:
        return value
