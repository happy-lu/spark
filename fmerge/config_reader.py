import configparser

DEFAULT_DICT = {'service_port': '9080',
                'log_level': 'INFO',
                'max_content_length': '128 * 1024 * 1024',
                'image_folder': './image',
                 'image_folder': './text'
}


class ConfigReader(object):
    def __init__(self, conf_file):
        self.config = configparser.ConfigParser()
        self.config.read(conf_file)
        self.config['DEFAULT'] = DEFAULT_DICT

    # 获取config配置文件
    def get_config(self, key):
        return self.config["default"][key]

    # 获取config配置文件
    def get_all_config_dict(self):
        return self.config["default"]
