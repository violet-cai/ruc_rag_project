import os
import re
from typing import Type

import numpy as np
import torch
import yaml
from yaml import FullLoader


class Config:
    # 设置配置文件，支持传入自定义config_file或者config_dict
    def __init__(self, external_config_file=None, external_config_dict=None):
        if external_config_dict is None:
            external_config_dict = {}
        self.yaml_loader = self._build_yaml_loader()
        # 获取自定义的config
        if external_config_file is not None:
            self.file_config = self._load_file_config(external_config_file)
            self.variable_config = {}
        else:
            self.variable_config = external_config_dict
            self.file_config = {}
        self.internal_config = self._get_internal_config()
        self.external_config = self._get_external_config()
        # 结合内部配置和外部配置
        self.final_config = self._get_final_config()
        self._init_device()
        self._set_seed()

    def __getitem__(self, item):
        return self.final_config.get(item)

    def __repr__(self):
        return self.final_config.__str__()

    def __contains__(self, key):
        if not isinstance(key, str):
            raise TypeError("index must be a str.")
        return key in self.final_config

    def __setitem__(self, key, value):
        if not isinstance(key, str):
            raise TypeError("index must be a str.")
        self.final_config[key] = value

    def _load_file_config(self, config_file) -> dict:
        file_config = dict()
        if config_file:
            with open(config_file, "r", encoding="utf-8") as f:
                file_config.update(yaml.load(f.read(), Loader=self.yaml_loader))
        return file_config

    def _build_yaml_loader(self) -> Type[FullLoader]:
        loader = yaml.FullLoader
        loader.add_implicit_resolver(
            "tag:yaml.org,2002:float",
            re.compile(
                r"""^(?:
                 [-+]?[0-9][0-9_]*\.[0-9_]*(?:[eE][-+]?[0-9]+)?
                |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
                |\.[0-9_]+(?:[eE][-+][0-9]+)?
                |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\.[0-9_]*
                |[-+]?\.(?:inf|Inf|INF)
                |\.(?:nan|NaN|NAN)
                |0x[0-9a-fA-F_]+\.[0-9a-fA-F_]*(p[+-]?[0-9]+)?)$""",
                re.X,
            ),
            list("-+0123456789."),
        )
        return loader

    def _update_dict(self, old_dict, new_dict) -> dict:
        same_keys = []
        for key, value in new_dict.items():
            if key in old_dict and isinstance(value, dict):
                same_keys.append(key)
        for key in same_keys:
            old_item = old_dict[key]
            new_item = new_dict[key]
            old_item.update(new_item)
            new_dict[key] = old_item
        old_dict.update(new_dict)
        return old_dict

    def _get_internal_config(self):
        current_path = os.path.dirname(os.path.realpath(__file__))
        internal_config = os.path.join(current_path, "basic_config.yaml")
        basic_config = self._load_file_config(internal_config)
        prompts_config_path = os.path.join(current_path, "prompts_config.yaml")
        prompts_config = self._load_file_config(prompts_config_path)
        return self._update_dict(basic_config, prompts_config)

    def _get_external_config(self):
        external_config = dict()
        external_config = self._update_dict(external_config, self.file_config)
        external_config = self._update_dict(external_config, self.variable_config)
        return external_config

    def _get_final_config(self):
        final_config = self._update_dict(self.internal_config, self.external_config)
        return final_config

    def _init_device(self):
        gpu_id = self.final_config["gpu_id"]
        if gpu_id is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            self.final_config["device"] = torch.device("cuda")
        else:
            self.final_config["device"] = torch.device("cpu")

    def _set_seed(self):
        seed_value = self.final_config["random_seed"]
        np.random.seed(seed_value)
        torch.manual_seed(seed_value)
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
