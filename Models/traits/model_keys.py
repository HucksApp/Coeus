from abc import abstractmethod
import os
import json


class CoeusModelKeys:
    def __init__(self, save_dir, settings_file, training=False, keys=[]):
        model_keys = self.get_setting("keys") or []
        self.save_dir = save_dir
        self.settings_file = settings_file
        if training:
            if len(model_keys) == 0 and len(keys) == 0:
                raise ValueError(
                    "Missing metas: 'keys' are required to index model optimized object"
                )
        else:
            if len(model_keys) == 0:
                raise ValueError(
                    "Missing metas: 'keys' are required to index model optimized object"
                )

        self.keys = list(set([*model_keys, *keys]))
        self._validate_keys(model_keys)
        if len(keys) > 0:
            self.update_settings_file('keys', self.keys)

    def update_settings_file(self, key, value):
        file_path = os.path.join(
            self.save_dir, self.settings_file)
        if os.path.exists(file_path):
            with open(file_path, "r") as file:
                data = json.load(file)
        else:
            data = {}
        data[key] = value
        with open(file_path, "w") as file:
            json.dump(data, file, indent=4)

    def get_setting(self, key):
        file_path = os.path.join(
            self.save_dir, self.settings_file)
        if os.path.exists(file_path):
            with open(file_path, "r") as file:
                data = json.load(file)
            return data.get(key)
        return None

    def _validate_keys(self, model_keys):
        if self.keys and model_keys:
            match_percentage = self._calculate_match_percentage(
                self.keys, model_keys)
            if match_percentage < 70:
                raise ValueError(
                    f"Key mismatch: At least 70% of the words in provided keys must match model keys. Current match: {
                        match_percentage}%"
                )

    def _calculate_match_percentage(self, keys1, keys2):
        words_in_keys1 = set(keys1)
        words_in_keys2 = set(keys2)
        match_count = len(words_in_keys1.intersection(words_in_keys2))
        total_words = len(words_in_keys1)
        return (match_count / total_words) * 100 if total_words > 0 else 0

    def manage_keys(self, keys_to_manage, rm=False):
        model_keys = self.get_setting("keys") or []
        if not rm:
            self.keys = list(set(self.keys + keys_to_manage))
        else:
            # Remove keys if rm is True
            self.keys = list(set(self.keys) - set(keys_to_manage))
        self.keys = list(set(self.keys + model_keys))
        self.update_settings_file('keys', self.keys)
