import json


def save_json(data: dict, save_path) -> str:
    """Util function: save data from dictionary to JSON file. """
    with open(save_path, 'w') as json_file:
        json.dump(data, json_file, indent=4)
    return save_path


def load_json(read_path: str) -> dict:
    """Util function: reads data from JSON file and returns dictionary. """
    with open(read_path) as json_file:
        data = json.load(json_file)
    return data
