import pickle


def save_valid_template(file, valid_template):
    with open(file, 'wb') as f:
        pickle.dump(valid_template, f)


def load_valid_template(file):
    valid_template = None
    try:
        with open(file, 'rb') as f:
            valid_template = pickle.load(f)
    except Exception:
        pass
    return valid_template


def generate_filenames(filename_base, valid_template):
    return [filename_base[: -4] + '_' + t + '.csv' for t in valid_template]
