import hashlib
import os
import toml


def file_hash(filename):
    """Calculate the SHA256 hash of a file."""
    h = hashlib.sha256()
    b = bytearray(10 * 1024 * 1024)  # read 10 megabytes at a time
    with open(filename, "rb", buffering=0) as f:
        while n := f.readinto(b):
            h.update(b)
    return h.hexdigest()


def string_hash(s):
    """Calculate the SHA256 hash of a string."""
    h = hashlib.sha256()
    h.update(s.encode("utf-8"))
    return h.hexdigest()


def get_toml_config(key=None, toml_file="config.toml"):
    toml_config_path = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            toml_file,
        )
    )

    toml_data = {}
    with open(toml_config_path, "r") as f:
        toml_data = toml.load(f)

    if key and key in toml_data:
        return toml_data[key]
    return toml_data


def update_toml_config(toml_dict, toml_file="config.toml"):
    toml_config_path = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            toml_file,
        )
    )

    with open(toml_config_path, "wb") as f:
        toml_content = toml.dumps(toml_dict)
        f.write(toml_content.encode())
