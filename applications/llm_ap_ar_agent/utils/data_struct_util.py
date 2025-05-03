# data_struct_util.py

def get_nested(data: dict, path: list, default=None):
    """
    Safely retrieve a nested value from a dictionary.

    Args:
        data (dict): The dictionary to retrieve from.
        path (list): List of keys representing the path, e.g. ['invoice', 'invoice_id']
        default (Any): Default value if the path is not found.

    Returns:
        The retrieved value or the default if not found.
    """
    current = data
    for key in path:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return default
    return current
