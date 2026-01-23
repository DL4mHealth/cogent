import importlib

def get_dataset(dataset_name: str):
    """
    Dynamically import and return the dataset
    """
    module_name = f"data_wrappers.dataset_{dataset_name}"
    try:
        dataset_module = importlib.import_module(module_name)
        return dataset_module
    except ModuleNotFoundError:
        raise ValueError(f"Dataset processor '{module_name}' not found.")

