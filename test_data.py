import pytest
from mlstart.data import SkData

@pytest.mark.parametrize("dataset_name", ["iris", "diabetes", "digits"])
def dataset_test(dataset_name):
    """
    Test the SkData class by loading different datasets.
    
    Parameters:
        dataset_name (str): The name of the dataset to load.
    
    Returns:
        None
    """
    dataset = SkData(dataset_name)
    
    assert len(dataset.X.shape) == 2
    assert len(dataset.X_names) > 1
    assert len(dataset.y.shape) == 1
    assert len(dataset.y_names) > 0

    return 1.

