import pytest
from mlstart.data import SkData

@pytest.mark.parametrize("dataset_name", ["iris", "digits", "wine"])
def test_dataset(dataset_name):
    """
    Test the SkData class by loading different datasets.
    
    Parameters:
        dataset_name (str): The name of the dataset to load.
    
    Returns:
        None
    """
    dataset = SkData(dataset_name)
    
    assert len(dataset.X.shape) == 2, "broken"
    assert len(dataset.X_names) > 1, "not enough features"
    assert len(dataset.y.shape) == 1, "no label"
    assert len(dataset.y_names) > 0, "no label names"
    print("wassup")
    return 1.

