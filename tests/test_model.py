import pytest
from mlstart.data import SkData
from mlstart.model import Model

@pytest.mark.parametrize("model_type", ["lr", "rf"])
def test_model(model_type):
    """
    Test the SkData class by loading different datasets.
    
    Parameters:
        dataset_name (str): The name of the dataset to load.
    
    Returns:
        None
    """
    dataset = SkData("iris")
    model   = Model(model_type)

    assert len(dataset.X.shape) == 2, "broken"
    assert len(dataset.X_names) > 1, "not enough features"
    assert len(dataset.y.shape) == 1, "no label"
    assert len(dataset.y_names) > 0, "no label names"
    
    model.fit(data=dataset)
    assert model.predict(dataset.X).shape[0] == dataset.y.shape[0], "wrong prediction shape"
    assert model.score(dataset.X, dataset.y) > 0.95, "Model ain't converging as it should"