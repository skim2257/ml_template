from sklearn import datasets

class SkData():
    def __init__(self, dataset_name="iris"):
        self.X, self.X_names, self.y, self.y_names = self.parse_dataset(dataset_name)
        
    @staticmethod
    def parse_dataset(name):
        if name == "iris":
            ds = datasets.load_iris()
        elif name == "diabetes":
            ds = datasets.load_diabetes()
        elif name == "digits":
            ds = datasets.load_digits()
        
        return ds.data, ds.feature_names, ds.target, ds.target_names