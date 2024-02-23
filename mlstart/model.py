from sklearn.linear_model import LogisticRegression

class Model():
    def __init__(self, 
                 type: str="lr"):
        """
        Initialize the Model class.

        Parameters:
        - type (str): The type of the model. Default is "lr" for Logistic Regression.

        Raises:
        - ValueError: If the model type is unknown.
        """
        self.type = type
        self.model = None
        
        if self.type == "lr":
            self.model = LogisticRegression()
        else:
            raise ValueError("Unknown model type.")
        
    def fit(self, X, y):
        """
        Fit the model to the training data.

        Parameters:
        - X: The input features.
        - y: The target variable.
        """
        self.model.fit(X, y)
        
    def predict(self, X):
        """
        Predict the target variable for the input features.

        Parameters:
        - X: The input features.

        Returns:
        - The predicted target variable.
        """
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """
        Predict the probabilities of the target variable for the input features.

        Parameters:
        - X: The input features.

        Returns:
        - The predicted probabilities of the target variable.
        """
        return self.model.predict_proba(X)
    
    def score(self, X, y):
        """
        Calculate the accuracy score of the model on the given data.

        Parameters:
        - X: The input features.
        - y: The target variable.

        Returns:
        - The accuracy score of the model.
        """
        return self.model.score(X, y)

    def save(self, path: str):
        """
        Save the model to the given path.

        Parameters:
        - path (str): The path to save the model to.
        """
        self.model.save(path)

    def load(self, path: str):
        """
        Load the model from the given path.

        Parameters:
        - path (str): The path to load the model from.
        """
        self.model.load(path)