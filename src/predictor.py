import pickle
from pathlib import Path

class Predictor:
    """Handles loading a pre-trained model and making predictions."""

    def __init__(self, model_path):
        self.model_path = Path(model_path)
        self.model = self._load_model()

    def _load_model(self):
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        try:
            with open(self.model_path, 'rb') as file:
                return pickle.load(file)
        except pickle.PickleError as e:
            raise pickle.PickleError(f"Failed to load model: {str(e)}")

    def predict(self, input_data):
        return self.model.predict(input_data)[0]
