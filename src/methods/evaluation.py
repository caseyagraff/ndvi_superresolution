"""
Evaluation code for running model on data.
"""

class ModelEvaluator:
    def __init__(self, model):
        self.model = model

    def evaluate(self, data):
        raise NotImplementedError()
