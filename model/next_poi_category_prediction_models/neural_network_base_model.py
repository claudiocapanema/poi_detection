from abc import ABC, abstractmethod

class NNBase(ABC):

    def __init__(self, model_name):
        self.model_name = model_name

    @abstractmethod
    def build(self):
        print(self.model_name)