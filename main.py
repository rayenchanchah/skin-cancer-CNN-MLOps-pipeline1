import argparse

from src.data.DataPreparation import DataPreparation
import matplotlib.pyplot as plt
from src.model.ModelTraining import Train
from src.model.Model import Model_init
from src.model.ModelValidation import ModelValidation

from src.test.ModelTesting import ModelTesting

import os

def main():
    num_classes,class_names,training_set,test_set=DataPreparation()
    model=Model_init(num_classes)
    hist=Train(model,training_set,test_set)
    ModelValidation(hist,model,test_set,training_set)
    ModelTesting(model,class_names)
    model.save(os.getcwd()+"models/model_first_try.h5")

if __name__ == "__main__":
    main()