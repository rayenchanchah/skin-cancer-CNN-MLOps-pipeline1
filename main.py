import argparse
import os
import yaml

from src.data.DataPreparation import DataPreparation
from src.model.ModelTraining import Train
from src.model.Model import Model_init
from src.model.ModelValidation import ModelValidation
from src.test.ModelTesting import ModelTesting

def main():

    parser = argparse.ArgumentParser(description="")
    

    with open("config.yaml") as f:
        config=yaml.safe_load(f)
        print(config)
        
    num_classes,class_names,training_set,test_set=DataPreparation(config["rescale"],config["shear_range"],config["zoom_range"],config["horizontal_flip"],config["batch_size_Data_Generator"])
    
    model=Model_init(num_classes)
    
    hist=Train(model,training_set,test_set,config["batch_size"],config["optimizer"],config["epochs"],config["lr"])
    
    ModelValidation(hist,model,test_set,training_set)
    
    ModelTesting(model,class_names)
    
    model.save(os.getcwd()+"/models/model_first_try.h5")

if __name__ == "__main__":
    main()