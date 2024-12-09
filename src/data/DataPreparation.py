from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import shutil

def DataPreparation(rescale,shear_range,zoom_range,horizontal_flip,batch_size_Data_Generator):
    train_datagen = ImageDataGenerator(rescale = 1./255,
                                    shear_range = shear_range,
                                    zoom_range = zoom_range,
                                    horizontal_flip = horizontal_flip)

    test_datagen = ImageDataGenerator(rescale = 1./255)

    #  Load the training Set and test set
    training_set = train_datagen.flow_from_directory(os.getcwd()+'/data/original_dataset/training_set',
                                                    target_size = (124, 124),
                                                    batch_size = batch_size_Data_Generator,
                                                    class_mode = 'categorical')

    test_set = test_datagen.flow_from_directory(os.getcwd()+'/data/original_dataset/test_set',
                                                    target_size = (124, 124),
                                                    batch_size = batch_size_Data_Generator,
                                                    class_mode = 'categorical')
    num_classes = training_set.num_classes
    class_names = list(training_set.class_indices.keys())
    
    try:
        shutil.rmtree(os.getcwd()+"/data/augmented_dataset/training_set/")
        shutil.rmtree(os.getcwd()+"/data/augmented_dataset/test_set/")
    except:
        pass
    
    os.makedirs(os.getcwd()+"/data/augmented_dataset/training_set/",exist_ok=True)
    os.makedirs(os.getcwd()+"/data/augmented_dataset/test_set/",exist_ok=True)
    
    training_genrator=train_datagen.flow_from_directory(directory=os.getcwd()+'/data/original_dataset/training_set',save_format="jpeg",save_to_dir=os.getcwd()+"/data/augmented_dataset/training_set")
    test_genrator=train_datagen.flow_from_directory(directory=os.getcwd()+'/data/original_dataset/test_set',save_format="jpeg",save_to_dir=os.getcwd()+"/data/augmented_dataset/test_set")
    
    for _ in range(20):
        next(training_genrator)
        next(test_genrator)
            
    return num_classes , class_names,training_set,test_set
    