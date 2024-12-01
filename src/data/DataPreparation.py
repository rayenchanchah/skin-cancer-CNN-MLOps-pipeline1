from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

def DataPreparation():
    train_datagen = ImageDataGenerator(rescale = 1./255,
                                    shear_range = 0.2,
                                    zoom_range = 0.2,
                                    horizontal_flip = True)

    test_datagen = ImageDataGenerator(rescale = 1./255)

    #  Load the training Set and test set
    training_set = train_datagen.flow_from_directory(os.getcwd()+'/base_melanome_CNN/training_set',
                                                    target_size = (124, 124),
                                                    batch_size = 16,
                                                    class_mode = 'categorical')

    test_set = test_datagen.flow_from_directory(os.getcwd()+'/base_melanome_CNN/test_set',
                                                    target_size = (124, 124),
                                                    batch_size = 16,
                                                    class_mode = 'categorical')
    num_classes = training_set.num_classes
    class_names = list(training_set.class_indices.keys())

    return num_classes , class_names,training_set,test_set
