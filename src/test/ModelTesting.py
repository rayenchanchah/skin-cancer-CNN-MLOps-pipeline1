from tensorflow.keras.utils import load_img
import matplotlib.pyplot as plt
import numpy as np
import os
from random import randint
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def ModelTesting(model,class_names):
        
        test_loader = ImageDataGenerator()
        
        test_images = test_loader.flow_from_directory(directory=os.getcwd()+'/data/original_dataset/real_test_set',target_size = (124,124))
        
        image_names=os.listdir(os.getcwd()+'/data/original_dataset/real_test_set/set')
        
        test_images=next(test_images)
                
        plt.figure(figsize=(10, 10))

        for i in range(len(test_images[0])):
                
                plt.subplot(4, 4, i+1) 
                plt.imshow(test_images[0][i].astype('uint8'))
                plt.axis("off")
                
                test_image = np.expand_dims(test_images[0][i].astype('uint8'), axis = 0)
                result= model.predict(test_image)
                t=0
                j=0
                
                for label in class_names:
                        print("\t%s ==> %.2f %%" % (label, result[t][j]*100))
                        j = j + 1

                plt.title(
                        label=image_names[i]+" pred : "+class_names[np.where(result[0]==np.max(result[0]))[0][0]]
                )
                
        plt.show()
        
        # test_image = load_img(os.getcwd()+'/data/original_dataset/cas.jpg', target_size = (124,124))
        # plt.imshow(test_image, interpolation = 'spline16')
        # plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
        # plt.show()
        
        