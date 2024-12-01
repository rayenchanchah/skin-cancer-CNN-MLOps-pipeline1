from tensorflow.keras.utils import load_img
import matplotlib.pyplot as plt
import numpy as np

def ModelTesting(model,class_names):
        test_image = load_img('base_melanome_CNN/cas.jpg', target_size = (124,124))
        plt.imshow(test_image, interpolation = 'spline16')
        plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
        plt.show()
        test_image = np.expand_dims(test_image, axis = 0)
        result= model.predict(test_image)
        t=0
        i=0
        for label in class_names:
                print("\t%s ==> %.2f %%" % (label, result[t][i]*100))
                i = i + 1