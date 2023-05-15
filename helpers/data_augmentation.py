from keras.preprocessing.image import ImageDataGenerator
from skimage import io
import cv2
import numpy as np
import os 

from data_ingestion import DataIngestion


# Construct an instance of the ImageDataGenerator class
# Pass the augmentation parameters through the constructor 
breakHis = DataIngestion(
    directory="BreaKHis_v1/histology_slides/breast/",
    sizes=(244, 244),
    batch=32,
    split_ratio=0.4    
)

train=breakHis.getData(123,"training")

datagen = ImageDataGenerator(
        rotation_range=45,     #Random rotation between 0 and 45
        width_shift_range=0.2,   #% shift
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='constant', cval=125)
"""
        .flow_from_directory(
            directory=  ,
            image_size=(244, 244),
            batch_size=32,
            color_mode='rgb',
            class_mode= 'categorical',
            seed=123,
            subset="training",            
            shuffle=True
        )

"""
print(datagen)

# Loading a sample image  
# C:\Users\Administrador\VisualStudioProjects\breast-histopathology-classification\BreaKHis_v1\histology_slides\breast\benign\SOB\adenosis\SOB_B_A_14-22549AB\40X\SOB_B_A-14-22549AB-40-001.png
path_try = os.path.dirname("BreaKHis_v1/histology_slides/breast/benign/SOB/adenosis/")
path = os.path.join(
    os.getcwd(),
    path_try, "SOB_B_A_14-22549AB", "40X", "SOB_B_A-14-22549AB-40-001.png"
)
# path_try = ('../BreaKHis_v1/histology_slides/breast/benign/SOB/adenosis/SOB_B_A_14-22549AB/40X/SOB_B_A-14-22549AB-40-001.png')
x = cv2.imread(path)  #Array with shape (256, 256, 3)

print(x)

x = x.reshape((1, ) + x.shape)

i = 0
for batch in datagen.flow_(x, batch_size=16,  
                          save_to_dir='../augmented', 
                          save_prefix='aug', 
                          save_format='png',
                          ):
    i += 1
    if i > 20:
        break