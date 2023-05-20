import os
import tensorflow as tf

from loguru import logger

class DataIngestion:
    def __init__(self,
                 directory,
                 sizes,
                 batch):
        self.directory=directory
        self.target_sizes=sizes
        self.batch_size=batch
        logger.info(f"Class Initialized: {self.__dict__}")
    
    def getData(self):
        data=tf.keras.utils.image_dataset_from_directory(
                directory=self.directory,
                image_size=self.target_sizes,
                batch_size=self.batch_size
            )
        
        return data
    
    def get_Data_Structure(self,Magnifications):

        
        return train, test, validation
        
        