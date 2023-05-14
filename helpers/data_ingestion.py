import os
import tensorflow as tf

from loguru import logger

class DataIngestion:
    def __init__(self,
                 directory,
                 sizes,
                 batch,
                 split_ratio):
        self.directory=directory
        self.target_sizes=sizes
        self.batch_size=batch
        self.validation_split=split_ratio
        logger.info(f"Class Initialized: {self.__dict__}")
    
    def getData(self,
                seed,
                subset):
        data=tf.keras.utils.image_dataset_from_directory(
                directory=self.directory,
                validation_split=self.validation_split,
                image_size=self.target_sizes,
                batch_size=self.batch_size,
                seed=seed,
                subset=subset
            )
        return data
        