import os
import tensorflow as tf

from loguru import logger

class DataSplit:
    def __init__(self):
      self.dataset = None
      self.left_size = None
      self.rigth_size = None
      self.shuffle=False
      self.seed = None
      logger.info(f"Class initialized: {self.__dict__}")
      
    def get_division(self):
        test_data,training_data = tf.keras.utils.split_dataset(
            dataset=self.dataset,
            left_size= self.left_size,
            shuffle=self.shuffle,
            seed=self.seed
        )
        return test_data,training_data
    
    def set_on_shuffle(self,seed):
        self.shuffle = True
        self.seed = seed
        return None
    
    def set_off_shuffle(self):
        self.shuffle=False
        self.seed = None
        return None
    
    def set_dataset(self,dataset):
        self.dataset = dataset
        return None
    
    def set_splitingratio(self,ratio):
        self.left_size = ratio
        self.rigth_size = 1 - ratio



    



        