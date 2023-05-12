# Hide TF logs
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import sys
sys.path.append(r'..\\helpers')

from loguru import logger

class BreaKHisPipeline:
    def __init__(
            self, 
            n_epochs =  25,
            learning_rate = 0.001,
            batch_size = 8,
            data_split_train_ratio = 0.6,
            image_size = (224, 224, 3)
            ):
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.data_split_train_ratio = data_split_train_ratio
        self.data_split_test_ratio = round(1 - data_split_train_ratio, 2)
        self.image_size = image_size

        
        logger.info(f"Class Initialized: {self.__dict__}")

    def fit():
        pass


pipeline = BreaKHisPipeline()