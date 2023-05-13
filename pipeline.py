# Hide TF logs
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import sys
sys.path.append(r'..\\helpers')

from loguru import logger
from helpers.model import DenseNetModel

class BreaKHisPipeline:
    def __init__(
            self, 
            num_epochs =  25,
            learning_rate = 0.001,
            batch_size = 8,
            data_split_train_ratio = 0.6,
            image_size = (224, 224, 3),
            num_classes=8
            ):
        
        self.n_epochs               = num_epochs
        self.num_classes            = num_classes 
        self.learning_rate          = learning_rate
        self.batch_size             = batch_size
        self.data_split_train_ratio = data_split_train_ratio
        self.data_split_test_ratio  = round(1 - data_split_train_ratio, 2)
        self.image_size             = image_size

        
        logger.info(f"Class Initialized: {self.__dict__}")

    def fit(self):

        # Get the model
        model_object = DenseNetModel(
            num_classes = self.num_classes,
            input_shape = self.image_size
        )

        model = model_object.model()
        model.summary()


pipeline = BreaKHisPipeline()

pipeline.fit()