# Hide TF logs
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import sys
sys.path.append(r'..\\helpers')

from loguru import logger
from helpers.focal_loss import LossMethod
from helpers.model import DenseNetModel
from helpers.data_split import DataSplit
from helpers.data_ingestion import DataIngestion

class BreaKHisPipeline:
    def __init__(
            self, 
            num_epochs =  25,
            learning_rate = 0.001,
            batch_size = 32,
            data_split_train_ratio = 0.6,
            image_size = (224, 224,3),
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

    def split(self):
        breakHis = DataIngestion(
            directory="BreaKHis_v1/histology_slides/breast/",
            sizes=self.image_size[0:2],
            batch=self.batch_size,
            split_ratio=self.data_split_test_ratio    
        )
        train=breakHis.getData(123,"training")
        test=breakHis.getData(123,"validation")
        
    def fit(self):

        # Get the model
        model_object = DenseNetModel(
            num_classes = self.num_classes,
            input_shape = self.image_size
        )

        model = model_object.model()
        model.summary()

        lm = LossMethod()

        # Compile the model with the focal loss
        model.compile(optimizer='adam', loss=lm.focal_loss(gamma=2.0, alpha=0.25), metrics=['accuracy'])

        # Train the model
        # model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_test, y_test))


        
        


pipeline = BreaKHisPipeline()

pipeline.split()

#pipeline.fit()