# Hide TF logs
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import sys
sys.path.append(r'..\\helpers')

import matplotlib.pyplot as plt
import tensorflow as tf
from loguru import logger
from helpers.focal_loss import LossMethod
from helpers.model import DenseNetModel
from helpers.data_ingestion import DataIngestion
from helpers.data_augmentation import DataAugmentation
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

import numpy as np

class BreaKHisPipeline:
    def __init__(
            self, 
            num_epochs =  80,
            learning_rate = 0.003,
            batch_size = 32,
            data_split_train_ratio = 0.6,
            image_size = (224, 224,3),
            num_classes=8,
            magnification = '400X'

            ):
        
        self.n_epochs               = num_epochs
        self.num_classes            = num_classes 
        self.learning_rate          = learning_rate
        self.batch_size             = batch_size
        self.data_split_train_ratio = data_split_train_ratio
        self.data_split_test_ratio  = round(1 - data_split_train_ratio, 2)
        self.image_size             = image_size
        self.magnification          = magnification

        
        logger.info(f"Class Initialized: {self.__dict__}")
        
    def split(self):        
        
        breakHis_train = DataIngestion(
            directory="BreakHis/"+self.magnification+"/train/",
            sizes=self.image_size[0:2],
            batch=self.batch_size   
        )
        train=breakHis_train.getData()
        
        breakHis_test = DataIngestion(
            directory="BreakHis/"+self.magnification+"/test/",
            sizes=self.image_size[0:2],
            batch=self.batch_size   
        )
        
        test = breakHis_test.getData()
        
        breakHis_validation = DataIngestion(
            directory="BreakHis/"+self.magnification+"/validation/",
            sizes=self.image_size[0:2],
            batch=self.batch_size   
        )
        validation = breakHis_validation.getData()

        return train, test, validation
        
        
    def fit(self):

        # Get the model
        model_object = DenseNetModel(
            num_classes = self.num_classes,
            input_shape = self.image_size
        )

        model = model_object.model()
        model.summary()

        callbacks = model_object.callbacks()

        lm = LossMethod()

        # We replace keras dataset with augmentation generator
        # train, _, validation = self.split()

        train_gen, valid_gen, test_gen = DataAugmentation(
            target_size = self.image_size[0:2],
            train_path=f'BreakHis/{self.magnification}/train/',
            valid_path=f'BreakHis/{self.magnification}/validation/',
            test_path=f'BreakHis/{self.magnification}/test/').PerformAugmentation()

        # Defining optimizer
        opt = tf.keras.optimizers.Adam(
                learning_rate=self.learning_rate)

        # Compile the model with the focal loss
        model.compile(optimizer=opt, loss=lm.focal_loss(gamma=2.0, alpha=0.25), metrics=['accuracy'])

        # Train the model
        valX, valY = valid_gen.next()

        model.fit(train_gen,
                  epochs=self.n_epochs, 
                  batch_size=self.batch_size, 
                  validation_data=(valX, valY),
                  callbacks=callbacks)
        
        # save model without optimizer, ready for prod 
        logger.info('Finished Training. Saving Model.')
        output_path = "out/model/"

        if not os.path.exists(output_path):
            os.makedirs(output_path)

        tf.keras.models.save_model(
            model, f"out/model/model.BreakHis.h5", include_optimizer=False, save_format='h5'
        )

    def evaluate_and_plot_roc(self):
        # Load the saved model
        model = tf.keras.models.load_model("out/model/model.BreakHis.h5")

        # Prepare the test data generator from DataAugmentation
        _, _, test_gen = DataAugmentation(
            target_size=self.image_size[0:2],
            train_path=f'BreakHis/{self.magnification}/train/',
            valid_path=f'BreakHis/{self.magnification}/validation/',
            test_path=f'BreakHis/{self.magnification}/test/'
        ).PerformAugmentation()

        # Compile the model
        opt = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        lm = LossMethod()
        model.compile(optimizer=opt, loss=lm.focal_loss(gamma=2.0, alpha=0.25), metrics=['accuracy'])

        # Evaluate the model
        logger.info('Evaluating the model.')
        test_data = []
        test_labels = []
        
        for i in range(len(test_gen)):
            data, labels = test_gen[i]
            test_data.append(data)
            test_labels.append(labels)
            
        test_data = np.vstack(test_data)
        test_labels = np.vstack(test_labels)
        test_loss, test_acc = model.evaluate(test_data, test_labels, verbose=0)
        logger.info(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}')
        
        # Plot ROC curve
        logger.info('Plotting the ROC curve.')
        test_predictions = model.predict(test_data)
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        
        for i in range(self.num_classes):
            fpr[i], tpr[i], _ = roc_curve(test_labels[:, i], test_predictions[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        
        plt.figure()
        lw = 2
        
        for i in range(self.num_classes):
            plt.plot(fpr[i], tpr[i], lw=lw, label=f'Class {i} (AUC = {roc_auc[i]:.2f})')
        
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.show()


        
        


pipeline = BreaKHisPipeline()

# pipeline.fit()