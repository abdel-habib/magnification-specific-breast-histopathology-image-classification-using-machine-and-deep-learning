import os
import tensorflow as tf
import tensorflow.keras as K

from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau

from loguru import logger

class DenseNetModel:
    def __init__(self, 
                 num_classes, 
                 input_shape
                 ):
        
        self.num_classes = num_classes
        self.input_shape = input_shape

        logger.info(f"Class Initialized: {self.__dict__}")
    
    def model(self):
        densenet_model = tf.keras.applications.DenseNet121(
            include_top=False,
            weights="imagenet",
            input_tensor=None,
            input_shape= self.input_shape,
            pooling=None,  # Remove global average pooling
        )
        
        # Freeze the pre-trained layers
        densenet_model.trainable = False

        # Add custom layers for breast cancer classification
        x = densenet_model.output # output layer is [ relu (Activation)  (None, 7, 7, 1024)   0   ['bn[0][0]'] ]
        
        # Add four dense blocks
        num_dense_blocks = 4
        for _ in range(num_dense_blocks):
            # Dense block
            x = self.dense_block(x)
            
            # Transition layer
            x = self.transition_layer(x)
        
        # Add a global average pooling layer
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        
        # Add a fully connected layer with the desired number of classes for BC histopathological images
        predictions = tf.keras.layers.Dense(self.num_classes, activation="softmax")(x)
        
        # Create the model
        model = tf.keras.Model(inputs=densenet_model.input, outputs=predictions)
        
        # Compile the model
        model.compile(metrics=["accuracy"])

        return model
    
    def dense_block(self, x):
        # Add convolutional layers with a reduced kernel size
        x1 = tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)
        x1 = tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x1)
        x1 = tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x1)
        x1 = tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x1)
        
        # Concatenate the input with the output of the dense block
        x = tf.keras.layers.Concatenate()([x, x1])
        
        return x

    def transition_layer(self, x):
        # Reduce the size of the feature map
        x = tf.keras.layers.Conv2D(64, (1, 1), activation="relu")(x)
        x = tf.keras.layers.AveragePooling2D((2, 2), strides=(2, 2), padding="same")(x)
        
        return x

