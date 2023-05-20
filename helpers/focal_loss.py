import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.losses import categorical_crossentropy
from loguru import logger

class LossMethod:
    def __init__(self):
        logger.info(f"Class Initialized: {self.__dict__}")

    
    # Define the focal loss function
    def focal_loss(self, gamma=2.0, alpha=0.25):

        # def focal_loss_fixed(y_true, y_pred):
        #     # Calculate the cross-entropy loss
        #     cross_entropy = categorical_crossentropy(y_true, y_pred, from_logits=True)

        #     # Calculate the true class probabilities
        #     p_t = tf.math.exp(-cross_entropy)

        #     # Calculate the focal loss
        #     focal_loss = alpha * ((1 - p_t) ** gamma) * cross_entropy

        #     return focal_loss
        
        def focal_loss_fn(y_true, y_pred):
            epsilon = K.epsilon()
            y_true = K.cast(y_true, dtype=tf.float32)  # Cast y_true to float32
            y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)
            cross_entropy = -y_true * K.log(y_pred)
            focal_loss = alpha * K.pow(1 - y_pred, gamma) * cross_entropy
            return K.mean(focal_loss, axis=-1)
            
        return focal_loss_fn

        # return focal_loss_fixed
