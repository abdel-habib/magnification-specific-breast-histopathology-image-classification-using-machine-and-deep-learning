import numpy as np
import cv2
from tensorflow.keras.applications.densenet import DenseNet169, preprocess_input
from tensorflow.keras.models import Model
from loguru import logger
import  os
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
import pandas as pd

class FeatureExtractor:
    def __init__(self):
        self.base_model = DenseNet169(include_top=False, weights='imagenet')
        self.n_features = 15000

        logger.info(f"Class Initialized: {self.__dict__}")

    def extract_features_model(self, image_path):
        img = cv2.imread(image_path)
        img = cv2.resize(img, (224, 224))  # Resize image to match DenseNet input size
        img = np.expand_dims(img, axis=0)  # Add batch dimension
        img = preprocess_input(img)  # Preprocess input according to DenseNet requirements
        features = self.base_model.predict(img)  # Extract features using DenseNet
        return features.flatten()[:self.n_features]  # Flatten the features to a 1D array

    def extract_features(self, magnification, split):
        dataset_path = f'../BreakHis/{magnification}/{split}/'
        features = []
        labels = []

        # Iterate through the dataset folders (assuming each folder represents a different class)
        for class_folder in tqdm(os.listdir(dataset_path)):
            class_path = os.path.join(dataset_path, class_folder)

            # Iterate through the images in each class folder
            for image_file in os.listdir(class_path):
                image_path = os.path.join(class_path, image_file)
                
                # Extract features for the current image
                image_features = self.extract_features_model(image_path)
                
                # Append the features and corresponding label to the lists
                features.append(image_features)
                labels.append(class_folder)  # Use folder name as the label

        # Convert the features and labels lists to numpy arrays
        features = np.array(features)
        labels = np.array(labels)

        label_encoder = LabelEncoder()
        encoded_labels = label_encoder.fit_transform(labels)

        # Create a DataFrame with features and labels
        data = pd.DataFrame(features)
        data['label'] = encoded_labels

        # Save the DataFrame to a CSV file
        output_path = r"../out/features/"

        if not os.path.exists(output_path):
            os.makedirs(output_path)

        data.to_csv(os.path.join(output_path, f'breakhis_features_{split}.csv'), index=False, header=False)

        return features, labels, encoded_labels