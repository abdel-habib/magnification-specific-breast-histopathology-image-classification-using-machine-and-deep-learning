import numpy as np
import cv2
from tensorflow.keras.applications.densenet import DenseNet169, preprocess_input
from tensorflow.keras.models import Model
from loguru import logger
import  os
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np

from sklearn.feature_selection import SelectKBest, RFE
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score


class FeatureExtractor:
    def __init__(self):
        self.base_model = DenseNet169(include_top=False, weights='imagenet')
        self.n_features = 1500

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
    
    def feature_selection(self, csv_dir, max_features, split):
        # Read the CSV file into a pandas DataFrame
        data = pd.read_csv(csv_dir, header=None)
        
        # Separate the features and target variable
        X = data.iloc[:, :-1]
        y = data.iloc[:, -1]
        
        # Calculate correlation coefficients
        corr_matrix = np.abs(X.corrwith(y))
        
        # Sort features based on correlation
        sorted_features = corr_matrix.sort_values(ascending=False)
        
        selected_features = []
        best_score = -np.inf
        
        for n_features in range(1, min(max_features, len(X.columns))+1):
            logger.info("Loop iteration...")
            # Select the top n features based on correlation
            top_features = sorted_features[:n_features].index.tolist()
            X_corr_selected = X[top_features]
            
            # Recursive Feature Elimination (RFE)
            rfe_selector = RFE(estimator=LinearRegression(), n_features_to_select=n_features)
            X_rfe_selected = rfe_selector.fit_transform(X, y)
            
            # Check if RFE score is better than correlation score
            if rfe_selector.score(X_rfe_selected, y) > best_score:
                selected_features = X.columns[rfe_selector.get_support()].tolist()
                best_score = rfe_selector.score(X_rfe_selected, y)
        
        # Create a new DataFrame with the selected features and target
        selected_data = data[selected_features + [data.columns[-1]]]
        
        # Save the DataFrame to a CSV file
        output_path = r"../out/features_selected/"

        if not os.path.exists(output_path):
            os.makedirs(output_path)

        # Save the selected data to a new CSV file
        selected_data.to_csv(os.path.join(output_path, f'breakhis_features_{split}.csv'), index=False, header=False)
        
        return selected_features
