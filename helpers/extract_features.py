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
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler


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
        return features.flatten()  # Flatten the features to a 1D array

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
        logger.info(f"Starting feature selection on {split} split...")
        data = np.genfromtxt(csv_dir, delimiter=',')

        # Split the data into features (X) and target variable (y)
        X = data[:, :-1]  # All columns except the last one
        y = data[:, -1]   # Last column

        # Standardize the training predictors
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        # Perform Lasso regularization with alpha=0.01
        selected_features = self.perform_lasso_regularization(X, y)

        # Select the features from the data
        selected_data = self.select_features_by_indices(data, selected_features)

        # Perform RFE for further feature selection
        n_features = max_features  # Example: Select max 10 features
        selected_features_rfe = self.perform_rfe(selected_data, y, n_features)

        # Select the features from the data based on RFE selection
        selected_data_rfe = self.select_features_by_indices(selected_data, selected_features_rfe)

        # Convert the selected data to a pandas DataFrame
        selected_df = pd.DataFrame(selected_data_rfe)

        # Join y with selected_df as the last column
        selected_df_with_y = pd.concat([selected_df, pd.DataFrame(y)], axis=1)


        logger.info(f"Finished feature selection on {split} split. New dimensions are {selected_df_with_y.shape}")
        
        # Save the DataFrame to a CSV file
        output_path = r"../out/features_selected/"

        if not os.path.exists(output_path):
            os.makedirs(output_path)

        # Save the selected data to a new CSV file
        selected_df_with_y.to_csv(os.path.join(output_path, f'breakhis_selected_features_{split}.csv'), index=False, header=False)
        
        return selected_features        
    
    def perform_lasso_regularization(self, X, y, alpha=0.01):
        """
        Perform Lasso regularization for feature selection with a given alpha value.

        Args:
            X: The input feature matrix.
            y: The target variable.
            alpha: The regularization parameter (default: 0.01).

        Returns:
            selected_features: A list of selected feature indices.
        """
        lasso = Lasso(alpha=alpha)
        lasso.fit(X, y)

        selected_features = np.nonzero(lasso.coef_)[0]

        return selected_features
    
    def select_features_by_indices(self, data, selected_indices):
        """
        Select features from the data based on the given indices.

        Args:
            data: The input data.
            selected_indices: A list of selected feature indices.

        Returns:
            selected_data: The selected data containing only the selected features.
        """
        selected_data = data[:, selected_indices]

        return selected_data
    
    def perform_rfe(self, selected_data, y, n_features):
        """
        Perform Recursive Feature Elimination (RFE) for further feature selection.

        Args:
            selected_data: The selected data containing only the selected features.
            y: The target variable.
            n_features: The desired number of features to select.

        Returns:
            selected_features: A list of selected feature indices.
        """
        estimator = LinearRegression()
        rfe = RFE(estimator, n_features_to_select=n_features)
        rfe.fit(selected_data, y)

        selected_features = np.nonzero(rfe.support_)[0]

        return selected_features
