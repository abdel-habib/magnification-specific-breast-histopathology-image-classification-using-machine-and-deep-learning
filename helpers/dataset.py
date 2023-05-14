import os
import pandas as pd

class BreasKHisDataset():
    def __init__(self):
        self.benign_dir = os.path.join(
            os.getcwd(), os.path.dirname("../BreaKHis_v1/histology_slides/breast/benign/"))   
        self.malignant_dir = os.path.join(
            os.getcwd(), os.path.dirname("../BreaKHis_v1/histology_slides/breast/malignant/")   
        )
        
    def get_classes(self):
        """
        Returns a dictionary with keys 'benign' and 'malignant',
        and corresponding values as lists of subdirectory names in
        the 'SOB' subdirectories of the benign_dir and malignant_dir,
        respectively.
        """
        benign_path = os.path.join(self.benign_dir, 'SOB')
        malignant_path = os.path.join(self.malignant_dir, 'SOB')
        
        classes = {
            'benign': [entry.name for entry in os.scandir(benign_path) if entry.is_dir()],
            'malignant': [entry.name for entry in os.scandir(malignant_path) if entry.is_dir()]
        }
        
        return classes

    
    def count_images(self, *dirs):
        """
        Counts the number of images found in all directories passed as arguments.
        Args:
            *dirs: Any number of directory paths to search for images in.
        Returns:
            The total count of images found.
        """
        image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp']
        count = 0
        for directory in dirs:
            for root, dirs, files in os.walk(directory):
                for file in files:
                    if any(file.endswith(ext) for ext in image_extensions):
                        count += 1
        return count
    
    def generate_dataset_dataframe_report(self):
        """
        Generates a Pandas dataframe that contains information about the dataset,
        including class names, directories, and number of images in each class directory.

        Returns:
            A Pandas dataframe with three columns: "class_name", "class_dir", and "class_files_count".
        """
        data = list()

        # Get the names of all classes and their respective directories
        classes = self.get_classes()

        # Loop over each class and its associated directories
        for class_type, class_names in classes.items():
            for class_name in class_names:
                # Create the full directory path for the current class and type
                class_dir = os.path.join(self.benign_dir if class_type == 'benign' else self.malignant_dir, 'SOB', class_name)

                # Count the number of images in the class directory
                num_images = self.count_images(class_dir)

                # Append the class name, directory, and image count to the data list
                data.append([class_name, class_dir, num_images])

        # Create a dataframe from the data list with column names
        df = pd.DataFrame(data, columns=['class_name', 'class_dir', 'class_files_count'])

        return df



# Uncomment the code below for testing the class methods
dataset = BreasKHisDataset()
classes = dataset.get_classes()
#count = dataset.count_images(dataset.benign_dir, dataset.malignant_dir)
#df = dataset.generate_dataset_dataframe_report()
#print(count)
print(classes)
#print(df)

