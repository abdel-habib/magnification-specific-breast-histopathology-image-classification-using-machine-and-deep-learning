from keras.preprocessing.image import ImageDataGenerator

class DataAugmentation:
    def __init__(self, target_size, train_path, valid_path, test_path):
        self.target_size = target_size
        self.train_path = train_path
        self.valid_path = valid_path
        self.test_path = test_path
        self.random_seed = 123
        
    def PerformAugmentation(self):
        train_datagen = ImageDataGenerator(
            rescale=1. / 255.0,             
            rotation_range=45,              
            width_shift_range=0.2,          
            height_shift_range=0.2,         
            shear_range=0.2,
            zoom_range=0.2,                 
            horizontal_flip=True,           
            brightness_range=(0.9, 1.1),    
            fill_mode='nearest')

        train_gen = train_datagen.flow_from_directory(
            self.train_path,
            target_size=self.target_size,
            color_mode='rgb',
            class_mode= 'categorical',
            batch_size=32,
            shuffle=True,
            seed=self.random_seed)
        
        valid_test_datagen = ImageDataGenerator(
            rescale=1. / 255.0             
            )

        valid_gen = valid_test_datagen.flow_from_directory(
            self.valid_path,
            target_size=self.target_size,
            color_mode='rgb',
            class_mode= 'categorical',
            batch_size=64, #32
            shuffle=True,
            seed=self.random_seed)

        test_gen = valid_test_datagen.flow_from_directory(
            self.test_path,
            target_size=self.target_size,
            color_mode='rgb',
            class_mode= 'categorical',
            batch_size=1024,
            shuffle=False)
        
        return train_gen, valid_gen, test_gen