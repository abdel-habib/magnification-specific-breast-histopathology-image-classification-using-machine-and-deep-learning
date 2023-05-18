import os
import random
import shutil
from loguru import logger

class DataSplit:
    def __init__(self):
      self.src_dir = os.getcwd()
      self.parentfolder = ['100X','200X','400X','40X']
      self.subfolders = ['papillary_carcinoma', 'ductal_carcinoma', 'mucinous_carcinoma', 'lobular_carcinoma', 'adenosis', 'tubular_adenoma', 'fibroadenoma', 'phyllodes_tumor']
      self.subsubfolders = ['test','validation','train']
      self.folderName = 'BreakHis'
      
    def get_files_for_magnification(self,home_dir,magnification):
        src_dir = home_dir = os.path.join(os.getcwd(),home_dir)
        result = []
        folder = []
        for root, dirs, files in os.walk(src_dir):
            for directory in dirs:
                if directory.endswith(magnification):
                    folder_path = os.path.join(root, directory)
                    folder.append(directory)
                    for file_name in os.listdir(folder_path):
                        file_path = os.path.join(folder_path, file_name)
                        result.append(file_path)        
        return result, folder
    
    def search_label_in_array(self,dir_array,label):
        list = []
        for i in range(0,len(dir_array)):
            if dir_array[i].__contains__(label):
                list.append(dir_array[i])
        return list

    def split_directories(self,dir_list,magnification,label):
        print('  ')
        print()
        random.seed(123)
        random.shuffle(dir_list)
        random.shuffle(dir_list)
        random.shuffle(dir_list)
    
        len_50 = round(len(dir_list)*0.5)
        len_30 = round(len(dir_list)*0.30)
    
        train = dir_list[0:len_50]
        test  = dir_list[len_50:len_50+len_30]
        validation = dir_list[len_50+len_30:len(dir_list)]
        
        print('Length of {}_{}: {}'.format(label,magnification,len(dir_list)))
        print('Length of train: {}'.format(len(train)))
        print('Length of validation: {}'.format(len(validation)))
        print('Length of test: {}'.format(len(test)))
    
        return train, validation, test
    
    def make_files(self):

        
        above_path = os.path.join(self.src_dir,'..',self.folderName)
        os.mkdir(above_path)
        
        for element in self.parentfolder:
            path = os.path.join(above_path,element)
            os.mkdir(path)
            for element in self.subfolders:
                sub_folder_path=os.path.join(path,element)
                os.mkdir(sub_folder_path)
                for element in self.subsubfolders:
                    subsub_folder_path = os.path.join(sub_folder_path,element)
                    os.mkdir(subsub_folder_path)
    
    def save_images_in_folder(self,data,folder_name,label,magnification):
        dest_dir = os.path.join(self.src_dir,'..',self.folderName,magnification,label,folder_name)
        for elements in data:
            shutil.copy(elements,dest_dir)
        
                    
    def split_data(self):
        home_dir = os.path.join(self.src_dir, os.path.dirname("../BreaKHis_v1/histology_slides/breast/"))
        for magnifications in self.parentfolder:
            for label in self.subfolders:
                result,folder = self.get_files_for_magnification(home_dir,magnifications)
                labels_array = self.search_label_in_array(result,label)
                train, validation, test = self.split_directories(labels_array,magnifications,label)
                self.save_images_in_folder(train,'train',label,magnifications)
                self.save_images_in_folder(validation,'validation',label,magnifications)
                self.save_images_in_folder(test,'test',label,magnifications)

split = DataSplit()

#uncommet for creating folder strcuture once
split.make_files()

split.split_data()                        
    
    
    



    



        