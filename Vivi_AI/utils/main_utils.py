import os 
import sys 
import cv2
import numpy as np 
import dill 
import yaml 
import pickle
import pandas as pd
from pandas import DataFrame 
from skimage import color, io
from skimage.feature import graycomatrix, graycoprops
from skimage.filters import threshold_otsu
from skimage.measure import shannon_entropy
from tensorflow import keras 
from skimage.util import img_as_ubyte
from glrlm import GLRLM
# import seaborn as sns 
# import matplotlib.pyplot as plt 
from Vivi_AI.exception import CustomException
from Vivi_AI.logger import logging 



###########################Read _yaml method ###################

def read_yaml(file_path:str)->dict:
    try:
        logging.info("ENtered into the read yaml file ")
        with open(file_path,'rb') as yaml_file:
            return yaml.safe_load(yaml_file)
        logging.info("Exiots from the read yaml file method ")
    except Exception as e:
        logging.info(e)
        raise CustomException(e,sys) from e 


################## wite yaml method ############################


def write_yaml_file(file_path:str,content:object,replace:bool=False)->None:
    try:
        logging.info("ENtered into write yaml file ")
        if replace:
            if os.path.exists(file_path):
                os.remove(file_path)
        os.makedirs(os.path.dirname(file_path),exist_ok=True)
        with open(file_path,'w') as file:
            yaml.dump(content,file)

        logging.info("Exist from write yamnl file ")

    except Exception as e:
        logging.info(e)
        raise CustomException(e,sys) from e 
    

#################### Load object ###############################

def load_object(file_path:str)->object:
    try:
        logging.info("ENtered inot the load object of utils ")
        with open(file_path,'rb') as obj:
            object=dill.load(obj)
        logging.info("Exist from the load object method ")
        return object
    except Exception as e:
        logging.info(e)
        raise CustomException(e,sys) from e 



##################### Save numpy ###############################

def save_numpy_array_data(file_path:str,array:np.array):
    try:
        logging.info("Entered inot the save numpy array")
        dir_path=os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        with open(file_path,'wb') as file_obj:
            np.save(file_obj,array)
        logging.info("Exits from save numpy method of utils ")
    except Exception as e:
        logging.info(e)
        raise CustomException(e,sys) from e 



###################### load numpy ##############################

def load_numpy_array_data(file_path: str) -> np.ndarray:
    """
    Load a numpy array from a file.

    :param file_path: Path to the numpy file.
    :return: Numpy array loaded from the file.
    """
    try:
        logging.info("Entered into the load_numpy_array_data method.")
        with open(file_path, 'rb') as file_obj:
            data = np.load(file_obj,allow_pickle=True)
        logging.info("Exited from the load_numpy_array_data method.")
        return data
    except Exception as e:
        logging.error(f"Error occurred while loading numpy array: {e}")
        raise CustomException(e, sys) from e
    



###################### save _bject #############################

def save_object(file_path:str,obj:object)->None:
    try:
        logging.info("ENtered into the save object method ")
        os.makedirs(os.path.dirname(file_path),exist_ok=True)
        with open(file_path,'wb') as file_obj:
            dill.dump(obj,file_obj)

        logging.info("Exited the save object of utils ")

    except Exception as e:
        logging.info(e)
        raise CustomException(e,sys) from e 





######################### Drop Columns #########################

def drop_columns(df:DataFrame,cols:list)->DataFrame:
    try:
        logging.info("ENtered drop_columsn method od utils ")
        df=df.drop(columns=cols,axis=1)
        logging.info("Exited the drop colums method od utils ")
        return df
        logging.info("Exited from the drop columsn ")
    except Exception as e:
        logging.info(e)
        raise CustomException(e,sys) from e 
    


######################### Feature Extraction method ##############

def label_extraction_func(filename):
    """Extracts and returns the label from the filename."""
    return filename.split('_')[0]

def binarize_image(image):
    """Converts an image to binary using Otsu's thresholding."""
    if len(image.shape) == 3:
        gray_image = color.rgb2gray(image)
    else:
        gray_image = image
    thresh = threshold_otsu(gray_image)
    binary_image = gray_image < thresh
    return binary_image



def entropy(image):
    """Computes and returns the Shannon entropy of an image."""
    return shannon_entropy(image)




def lacunarity(image, window_size):
    """Calculates lacunarity of an image."""
    def sliding_window(image, window_size):
        for x in range(0, image.shape[0] - window_size + 1, window_size):
            for y in range(0, image.shape[1] - window_size + 1, window_size):
                yield image[x:x + window_size, y:y + window_size]

    lacunarity_values = []
    for window in sliding_window(image, window_size):
        mean = np.mean(window)
        if mean > 0:
            lacunarity_values.append(np.var(window) / mean**2)
    return np.mean(lacunarity_values)

def fractal_dimension(image):
    """Calculates the fractal dimension of a binary image."""
    binary_image = binarize_image(image)
    p = min(binary_image.shape)
    n = 2**np.floor(np.log2(p))

    def boxcount(img, k):
        S = np.add.reduceat(
            np.add.reduceat(img, np.arange(0, img.shape[0], k), axis=0),
            np.arange(0, img.shape[1], k), axis=1)
        return len(np.where((S > 0) & (S < k*k))[0])

    sizes = 2**np.arange(int(np.log2(n)), 1, -1)
    counts = [boxcount(binary_image, size) for size in sizes]

    coeffs = np.polyfit(np.log(sizes), np.log(counts), 1)
    return -coeffs[0], sizes, counts

def load_image(image_path):
    """Loads and returns an image from the specified path."""
    image = io.imread(image_path)
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif len(image.shape) == 3 and image.shape[2] == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    return img_as_ubyte(image)

def annotated_image(image_path):
    """Annotates an image with bounding boxes around contours."""
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    edges = cv2.Canny(blurred_image, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    annotated_image = image_rgb.copy()
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(annotated_image, (x, y), (x+w, y+h), (255, 0, 0), 2)
    return annotated_image

def GLCM(image):
    """Computes GLCM features for an image and returns them in a DataFrame."""
    img_dataset = pd.DataFrame()
    
    image =  cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
    distances = [1, 3, 5]
    angles = [0, np.pi/4, np.pi/2]
    props = ['contrast', 'dissimilarity', 'homogeneity', 'ASM', 'energy', 'correlation']
    
    df = pd.DataFrame()
    
    for dist in distances:
        GLCM = graycomatrix(image, [dist], [0])
        for prop in props:
            prop_val = graycoprops(GLCM, prop)[0, 0]
            df[f'{prop.capitalize()}{dist}'] = [prop_val]
    
    for angle in angles:
        GLCM = graycomatrix(image, [1], [angle])
        for prop in props:
            prop_val = graycoprops(GLCM, prop)[0, 0]
            angle_deg = int(np.degrees(angle))
            df[f'{prop.capitalize()}_{angle_deg}'] = [prop_val]
    
    img_dataset = pd.concat([img_dataset, df], ignore_index=True)
    
    return img_dataset


def process_images_and_extract_features_glrm(image):
    """Processes an image to extract GLRLM features and returns them in a DataFrame."""
    img_dataset = pd.DataFrame()
    
    glrlm_props = ['SRE', 'LRE', 'GLU']
    
    # Convert the image to grayscale
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Crop and resize the image
    h, w = image.shape
    ymin, ymax, xmin, xmax = h//3, h*2//3, w//3, w*2//3
    crop = image[ymin:ymax, xmin:xmax]
    resize = cv2.resize(crop, (0, 0), fx=0.5, fy=0.5)
    
    # Calculate GLRLM features
    app = GLRLM()
    glrlm_features = app.get_features(resize, 8)
    return glrlm_features.SRE, glrlm_features.LRE, glrlm_features.GLU

    # Store the features in a DataFrame
    # df = pd.DataFrame([feature], columns=glrlm_props)
    # img_dataset = pd.concat([img_dataset, df], ignore_index=True)
    
    # return img_dataset


def feature_Extraction(image):
    data=[]

    fd, sizes, counts = fractal_dimension(image)
    ent = entropy(image)
    lac = lacunarity(image, window_size=8)
    data.append({
        'fractal_dimension': fd,
        'entropy': ent,
        'lacunarity': lac,
    })
            
    df = pd.DataFrame(data)
    return df




def final_feature_Extr(image):
    # Extract features using different methods
    df1 = feature_Extraction(image)
    # logging.info(f"features of fD{df1}")
    # df_glcm = GLCM(image)/
    # logging.info(f"features glcm:{df_glcm}")
    df_glrlm = process_images_and_extract_features_glrm(image)
    # logging.info(f"GLRMR :{df_glrlm}")
    
    # Reset index and rename columns to 'filename'
    # df1 = df1.reset_index().rename(columns={'index': 'filename'})
    # df_glcm = df_glcm.reset_index().rename(columns={'index': 'filename'})
    # df_glrlm = df_glrlm.reset_index().rename(columns={'index': 'filename'})

    merged_data = pd.concat([df1, df_glrlm], axis=1)
    
    # Add the class label to the merged dataset
    # merged_data['label'] = class_name
    
    return merged_data

def load_model(model_path):
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    return model 






list_=['fractal_dimension', 'entropy', 'lacunarity', 'Contrast3',
       'Dissimilarity3', 'Homogeneity3', 'Asm3', 'Energy3', 'Correlation3',
       'Contrast5', 'Dissimilarity5', 'Homogeneity5', 'Asm5', 'Energy5',
       'Correlation5', 'Contrast_0', 'Dissimilarity_0', 'Homogeneity_0',
       'Asm_0', 'Energy_0', 'Correlation_0', 'Contrast_45', 'Dissimilarity_45',
       'Homogeneity_45', 'Asm_45', 'Energy_45', 'Correlation_45',
       'Contrast_90', 'Dissimilarity_90', 'Homogeneity_90', 'Asm_90',
       'Energy_90', 'Correlation_90', 'SRE', 'LRE', 'GLU', 'RLU']


    

    

