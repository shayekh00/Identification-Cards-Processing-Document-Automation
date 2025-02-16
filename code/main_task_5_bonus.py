import image_processing_pipeline
import json
import os
import cv2 as cv
import pytesseract
# for testing
from datetime import datetime
from PIL import Image
import numpy as np

def main():
    debug = False # if True, will save images into folder `predicted_images` for the different steps
    dirname = os.path.dirname(__file__)
    pipeline = image_processing_pipeline.ImageProcessingPipeline()
    
    # testing against 6_end_to_end-dataset
    image_dir_path = os.path.join(dirname,'dataset/5_ocr_bonus_task/')
    with open(image_dir_path + 'gt.json', 'r') as json_file:
        img_data = json.load(json_file)
        
    number_of_gt_words = 0
    number_of_gt_words_found_uncleaned = 0
    number_of_words_found_uncleaned = 0
    number_of_gt_words_found_cleaned = 0
    number_of_words_found_cleaned = 0
        
    for file_name in img_data:
        img_path = os.path.join(image_dir_path,file_name)
        img = cv.imread(img_path)
        
        save_images_path = os.path.join(dirname, 'predicted_images/')
                            
        rotation = pipeline.rotate_image(img / 255.0)
        #print(f"shape of rotation: {rotation.shape}") #debug
        if debug:
            rotation_file_name = datetime.now().strftime("%y%m%d_%H%M%S")+'5_ocr_bonus_task-0-'+file_name
            image = Image.fromarray(rotation)
            image.save(save_images_path+rotation_file_name)
        
        binarization = pipeline.binarize_image(rotation)
        binarization = (binarization*255.0).astype(np.uint8)
        #print(f"shape of binarization: {binarization.shape}") #debug
        if debug:
            binarization_file_name = datetime.now().strftime("%y%m%d_%H%M%S")+'5_ocr_bonus_task-1-'+file_name
            image = Image.fromarray(binarization)
            image.save(save_images_path+binarization_file_name)
        
        image_text_uncleaned = pytesseract.image_to_string(rotation, lang='deu')
        image_text_cleaned = pytesseract.image_to_string(binarization, lang='deu')
        
        for word in img_data[file_name]:
            number_of_gt_words = number_of_gt_words + 1
            if image_text_uncleaned:
                if word in image_text_uncleaned.split():
                    number_of_gt_words_found_uncleaned = number_of_gt_words_found_uncleaned + 1
            if image_text_cleaned:
                if word in image_text_cleaned.split():
                    number_of_gt_words_found_cleaned = number_of_gt_words_found_cleaned + 1
                    
        number_of_words_found_uncleaned = number_of_words_found_uncleaned + len(image_text_uncleaned.split())
        number_of_words_found_cleaned = number_of_words_found_cleaned + len(image_text_cleaned.split())
    
    print("\n--------------------")
    print("Uncleaned Images:")
    print(f"Total number of ground truth words: {number_of_gt_words}; Number of ground truth words found in image: {number_of_gt_words_found_uncleaned}; Number of words found in image: {number_of_words_found_uncleaned}")
    print(f"With that, the recall is: {np.round(number_of_gt_words_found_uncleaned/number_of_gt_words,3)}")
    print(f"With that, the precision is: {np.round(number_of_gt_words_found_uncleaned/number_of_words_found_uncleaned,3)}")
    print("--------------------")
    print("\n--------------------")
    print("Cleaned Images:")
    print(f"Total number of ground truth words: {number_of_gt_words}; Number of ground truth words found in image: {number_of_gt_words_found_cleaned}; Number of words found in image: {number_of_words_found_cleaned}")
    print(f"With that, the recall is: {np.round(number_of_gt_words_found_cleaned/number_of_gt_words,3)}")
    print(f"With that, the precision is: {np.round(number_of_gt_words_found_cleaned/number_of_words_found_cleaned,3)}")
    print("--------------------")

if __name__ == "__main__":
    main()