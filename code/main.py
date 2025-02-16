import image_processing_pipeline
import json
import os
import cv2 as cv
# for testing
from datetime import datetime
from PIL import Image
import numpy as np

def myprint_classification(s):
    with open('classification_model_summary.txt','a') as f:
        print(s, file=f)
        
def myprint_segmentation(s):
    with open('segmentation_model_summary.txt','a') as f:
        print(s, file=f)
        
def myprint_binarization(s):
    with open('binarization_model_summary.txt','a') as f:
        print(s, file=f)

def main():
    debug = False # if True, will save images into folder `predicted_images` for the different steps
    dirname = os.path.dirname(__file__)
    pipeline = image_processing_pipeline.ImageProcessingPipeline()
    
    #pipeline.classification_model.summary(print_fn=myprint_classification)
    #pipeline.segmentation_model.summary(print_fn=myprint_segmentation)
    #pipeline.binarization_model.summary(print_fn=myprint_binarization)
    
    # testing against 6_end_to_end-dataset
    image_dir_path = os.path.join(dirname,'dataset/6_end_to_end/6_end_to_end/')
    with open(image_dir_path + 'gt.json', 'r') as json_file:
        img_data = json.load(json_file)
        
    number_of_gt_words = 0
    number_of_gt_words_found = 0
    number_of_words_found = 0
        
    for file_name in img_data:
        img_path = os.path.join(image_dir_path,file_name)
        img = cv.imread(img_path)
        
        if debug:
            save_images_path = os.path.join(dirname, 'predicted_images/')
            
            prediction = pipeline.classify_image(img)
            print(f"The prediction for {file_name} is {prediction}.")
            
            segmentation = pipeline.segment_image(img)
            #print(f"shape of segmentation: {segmentation.shape}")
            segmentation_file_name = datetime.now().strftime("%y%m%d_%H%M%S")+'-0-'+file_name+'-segmentation.png'
            image = Image.fromarray((segmentation*255.0).astype(np.uint8))
            image.save(save_images_path+segmentation_file_name)
            
            # Perform opening
            # Perform erosion (pre-processing step of rotation)
            kernel = np.ones((25,25),np.uint8)
            erosion = cv.erode((segmentation*255.0).astype(np.uint8),kernel,iterations = 1)
            # Perform dilation
            opened_segmentation = cv.dilate(erosion,kernel,iterations = 1) # reassign opened image to src-variable
            opened_segmentation_file_name = datetime.now().strftime("%y%m%d_%H%M%S")+'-1-'+file_name+'-opened_segmentation.png'
            image = Image.fromarray(opened_segmentation)
            image.save(save_images_path+opened_segmentation_file_name)
            
            # canny edge detection
            dst = cv.Canny(opened_segmentation, 200, 240, None, 3)
            #print(f"shape of dst: {dst.shape}")
            dst_file_name = datetime.now().strftime("%y%m%d_%H%M%S")+'-2-'+file_name+'-canny_edge_detection.png'
            image = Image.fromarray(dst)
            image.save(save_images_path+dst_file_name)
            
            # Try closing (done for some testing regarding the upside down detection question in task 3)
            segmentation = pipeline.segment_image(img)
            kernel = np.ones((25,25),np.uint8)
            # Perform dilation
            dilation = cv.dilate((segmentation*255.0).astype(np.uint8),kernel,iterations = 1)
            # Perform erosion
            closed_segmentation = cv.erode(dilation,kernel,iterations = 1)
            closed_segmentation_file_name = datetime.now().strftime("%y%m%d_%H%M%S")+'-3-'+file_name+'-closed_segmentation.png'
            image = Image.fromarray(closed_segmentation)
            image.save(save_images_path+closed_segmentation_file_name)
                        
            rotation = pipeline.rotate_image(segmentation)
            #print(f"shape of rotation: {rotation.shape}")
            rotation_file_name = datetime.now().strftime("%y%m%d_%H%M%S")+'-4-'+file_name+'-rotation.png'
            image = Image.fromarray(rotation)
            image.save(save_images_path+rotation_file_name)
            
            binarization = pipeline.binarize_image(rotation)
            #print(f"shape of binarization: {binarization.shape}")
            binarization_file_name = datetime.now().strftime("%y%m%d_%H%M%S")+'-5-'+file_name+'-binarization.png'
            image = Image.fromarray((binarization*255.0).astype(np.uint8))
            image.save(save_images_path+binarization_file_name)
        
        image_text = pipeline.process_image(img)
        
        for word in img_data[file_name]:
            number_of_gt_words = number_of_gt_words + 1
            if image_text:
                if word in image_text.split():
                    number_of_gt_words_found = number_of_gt_words_found + 1
                    
        number_of_words_found = number_of_words_found + len(image_text.split())
    
    print(f"Total number of ground truth words: {number_of_gt_words}; Number of ground truth words found in image: {number_of_gt_words_found}; Number of words found in image: {number_of_words_found}")
    print(f"With that, the recall is: {np.round(number_of_gt_words_found/number_of_gt_words,3)}")
    print(f"With that, the precision is: {np.round(number_of_gt_words_found/number_of_words_found,3)}")

if __name__ == "__main__":
    main()