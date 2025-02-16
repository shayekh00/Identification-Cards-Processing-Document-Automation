# Import relevant libraries
import numpy as np
import tensorflow as tf
import os
import cv2 as cv
import pytesseract
from PIL import Image

# The following metadata is for our best models, but users can technically specify models when creating the
# ImageProcessingPipeline-object, if they so choose
dirname = os.path.dirname(__file__)
# task 1
constant_classification_metadata = (128,128,0.6,os.path.join(dirname,'models/ipda_task_1_clean_16'))
# task 2
constant_segmentation_metadata = (256,256,0.1,os.path.join(dirname,'models/ipda_task_2_clean_3'))
# task 4
constant_binarization_metadata = (512,512,os.path.join(dirname,'models/ipda_task_4_clean_2'))

#print(f"pandas-version: {tf.__version__}")
#print(f"numpy-version: {np.__version__}")
#print(f"tensorflow-version: {tf.__version__}")
#print(f"opencv-version: {cv.__version__}")
#print(f"pytesseract-version: {pytesseract.__version__}")

class ImageProcessingPipeline():
    def __init__(self, classification_metadata=None, segmentation_metadata=None, binarization_metadata=None) -> None:
        if classification_metadata is None:
            self.classification_img_width, self.classification_img_height, self.classification_threshold, self.classification_model_path = constant_classification_metadata
        else:
            self.classification_img_width, self.classification_img_height, self.classification_threshold, self.classification_model_path = classification_metadata
        self.classification_model = tf.keras.saving.load_model(self.classification_model_path)
        
        if segmentation_metadata is None:
            self.segmentation_img_width, self.segmentation_img_height, self.segmentation_threshold, self.segmentation_model_path = constant_segmentation_metadata
        else:
            self.segmentation_img_width, self.segmentation_img_height, self.segmentation_threshold, self.segmentation_model_path = segmentation_metadata
        self.segmentation_model = tf.keras.saving.load_model(self.segmentation_model_path)
        
        if binarization_metadata is None:
            self.binarization_img_width, self.binarization_img_height, self.binarization_model_path = constant_binarization_metadata
        else:
            self.binarization_img_width, self.binarization_img_height, self.binarization_model_path = binarization_metadata
        self.binarization_model = tf.keras.saving.load_model(self.binarization_model_path)
        
    def classify_image(self, img: np.ndarray):
        """Classifies an input image `img` according to whether it's an ID or not. The prediction is a float between 0 and 1
                with 0 meaning the model's very sure it contains an ID and 1 meaning the model's very sure it doesn't contain an ID.

        Args:
            img (np.ndarray): Image as an np.ndarray with shape (x,y,3) and the expected values are float values between 0 and 1.

        Returns:
            numpy.float32: Prediction between 0 and 1. 0 meaning model's sure that img contains an ID and 1 meaning sure that img 
                doesn't contain an ID.
        """
        img_resized = self.classify_image_pre_processing(img)
        prediction = self.classification_model.predict(img_resized)
        return prediction[0][0]
    
    def segment_image(self, img: np.ndarray):
        """Segments and crops an image containing an ID.

        Args:
            img (np.ndarray): Image as an np.ndarray with shape (x,y,3) and the expected values are float values between 0 and 1.

        Returns:
            np.ndarray: Segmented and cropped ID. The values of the image are float values between 0 and 1.
        """
        img_resized = self.segment_image_pre_processing(img)
        predicted_mask = self.segmentation_model.predict(img_resized)
        img_segmented = self.segment_image_apply_mask_to_image(img/255.0,predicted_mask[0]) # /255.0 to have same format as predicted_mask
        img_cropped = self.segment_image_crop_image_using_mask(img_segmented,predicted_mask[0])
        return img_cropped
    
    def rotate_image(self, img: np.ndarray):
        """Rotates an image containing the segmented and cropped image of an ID.

        Args:
            img (np.ndarray): Image as an np.ndarray with shape (x,y,3) and the expected values are float values between 0 and 1.

        Returns:
            np.ndarray: Rotated version of the segmented and cropped ID. The values of the image are integer values between 0 and 255.
        """
        img_prepared = self.rotate_image_pre_processing(img)
        angle = self.rotate_image_perform_hough_transform(img_prepared, 75 , 25)
        rotated_image = self.rotate_image_rotate_by_dominant_angle((img * 255.0).astype(np.uint8),angle,True)
        return rotated_image
    
    def binarize_image(self, img: np.ndarray):
        """Cleans and binarizes an image with an already rotated ID.

        Args:
            img (np.ndarray): Image as an np.ndarray with shape (x,y,3) and the expected values are float values between 0 and 1.

        Returns:
            np.ndarray: Cleaned and binarized version of the roated ID. The values of the image are float values between 0 and 1.
        """
        img_prepared = self.binarize_image_pre_processing(img)
        predicted_binarization = self.binarization_model.predict(img_prepared)[0].squeeze(axis=-1) # turn shape (1,512,512,1) into (512,512)
        #original_height, original_width, _ = img.shape
        #final_binarized = self.binarize_image_post_processing(predicted_binarization,original_height,original_width) # TODO: testing whether this actually does anything
        return predicted_binarization
    
    def process_image(self, img: np.ndarray):
        """Takes an unclassified, unsegmented, not yet rotated, and not yet cleaned & binarized ID and returns the on it as a string.

        Args:
            img (np.ndarray): Image as an np.ndarray with shape (x,y,3) and the expected values are float values between 0 and 1.

        Returns:
            str: String of the text contained in the input image.
        """
        # expects img.shape = (x, y, 3)
        prediction = self.classify_image(img)
        if prediction >= 0.6:
            return
        
        cropped_segmentation = self.segment_image(img) # (x,y,3) w/ x,y element of [0,1] of float numbers
        rotation = self.rotate_image(cropped_segmentation) # (x,y,3) w/ x,y element of [0,255] of integer numbers
        binarization = self.binarize_image(rotation)
        ocr_ready = self.process_image_pre_processing(binarization)
        return pytesseract.image_to_string(ocr_ready, lang='deu')
    
    
    # Helper-Methods for General Image Pre-Processing / Post-Processing
    
    def classify_image_pre_processing(self, img: np.ndarray):
        """Prepares a given image for the `classify_image`-function according to parameters defined in `self`.

        Args:
            img (np.ndarray): Image as an np.ndarray with shape (x,y,3) and the expected values are float values between 0 and 1.

        Returns:
            np.ndarray: Resized img into (1,x_hat,y_hat,3). x_hat is self.classification_img_width and y_hat is
                self.classification_img_height.
        """
        img_resized = cv.resize(img, (self.classification_img_width,self.classification_img_height), interpolation=cv.INTER_NEAREST)
        img_resized = img_resized / 255.0
        img_resized = np.expand_dims(img_resized,axis=0) # set batch dimension to 1
        return img_resized
    
    def segment_image_pre_processing(self, img: np.ndarray):
        """Prepares a given image for the `segment_image`-function according to parameters defined in `self`.

        Args:
            img (np.ndarray): Image as an np.ndarray with shape (x,y,3) and the expected values are float values between 0 and 1.

        Returns:
            np.ndarray: Resized img into (1,x_hat,y_hat,3). x_hat is self.segmentation_img_width and y_hat is
                self.segmentation_img_height.
        """
        img_resized = cv.resize(img, (self.segmentation_img_width,self.segmentation_img_height), interpolation=cv.INTER_LANCZOS4)
        img_resized = img_resized / 255.0
        img_resized = np.expand_dims(img_resized,axis=0) # set batch dimension to 1
        return img_resized
    
    def rotate_image_pre_processing(self, img: np.ndarray):
        """Prepares a given image for the `rotate_image`-function.

        Args:
            img (np.ndarray): Image as an np.ndarray with shape (x,y,3) and the expected values are float values between 0 and 1.

        Returns:
            np.ndarray: Grayscale version of the input image with the same shape but the values are now integer values between
                0 and 255.
        """
        # turns image into grayscale representation
        return cv.cvtColor((img * 255.0).astype(np.uint8), cv.COLOR_RGB2GRAY)
    
    def binarize_image_pre_processing(self, img:np.ndarray):
        """Prepares a given image for the `binarize_image`-function according to parameters defined in `self`.

        Args:
            img (np.ndarray): Image as an np.ndarray with shape (x,y,3) and the expected values are integer values between 0 and 255.

        Returns:
            np.ndarray: Resized img into (1,x_hat,y_hat,3). x_hat is self.binarization_img_width and y_hat is
                self.binarization_img_height.
        """
        img_resized = cv.resize(img, (self.binarization_img_width,self.binarization_img_height), interpolation=cv.INTER_LANCZOS4)
        img_resized = img_resized / 255.0
        img_resized = np.expand_dims(img_resized,axis=0) # set batch dimension to 1
        return img_resized
    
    def binarize_image_post_processing(self, img:np.ndarray, height, width):
        """Legacy function. Was used for turning the image dimensions of the cleaned and binarized version back to the original
            image's dimensions. This reduced the OCR's performance and was thus scrapped.

        Args:
            img (np.ndarray): Image as an np.ndarray with shape (x,y,3) and the expected values are float values between 0 and 1.

        Returns:
            np.ndarray: Resized img into (1,x_hat,y_hat,3). x_hat is the previous image's x-axis resolution and y_hat is the previous 
                image's y-axis resolution.
        """
        img_resized = cv.resize((img * 255.0).astype(np.uint8), (width,height), interpolation=cv.INTER_LANCZOS4)
        return img_resized
    
    def process_image_pre_processing(self, img:np.ndarray):
        """Prepares an image for the OCR.

        Args:
            img (np.ndarray): Image as an np.ndarray with shape (x,y,3).

        Returns:
            Image: Image object of the input np.ndarray.
        """
        if not np.any(img > 1):
            img_prepared = (img * 255.0).astype(np.uint8)
        else:
            img_prepared = img
        return Image.fromarray(img_prepared)
    
    
    # Helper-Methods for Image Segmentation (Task 2)
    
    def segment_image_threshold_mask(self, mask,threshold_value):
        """Thresholds a given numpy array such that it's values are either 0 or 1.

        Args:
            mask (np.ndarray): Numpy array of shape (x,y). 
            threshold_value (float): Threshold value representing the cut off point at which the value is either set to 0 or 1.

        Returns:
            np.ndarray: Thresholded numpy array.
        """
        return np.where(mask >= threshold_value, 1, 0).astype(np.uint8)

    def segment_image_resize_mask_to_image(self, image, mask):
        """Resizes a given mask to the same dimensions as a given image (x and y).

        Args:
            image (np.ndarray): Numpy array of shape (x,y,3) with x being the height, and y being the width 
                channels.
            mask (np.ndarray): Numpy array of shape (x,y).

        Returns:
            np.ndarray: Numpy array of the mask, resized to the same dimensions as image.
        """
        image_height, image_width = image.shape[:2]
        resized_mask = cv.resize(mask, (image_width, image_height), interpolation=cv.INTER_CUBIC)
        return resized_mask

    def segment_image_apply_mask_to_image(self, image, mask):
        """Applies a mask to a given image. The mask may have any resolution and will be resized to the same size as the input
            image before being applied.

        Args:
            image (np.ndarray): Numpy array of shape (x,y,3).
            mask (np.ndarray): Numpy array of shape (x,y).

        Returns:
            np.ndarray: Numpy array of the input image with the mask applied to it.
        """
        # Resize Mask
        resized_mask = self.segment_image_resize_mask_to_image(image, mask)

        # apply thresholding
        thresholded_mask = self.segment_image_threshold_mask(resized_mask,self.segmentation_threshold)

        # applying the mask to the image
        masked_image = cv.bitwise_and(image, image, mask=thresholded_mask)
        return masked_image

    def segment_image_create_bounding_box(self,mask):
        """Finds the biggest contour in an image and defines the bounding box for it. It is used to find the rectangle which 
            we want to crop.

        Args:
            mask (np.ndarray): Numpy array of shape (x,y).

        Returns:
            tuple: `x`- and `y`-coordinates of the upper left corner of the bounding box and the width and height of the 
                box.
        """
        thresholded_mask = self.segment_image_threshold_mask(mask,self.segmentation_threshold)

        # Find contours from the mask
        contours, _ = cv.findContours(thresholded_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        # Get the bounding box coordinates of the largest contour
        if contours:
            largest_contour = max(contours, key=cv.contourArea)
            x, y, w, h = cv.boundingRect(largest_contour)
            return x, y, w, h
        else:
            return None

    def segment_image_crop_image_using_mask(self, image, mask):
        """Crops a given image with a mask.

        Args:
            image (np.ndarray): Numpy array of shape (x,y,3).
            mask (np.ndarray): Numpy array of shape (x,y).

        Returns:
            mixed: If a contour is found, the cropped image is returned. Otherwise, it returns None.
        """
        # Resize Mask
        resized_mask = self.segment_image_resize_mask_to_image(image, mask)

        # Create bounding box around the mask
        x, y, w, h = self.segment_image_create_bounding_box(resized_mask)

        if x is not None:
            # Crop the image using the bounding box coordinates
            cropped_image = image[y:y+h, x:x+w]
            return cropped_image
        else:
            return None
    
    
    # Helper-Methods for Image Rotation (Task 3)
    
    def rotate_image_perform_hough_transform(self,variable, threshold, opening_kernel_pixels):
        """Finds out the angle to rotate by with probabilistic hough transform.

        Args:
            variable (np.ndarray): Numpy array of shape (x,y).
            threshold (int): Threshold for probabilistic hough transform.
            opening_kernel_pixels (int): Determines width and height of the filter for the opening (erosion & dilation).

        Raises:
            ValueError: Handles the input image.

        Returns:
            float: Angle to rotate by in degrees.
        """
        # Handle variable
        if isinstance(variable, str):
            src = cv.imread(variable, cv.IMREAD_GRAYSCALE)
        elif isinstance(variable, np.ndarray):
            src = variable
        else:
            raise ValueError("Variable is neither a string nor a np.array.")

        # Perform Opening
        if opening_kernel_pixels is not None:
            # Perform erosion
            kernel = np.ones((opening_kernel_pixels,opening_kernel_pixels),np.uint8)
            erosion = cv.erode(src,kernel,iterations = 1)
            # Perform dilation
            src = cv.dilate(erosion,kernel,iterations = 1) # reassign opened image to src-variable


        dst = cv.Canny(src, 200, 240, None, 3)

        # save angles of all lines in angles
        angles = []

        # threshold from 50 to 500
        linesP = cv.HoughLinesP(dst, 1, np.pi / 180, threshold, None, 50, 10)

        if linesP is not None:
            for i in range(0, len(linesP)):
                l = linesP[i][0]

                angle = np.arctan2(l[3] - l[1], l[2] - l[0]) * 180.0 / np.pi
                angles.append(angle)

        dominant_angle = None
        if angles:
            angles = np.array(angles)
            hist, bin_edges = np.histogram(angles, bins=180)  # Create a histogram of angles
            dominant_bin = np.argmax(hist)  # Find the bin with the highest frequency
            dominant_angle = bin_edges[dominant_bin]  # Get the dominant angle

        return dominant_angle

    def rotate_image_rotate_by_dominant_angle(self,variable, dominant_angle, crop_image):
        """Rotate a given image by an angle and possibly crop the image.

        Args:
            variable (np.ndarray): Numpy array of shape (x,y,3).
            dominant_angle (float): Angle to rotate by.
            crop_image (bool): Decides whether ID in image is cropped.

        Raises:
            ValueError: Handle input image type.
            ValueError: Handle input image color channel.

        Returns:
            np.ndarray: Rotated and possibly cropped image.
        """
        # Handle variable
        if isinstance(variable, str):
            src = cv.imread(variable)
        elif isinstance(variable, np.ndarray):
            src = variable
        else:
            raise ValueError("Variable is neither a string nor a np.array.")

        # Check if the image is grayscale or color
        if len(src.shape) == 2:  # Grayscale image
            channels = [src]
        elif len(src.shape) == 3:  # Color image
            channels = [src[:, :, i] for i in range(src.shape[2])]
        else:
            raise ValueError("Unsupported number of image channels.")

        # Rotate each channel separately
        rotated_channels = []
        for channel in channels:
            rows, cols = channel.shape
            M = cv.getRotationMatrix2D((cols / 2, rows / 2), dominant_angle, 1)
            rotated_channels.append(cv.warpAffine(channel, M, (cols, rows)))
        #rotated_channels = [cv.warpAffine(channel, cv.getRotationMatrix2D((channel.shape[1] / 2, channel.shape[0] / 2), dominant_angle, 1), (channel.shape[1], channel.shape[0])) for channel in channels]

        # Combine the rotated channels
        rotated_img = np.stack(rotated_channels, axis=-1)

        if crop_image:
            # Crop image
            contours, _ = cv.findContours(rotated_img[:, :, 0], cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE) # find contours
            if contours:
                largest_contour = max(contours, key=cv.contourArea)
                x, y, w, h = cv.boundingRect(largest_contour)
                cropped_image = rotated_img[y:y+h, x:x+w, :]
                return cropped_image
            else:
                return rotated_img
        else:
            return rotated_img