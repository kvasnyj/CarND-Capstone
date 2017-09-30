from styx_msgs.msg import TrafficLight
import numpy as np
import cv2

class TLClassifier(object):
    def __init__(self, max_row, threshold):
        """Traffic light classifier constructor.

        Args:
          max_row: Maximum row in image considered during traffic light state classification. 
          threshold: Threshold for the fraction of image pixels necessary to declare traffic light a specific color 
        """
        self.max_row = max_row
        self.threshold = threshold

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        # Normalize by number of pixels in image
        num_pixels = float(self.max_row) * image.shape[1]

        # Thresholded Red Channel
        _, r_img = cv2.threshold(image[:self.max_row,:,2], 230, 255, cv2.THRESH_BINARY)
        # Thresholded Green Channel
        _, g_img = cv2.threshold(image[:self.max_row,:,1], 245, 255, cv2.THRESH_BINARY)
        # Thresholded Yellow Channel
        y_img = (r_img > 0.) * (g_img > 0.) * 255.0 
        
        # Determine fraction of thresholeded image with pixels exceeding threshold.
        r_frac = (np.sum(r_img)/num_pixels)
        g_frac = (np.sum(g_img)/num_pixels)
        y_frac = (np.sum(y_img)/num_pixels)

        if y_frac > self.threshold:
            return TrafficLight.YELLOW
        elif r_frac > self.threshold:
            return TrafficLight.RED
        elif g_frac > self.threshold:
            return TrafficLight.GREEN
        else:
            return TrafficLight.UNKNOWN
