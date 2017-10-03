from styx_msgs.msg import TrafficLight
import numpy as np
import cv2

class TLClassifier(object):
    def __init__(self, threshold):
        self.threshold = threshold
        self.lower_red = np.array([0,  100, 100], dtype="uint8")       # lower_red_hue_range
        self.upper_red = np.array([10, 255, 255], dtype="uint8")       #


    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        Ressources:
            http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_colorspaces/py_colorspaces.html

        """
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        # Threshold the HSV image to get only green colors
        mask = cv2.inRange(hsv_image, self.lower_red, self.upper_red)

        if cv2.countNonZero(mask) > self.threshold:
            return TrafficLight.RED

        return TrafficLight.UNKNOWN