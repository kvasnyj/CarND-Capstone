from styx_msgs.msg import TrafficLight

import cv2
import numpy as np
import pickle as pkl
import tensorflow as tf

class TLClassifier(object):

    def __init__(self, model_path, label_map_path, num_classes):
        """Traffic Light Classifier Construction"""
        self.num_classes = num_classes
        self.label_map_path = label_map_path
        self.category_index = self._load_label_map()
        self.model_path = model_path

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(self.model_path, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
            self.session = tf.Session(graph=self.detection_graph, config=config)
            

        #Definite input and output Tensors for detection_graph
        self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        # Each box represents a part of the image where a particular object was detected.
        self.detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        self.detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        self.num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')

    def _load_label_map(self):
        category_index = pkl.load(open(self.label_map_path, 'r'))
        return category_index

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image, axis=0)
        # Actual detection.
        (boxes, scores, classes, num) = self.session.run(
            [self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections],
            feed_dict={self.image_tensor: image_np_expanded})
        if num > 0:
            scores = scores.squeeze()
            argmax_score = np.argmax(scores)
            if scores[argmax_score] > 0.5:
                classes = classes.squeeze()
                class_name = self.category_index[classes[argmax_score]]['name']
                if 'Red' in class_name:
                    return TrafficLight.RED
                elif 'Yellow' in class_name:
                    return TrafficLight.YELLOW
                elif 'Green' in class_name:
                    return TrafficLight.GREEN
                else:
                    return TrafficLight.UNKNOWN
        else:
            return TrafficLight.UNKNOWN