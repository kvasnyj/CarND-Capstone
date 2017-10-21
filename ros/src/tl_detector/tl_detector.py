#!/usr/bin/env python
import rospy
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped, Pose
from styx_msgs.msg import TrafficLightArray, TrafficLight
from styx_msgs.msg import Lane
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from light_classification.tl_classifier import TLClassifier
import math
import tf
import cv2
import numpy as np
import yaml
import os

STATE_COUNT_THRESHOLD = 3
HORIZON = 100

# Constants for where to save camera images if debugging.
DEBUG = False
SAVE_IMAGES = False
IMAGE_DIR = './images'

# Dictionary mapping traffic light color integers to strings.
UNKNOWN=4
GREEN=2
YELLOW=1
RED=0
TRAFFIC_LIGHT_COLORS = dict(
    (k,v) for k,v in zip([UNKNOWN, GREEN, YELLOW, RED], ['UNKNOWN', 'GREEN', 'YELLOW', 'RED']))

def distance(pos1, pos2):
    x1 = pos1.position.x
    y1 = pos1.position.y
    x2 = pos2.position.x
    y2 = pos2.position.y
    z1 = pos1.position.z
    z2 = pos1.position.z
    return math.sqrt((x2-x1)**2+(y2-y1)**2+(z1-z2)**2)

def coordToPoseStamped(coord):
    msg = PoseStamped()
    msg.pose.position.x = coord[0]
    msg.pose.position.y = coord[1]
    return msg

def quaterion_to_euler(pose):
    """ Get the euler angle from quaterion """
    _, _, angle = tf.transformations.euler_from_quaternion([pose.orientation.x,
                                                     pose.orientation.y,
                                                     pose.orientation.z,
                                                     pose.orientation.w])
    return angle

def is_behind(pose, waypoint):
    """
    Somehow this is not working in the rosbag, it always returns false.
        Args:
            pose (obj) - pose.position.x
            waypoint (obj) - waypoint.position.x

        Return:
             True if the car is behind the waypoint, false otherwise
        Ressources: https://github.com/harinando/sdc-path-planning/blob/master/src/main.cpp#L65
    """
    pose_x = pose.position.x
    pose_y = pose.position.y
    waypoint_x = waypoint.position.x
    waypoint_y = waypoint.position.y
    pose_yaw = quaterion_to_euler(pose)
    return True
    # return ((waypoint_x-pose_x) * math.cos(0 - pose_yaw) - (waypoint_y-pose_y) * math.sin(0 - pose_yaw)) > 0

def saveImage(image, camera_image):
    if not os.path.exists(IMAGE_DIR):
        os.mkdir(IMAGE_DIR)
    img_name = os.path.join(IMAGE_DIR, '%s.png' % camera_image.header.seq)
    cv2.imwrite(img_name, image)


class TLDetector(object):

    lights2waypoint = {}

    def __init__(self):
        rospy.init_node('tl_detector')

        self.pose = None
        self.waypoints = None
        self.camera_image = None
        self.lights = []
        self.light_classifier = None

        sub1 = rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        sub2 = rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        '''
        /vehicle/traffic_lights provides you with the location of the traffic light in 3D map space and 
        helps you acquire an accurate ground truth data source for the traffic light
        classifier by sending the current color state of all traffic lights in the
        simulator. When testing on the vehicle, the color state will not be available. You'll need to
        rely on the position of the light and the camera image to predict it.
        '''
        sub3 = rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray, self.traffic_cb)
        sub6 = rospy.Subscriber('/image_color', Image, self.image_cb)

        config_string = rospy.get_param("/traffic_light_config")
        self.config = yaml.load(config_string)

        self.upcoming_red_light_pub = rospy.Publisher('/traffic_waypoint', Int32, queue_size=1)
        self.bridge = CvBridge()

        # Load Traffic Light Classifier
        model_path = rospy.get_param('~model_path')
        label_map_path = rospy.get_param('~label_map_path')
        num_classes = int(rospy.get_param('~num_classes'))
        self.light_classifier = TLClassifier(model_path, label_map_path, num_classes)
        rospy.loginfo('Finished Loading Traffic Classifier')

        self.listener = tf.TransformListener()

        self.state = TrafficLight.UNKNOWN
        self.last_state = TrafficLight.UNKNOWN
        self.last_wp = -1
        self.state_count = 0

        rospy.spin()

    def pose_cb(self, msg):
        self.pose = msg

    def waypoints_cb(self, waypoints):
        self.waypoints = waypoints

    def traffic_cb(self, msg):
        self.lights = msg.lights

    def image_cb(self, msg):
        """Identifies red lights in the incoming camera image and publishes the index
            of the waypoint closest to the red light to /traffic_waypoint
        Args:
            msg (Image): image from car-mounted camera
        """
        self.has_image = True
        self.camera_image = msg
        # Optionally save images for debugging classifications.
        if SAVE_IMAGES:
            cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "bgr8")
            saveImage(cv_image, self.camera_image)

        light_wp, state = self.process_traffic_lights()

        '''
        Publish upcoming red lights at camera frequency.
        Each predicted state has to occur `STATE_COUNT_THRESHOLD` number
        of times till we start using it. Otherwise the previous stable state is
        used.
        '''
        if self.state != state:
            self.state_count = 0
            self.state = state
        elif self.state_count >= STATE_COUNT_THRESHOLD:
            self.last_state = self.state
            light_wp = light_wp if state == TrafficLight.RED else -1
            self.last_wp = light_wp
            self.upcoming_red_light_pub.publish(Int32(light_wp))
        else:
            self.upcoming_red_light_pub.publish(Int32(self.last_wp))
        self.state_count += 1

    def get_closest_waypoint(self, pose):
        """Identifies the closest path waypoint to the given position
            https://en.wikipedia.org/wiki/Closest_pair_of_points_problem
        Args:
            pose (Pose): position to match a waypoint to
                pose.position.x or pose.position.y
                self.waypoints.waypoints[i].pose.pose.position.x
        Returns:
            int: index of the closest waypoint in self.waypoints
        """
        if self.waypoints is None:
            return 0

        closest_dist = float('inf')
        closest_way_point = 0

        for i, waypoint in enumerate(self.waypoints.waypoints):
            dist = distance(pose, waypoint.pose.pose)
            if dist < closest_dist:
                closest_dist = dist
                closest_way_point = i
        return closest_way_point

    def get_next_waypoint(self, pose):
        closest_way_point = self.get_closest_waypoint(pose)
        is_car_in_front = not is_behind(pose, self.waypoints.waypoints[closest_way_point].pose.pose)
        if is_car_in_front:
            closest_way_point += 1
        return closest_way_point

    def get_next_visible_light_waypoint(self, car_position, stop_line_positions):
        """Project point from 3D world coordinates to 2D camera image location
        Args:
            car_position (Int): closest waypoint near the car
            stop_line_positions [Point]: stop line coordinate
        Returns:
            light_wp (Int): visible light waypoint -1 if not found
        TODO:
            - Consider car's heading to handle edge cases where the car goes off-road or drive backward
            - Should handle cases where waypoint is None
            [[1148.56, 1184.65], [1559.2, 1158.43], [2122.14, 1526.79], [2175.237, 1795.71], [1493.29, 2947.67], [821.96, 2905.8], [161.76, 2303.82], [351.84, 1574.65]]
            [[20.991, 22.837]]
        """
        for idx, stop_light in enumerate(stop_line_positions):
            stop_light_pose = coordToPoseStamped(stop_light)

            if TLDetector.lights2waypoint.get(idx, None) is None:
                TLDetector.lights2waypoint[idx] = self.get_closest_waypoint(stop_light_pose.pose) # [292, 753, 2047, 2580, 6294, 7008, 8540, 9733]
            light_wp = TLDetector.lights2waypoint[idx]

            #print(car_position, light_wp, \
            #     is_behind(self.waypoints.waypoints[car_position].pose.pose, self.waypoints.waypoints[light_wp].pose.pose))
            if math.fabs(light_wp - car_position) < HORIZON and \
                    is_behind(self.waypoints.waypoints[car_position].pose.pose, self.waypoints.waypoints[light_wp].pose.pose):
                return light_wp, idx
        return -1, -1

    def get_light_state(self, light):
        """Determines the current color of the traffic light
        Args:
            light (TrafficLight): light to classify
        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)
        """
        if not self.has_image:
            self.prev_light_loc = None
            return TrafficLight.UNKNOWN

        if not light or self.light_classifier is None:
            return TrafficLight.UNKNOWN

        cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "rgb8")

        # Get classification
        light_state = self.light_classifier.get_classification(cv_image)
        return light_state

    def process_traffic_lights(self):
        """Finds closest visible traffic light, if one exists, and determines its
            location and color
        Returns:
            int: index of waypoint closes to the upcoming stop line for a traffic light (-1 if none exists)
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)
        """
        light = None
        light_wp = -1
        car_position = None
        light_idx = -1
        # List of positions that correspond to the line to stop in front of for a given intersection
        stop_line_positions = self.config['stop_line_positions']
        if self.pose:
            car_position = self.get_closest_waypoint(self.pose.pose)

        if car_position:
            light_wp, light_idx = self.get_next_visible_light_waypoint(car_position, stop_line_positions)

        if light_wp > -1:
            light = self.lights[light_idx]

        if light:
            state = self.get_light_state(light)

            if DEBUG:
                rospy.loginfo("CAR_POS: %s, IDX: %s, PREDICTED STATE: %s, ACTUAL STATE: %s",
                    car_position,
                              light_wp,TRAFFIC_LIGHT_COLORS.get(state),TRAFFIC_LIGHT_COLORS.get(light.state))

            return light_wp, state
        else:
            if DEBUG:
                rospy.loginfo("CAR_POS: %s, 'LIGHT_WP: %s', IDX: %s", car_position, stop_line_positions[light_idx], light_wp)
        return -1, TrafficLight.UNKNOWN

if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')
