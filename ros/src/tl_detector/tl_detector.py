#!/usr/bin/env python
import rospy
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped, Pose
from styx_msgs.msg import TrafficLightArray, TrafficLight
from styx_msgs.msg import Lane
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from light_classification.tl_classifier import TLClassifier
from math import sqrt
import tf
import cv2
import numpy as np
import yaml

STATE_COUNT_THRESHOLD = 3
HORIZON = 50

def distance(pos1, pos2):
    x1 = pos1.position.x
    y1 = pos1.position.y
    x2 = pos2.position.x
    y2 = pos2.position.y
    return sqrt((x2-x1)*(x2-x1)+(y2-y1)*(y2-y1))

def coordToPoseStamped(coord):
    msg = PoseStamped()
    msg.pose.position.x = coord[0]
    msg.pose.position.y = coord[1]
    return msg

class TLDetector(object):
    def __init__(self):
        rospy.init_node('tl_detector')

        self.pose = None
        self.waypoints = None
        self.camera_image = None
        self.lights = []

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
        self.light_classifier = TLClassifier()
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

        closestDist = float('inf')
        closestWayPoint = 0

        for i, waypoint in enumerate(self.waypoints.waypoints):
            dist = distance(pose, waypoint.pose.pose)
            if dist < closestDist:
                closestDist = dist
                closestWayPoint = i

        return closestWayPoint

    def get_next_visible_light_waypoint(self, car_position, stop_line_positions):
        """Project point from 3D world coordinates to 2D camera image location

        Args:
            car_position (Int): closest waypoint near the car
            stop_line_positions [Point]: stop line coordinate

        Returns:
            light_wp (Int): visible light waypoint -1 if not found

        TODO:
            - Should memoize light_waypoints to avoid extra computation
            - Consider car heading to handle edge cases where the car goes off-road
            - Should handle cases where waypoint is None

 [[1148.56, 1184.65], [1559.2, 1158.43], [2122.14, 1526.79], [2175.237, 1795.71], [1493.29, 2947.67], [821.96, 2905.8], [161.76, 2303.82], [351.84, 1574.65]]
        """
        light_waypoints = []
        for idx, stop_light in enumerate(stop_line_positions):
            stop_light_pose = coordToPoseStamped(stop_light)
            light_waypoints.append(self.get_closest_waypoint(stop_light_pose.pose)) # [292, 753, 2047, 2580, 6294, 7008, 8540, 9733]

        for i, light_wp in enumerate(light_waypoints):
            if light_wp > car_position and light_wp - car_position < HORIZON:
                return light_wp, i
        return -1, -1

    def project_to_image_plane(self, point_in_world):
        """Project point from 3D world coordinates to 2D camera image location

        Args:
            point_in_world (Point): 3D location of a point in the world

        Returns:
            x (int): x coordinate of target point in image
            y (int): y coordinate of target point in image

        """

        fx = self.config['camera_info']['focal_length_x']
        fy = self.config['camera_info']['focal_length_y']
        image_width = self.config['camera_info']['image_width']
        image_height = self.config['camera_info']['image_height']

        # get transform between pose of camera and world frame
        trans = None
        try:
            now = rospy.Time.now()
            self.listener.waitForTransform("/base_link",
                  "/world", now, rospy.Duration(1.0))
            (trans, rot) = self.listener.lookupTransform("/base_link",
                  "/world", now)

        except (tf.Exception, tf.LookupException, tf.ConnectivityException):
            rospy.logerr("Failed to find camera to map transform")

        # Generate Homogenous Coordinates of point in world.
        X = [point_in_world.x, point_in_world.y, point_in_world.z, 1]
        # Create a transformation matrix T mapping points for world coordinate frame to camera frame.
        T = self.listener.fromTranslationRotation(trans, rot)
        # Create camera projection matrix K with optical center at center of image.
        K = np.array([[fx,0,image_width/2,0],[0,fy,image_height/2,0],[0,0,1,0]])
        # Generate Homogenous Coordinates of point in camera frame
        P = np.dot(K, np.dot(T,X))
        # Generate pixel coordinates of point in camera frame
        x = P[0] / P[2]
        y = P[1] / P[2]

        return int(x), int(y)

    def get_light_state(self, light):
        """Determines the current color of the traffic light

        Args:
            light (TrafficLight): light to classify

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        if(not self.has_image):
            self.prev_light_loc = None
            return False

        cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "bgr8")

        x, y = self.project_to_image_plane(light.pose.pose.position)

        #Get classification
        light_state = self.light_classifier.get_classification(cv_image[:y,:,:])
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

        if (self.pose):
            car_position = self.get_closest_waypoint(self.pose.pose)

        if car_position:
            light_wp, light_idx = self.get_next_visible_light_waypoint(car_position, stop_line_positions)

        if light_wp > -1:
            light = self.lights[light_idx]

        rospy.loginfo("CAR_POS: %s, 'LIGHT_WP: %s', IDX: %s", car_position, stop_line_positions[light_idx], light_wp)

        if light:
            state = self.get_light_state(light)
            return light_wp, state
        return -1, TrafficLight.UNKNOWN

if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')
