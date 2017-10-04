#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped, TwistStamped
from styx_msgs.msg import Lane, Waypoint
from std_msgs.msg import Int32

import math

'''
This node will publish waypoints from the car's current position to some `x` distance ahead.

As mentioned in the doc, you should ideally first implement a version which does not care
about traffic lights or obstacles.

Once you have created dbw_node, you will update this node to use the status of traffic lights too.

Please note that our simulator also provides the exact location of traffic lights and their
current status in `/vehicle/traffic_lights` message. You can use this message to build this node
as well as to verify your TL classifier.

TODO (for Yousuf and Aaron): Stopline location for each traffic light.
'''

ONE_MPH = 0.44704

LOOKAHEAD_WPS = 200  # Number of waypoints we will publish. You can change this number
TARGET_VELOCITY_MPH = 40 # Target velocity in MPH. You can change this number

class WaypointUpdater(object):
    def __init__(self):
        self.waypoints = None
	self.tf_waypoint_id = -1

        rospy.logdebug("WaypointUpdater started")
        rospy.init_node('waypoint_updater', log_level=rospy.DEBUG)

        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/current_velocity', TwistStamped, self.velocity_cb)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)
        rospy.Subscriber('/traffic_waypoint', Int32, self.traffic_cb)
        #rospy.Subscriber('/obstacle_waypoint', Int32, self.obstacle_cb)

        # TODO: Add a subscriber for /traffic_waypoint and /obstacle_waypoint below

        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)

        # TODO: Add other member variables you need below

        rospy.spin()

    def pose_cb(self, msg):
        rospy.logdebug("pose_cb fired. X: %s, Y: %s, Z: %s", msg.pose.position.x, msg.pose.position.y, msg.pose.position.z)
        if self.waypoints is not None: self.find_closest_publish_lane(msg)
        pass

    def velocity_cb(self, msg):
        self.current_velocity = msg.twist.linear.x / ONE_MPH 
        rospy.logdebug("velocity_cb fired. v: %s", self.current_velocity)
        pass

    def waypoints_cb(self, waypoints):
        rospy.logdebug("waypoints_cb fired")
        self.waypoints = waypoints
        pass

    def traffic_cb(self, msg):
        if self.waypoints is not None: self.tf_waypoint_id = msg.data
        rospy.logdebug("traffic_cb fired")
        pass

    def obstacle_cb(self, msg):
        rospy.logdebug("obstacle_cb fired")
        pass

    def get_waypoint_velocity(self, waypoint):
        return waypoint.twist.twist.linear.x

    def set_waypoint_velocity(self, waypoints, waypoint, velocity):
        waypoints[waypoint].twist.twist.linear.x = velocity

    def distance(self, waypoints, wp1, wp2):
        dist = 0
        dl = lambda a, b: math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2)
        for i in range(wp1, wp2 + 1):
            dist += dl(waypoints[wp1].pose.pose.position, waypoints[i].pose.pose.position)
            wp1 = i
        return dist

    def find_closest_publish_lane(self, pose):
        waypoints = self.waypoints.waypoints

        dl = lambda a, b: math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2)
        i_min = -1
        dist_min = float('inf')
        for i, wp in enumerate(waypoints):
            dist = dl(pose.pose.position, wp.pose.pose.position)
            if dist < dist_min:
                dist_min = dist
                i_min = i

        if i_min < 0:
            rospy.logwarn('No waypoints ahead')
            return

        rest_wp = waypoints[i_min: min(i_min + LOOKAHEAD_WPS, len(waypoints))]
        while len(rest_wp) < LOOKAHEAD_WPS:
            rest_wp += waypoints[:min(LOOKAHEAD_WPS-len(rest_wp), len(waypoints))]

        for wp in rest_wp:
            wp.twist.twist.linear.x = TARGET_VELOCITY_MPH * ONE_MPH

        #Stopping at red traffic light
        if (self.tf_waypoint_id > 0):
            distance_to_tl_in_wps = self.tf_waypoint_id - i_min
            v0 = self.current_velocity
            d0 = dl(pose.pose.position, waypoints[self.tf_waypoint_id].pose.pose.position)
            rospy.loginfo('distance_to_tl_in_m %s', d0)
            if (d0 > 0 and d0 <= 5):
                for wp in rest_wp:
                    wp.twist.twist.linear.x = 0
            if (d0 > 5 and d0 <= 35):
                for wp in rest_wp:
                    wp.twist.twist.linear.x = 3 * ONE_MPH
            if (d0 > 35 and d0 < 70):
                for i in range(len(rest_wp)):
                    d = dl(rest_wp[i].pose.pose.position, waypoints[self.tf_waypoint_id].pose.pose.position)
                    desired_speed_sqrt = v0 ** 2 - (v0 **2 - 6 ** 2) * (d0 - d)/(d0 - 35)
                    if (desired_speed_sqrt > 0):		    	
                        rest_wp[i].twist.twist.linear.x = math.sqrt(desired_speed_sqrt) * ONE_MPH
                    else:
                        rest_wp[i].twist.twist.linear.x = 6 * ONE_MPH  



        lane = Lane()
        lane.header = self.waypoints.header
        lane.waypoints = rest_wp 

        self.final_waypoints_pub.publish(lane)
        rospy.loginfo('lane nearest wp dist: %s, i: %s, X: %s, Y: %s', dist_min, i_min, wp.pose.pose.position.x, wp.pose.pose.position.y)
        rospy.loginfo('nearest wp %s, traffic_light_wp: %s', i_min, self.tf_waypoint_id)


if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
