
import rospy
from pid import PID
from yaw_controller import YawController
from lowpass import LowPassFilter


GAS_DENSITY = 2.858
ONE_MPH = 0.44704

MIN_LINEAR_VEL = 0.0
MAX_LINEAR_VEL = 1.0

KP = 1.0
KI = 0.01
KD = 0.1

TAU = 0.1
TS = 0.1

class Controller(object):
    def __init__(self, vp):
        # TODO: Implement
        self.throttle_pid = PID(KP, KI, KD,
                                mn=MIN_LINEAR_VEL,
                                mx=MAX_LINEAR_VEL)
        self.steering_filter = LowPassFilter(TAU,TS)        
        self.angular_controller = YawController(vp.wheel_base, vp.steer_ratio, MIN_LINEAR_VEL, 
                                                vp.max_lat_accel, vp.max_steer_angle)
 
        # Initiate Time
        self.last_time = -1.0

    def control(self, proposed_linear_velocity, proposed_angular_velocity, 
                current_linear_velocity, dbw_enabled):
        # TODO: Change the arg, kwarg list to suit your needs
        # Return throttle, brake, steer


        throttle = 0.0
        brake = 0.0
        steering =0.0
        
        time_now = rospy.get_time() 
        
        if self.last_time < 0.0:
            # first time we do nothing
            self.last_time = self.last_time
        else: 
            steering = self.steering_filter.filt(self.angular_controller.get_steering(proposed_linear_velocity,
                                                                                      proposed_angular_velocity,
                                                                                      current_linear_velocity))

            # Only update pid controller if Drive By Wire is enabled
            if dbw_enabled:
                throttle = self.throttle_pid.step(proposed_linear_velocity - current_linear_velocity, time_now - self.last_time )
            else:
                self.throttle_pid.reset()
        
        self.last_time = time_now

        if throttle < 0:
        	throttle = 0
        	brake  = -throttle

        # Return throttle, brake, steering
        return throttle, brake, steering        
