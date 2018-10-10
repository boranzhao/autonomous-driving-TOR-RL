import numpy as np
import random
from gym.spaces import MultiDiscrete, Dict, Discrete, Box
import gym 

from collections import namedtuple

from enum import Enum
class CarDrivingMode(Enum):
    AUTONOMOUS = 1
    DRIVER_INTEVENTION = 2

class DriverAction(Enum):
    NONE = 0
    BRAKE = 1
    ACCEL = 2
    STEER= 3

class DriverMode(Enum):    
    STOP = 0
    DRIVE_TO_SPEED = 1
    PERCEIVE = 2
    BE_DISTRACTED = 3  # Meaning that the car is in autonomous driving mode
    FOLLOW_WITH_SAFE_DISTANCE = 4
    SWERVE_TO_LEFT = 5
    CHANGE_TO_RIGHT = 6
    EMERGENCY_BRAKE = 7

class Lane(Enum):
    RIGHT= 0
    MIDDLE = 1
    LEFT = 2
class UrgencyDegree(Enum):
    MILD = 1
    URGENT = 2
    EXTREMELY_URGENT= 3

"""
Define the environment of the warning system 
state:  {time_to_collision, confidence_level}
"""

NOT_WARN = 0
WARN = 1


IGNORE = 0
ACKNOWLEDGE = 1

VALID_DRIVER_INTENTIONS = [IGNORE, ACKNOWLEDGE]

STAND_GRAVITY = 9.8
MIN_ACCELERATION_LONG = -1*STAND_GRAVITY   # Maximum longitudinal deceleration during braking a car
MAX_ACCELERATION_LONG = 0.9*STAND_GRAVITY # Maximum longitudinal acceleration: 0.9g
MAX_ACCELERATION_LAT = STAND_GRAVITY      # Maximum later acceleration

LANE_WIDTH = 3.7                # the width of a lane on the freeway

epislon_speed = 0.5             # tolerance for evaluating whether a target speed is reached
epislon_distance = 1            # tolerance for evalu

class State():
    def __init__(self,time_to_collision =float('inf'), confidence_level = 1):
        self.time_to_collision = time_to_collision
        self.confidence_level = confidence_level

class Barrier():
    def __init__(self,position,speed=0,length=120):
        self.position = position
        self.length = length 
        self.speed = speed                 # Some barriers such as a cleaning vehicle may have non-zero speed
        self.lane = Lane.MIDDLE                  
    def update_status(self,sample_time):
        self.position += self.speed/3.6*sample_time         # /3.6 is for converting from km/h to m/s

class DrivingEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self,car,driver,num_boundaries=5,speed = 0, target_speed =100, sample_time = 0.25,discount_factor=0.9,always_penalize_warning = True):
        self.car = car
        self.driver = driver
        self.action_space = [NOT_WARN,WARN]
        self.observation_space = Dict({'time_to_collision':Box(low= 0, high= float('inf'), shape=(1,)),\
                                 'confidence_level':Box(low= 0, high= 1.0, shape=(1,))})
        self.sample_time = sample_time
        
        # Create the barriers
        self.barriers = []
        for i in range(num_boundaries):
            barrier = Barrier((i+1)*1000)
            self.barriers.append(barrier)
        
        self.relative_distance = max(self.barriers[0].position - self.car.position,0)                     # unit: m
        self.relative_velocity = self.car.speed - self.barriers[0].speed                           # unit: km/h
        self.index_boundary_ahead = 0

        self.state =State(time_to_collision =float('inf'), confidence_level=1)

        self.min_safe_ttc = 0.5                                   # minimum safe ttc; driver will not have enough time to respond if ttc is smaller than this value

        self.num_near_crashes = 0                                 # number of near crashes
        self.num_crashes = 0                                      # number of crashes
        
        # Boolean variables to indicate whether crash or near crash happened 
        self.crash = False
        self.near_crash = False
       
        self.reward_range = (-float('inf'),float('inf'))
        self.warning_action = NOT_WARN

        self.discount_factor = discount_factor                  # for discounting the reward in accumulating the reward in driver intervening
        self.accumulated_reward = 0
        self.driver_emergency_action = False

        self.always_penalize_warning = always_penalize_warning  # set this to True will help reduce the false positive warnings

        # for rendering 
        self.viewer = None
        self.MIN_POSITION = 0 
        self.MAX_POSITION = 500

    def reset(self):
        self.car.reset()
        self.index_boundary_ahead = 0
        self.update_state()

        self.num_near_crashes = 0                                 # number of near crashes
        self.num_crashes = 0     
        self.crash = False
        self.near_crash = False 

        self.driver_emergency_action = False
        self.driver.reset()

        self.driver.driver_mode = DriverMode.DRIVE_TO_SPEED
        self.car.driving_mode = CarDrivingMode.DRIVER_INTEVENTION

        return np.array([self.state.time_to_collision,self.state.confidence_level])

    def step(self,action):
        """
        This function simulates executing one action (WARN or NOT_WARN) to the environment
        """
        self.warning_action = action 
        reward = 0

        # Take-over request (i.e. the warning) should only be issued when driver is distracted, 
        # i.e. the car is in autonomous driving mode
        if self.driver.driver_mode == DriverMode.BE_DISTRACTED and action == WARN:
            self.driver.decide_whether_to_acknowledge()

        # Update driver status
        self.driver.update_status(self.car, self.state.time_to_collision, self.relative_distance,self.relative_velocity,
                                 self.warning_action,self.sample_time)

        # Update car status
        self.car.update_status(self.driver,self.sample_time)

        # Update barrier status
        for barrier in self.barriers:
            barrier.update_status(self.sample_time)

        # Update ther index of the barrier ahead
        if self.car.position > self.barriers[self.index_boundary_ahead].position + self.barriers[self.index_boundary_ahead].length \
            and self.car.lane != self.barriers[self.index_boundary_ahead].lane:
            self.index_boundary_ahead += 1            

        # Update the state
        self.state.time_to_collision, self.state.confidence_level = self.update_state()        

        # A near-crash happens when the longitudinal deceleration of the car is larger than 0.5g, or the lateral acceleration is larger than 0.4g,
        # or time_to_collision is smaller than 2 seconds at any time
        if (self.car.acceleration_longitudinal <= -0.5*STAND_GRAVITY or self.car.acceleration_lateral>=0.3*STAND_GRAVITY 
            or self.state.time_to_collision< self.min_safe_ttc):
            if not self.near_crash:
                self.num_near_crashes += 1
            self.near_crash = True
        else:
            self.near_crash = False

        # A crash happens when the relative distance is smaller than a threshold value 
        if self.relative_distance<=0:
            if not self.crash:
                self.num_crashes +=1         
            self.crash = True  
        else:
            self.crash = False     
        
        # Update the reward based on current state 
        if self.crash:
            reward = -10            # get a large penalty for a crash 
        elif self.near_crash:
            reward = -1             # get a penalty for a near-crash
        
        done = False

        if self.driver.driver_mode in [DriverMode.SWERVE_TO_LEFT, DriverMode.EMERGENCY_BRAKE]:   
            self.driver_emergency_action = True 
        else:
            if self.driver_emergency_action:
                # Emergency action ends, so an episode terminates
                done = True
                if not self.crash:
                    reward = 10      # get a reward after successfully circumventing a barrier 
            self.driver_emergency_action = False

        
        if self.crash or (self.car.position >= self.barriers[-1].position+self.barriers[-1].length):
            game_over = True
            done = True       # An episode is also done if there is a crash regardless of whether driver intervenes or not
        else:
            game_over = False      

        if self.always_penalize_warning and action== WARN:
            reward -= 0.5

        # print(self.relative_distance)
        return np.array([self.state.time_to_collision,self.state.confidence_level]), reward, done, game_over

    def update_state(self):
        # Update relative distance, relative speed and time to collision
        if self.index_boundary_ahead>= len(self.barriers) or self.car.lane != self.barriers[self.index_boundary_ahead].lane:
            self.relative_distance = float('inf')
            self.relative_velocity = self.car.speed
            time_to_collision = float('inf')
        else:
            self.relative_distance = self.barriers[self.index_boundary_ahead].position - self.car.position
            self.relative_velocity = self.car.speed- self.barriers[self.index_boundary_ahead].speed
            if self.relative_velocity <= 0 and self.relative_distance >0:
                time_to_collision = float('inf')
            else:
                time_to_collision = self.relative_distance/(self.relative_velocity/3.6+1e-6)   # 1e-6 is added to avoid division by zero
            

        ## TO DO: implement a realistic way to calculate the confidence level
        confidence_level = 1

        return time_to_collision, confidence_level

    def render(self, mode = "human"):
        screen_width = 600
        screen_height = 600

        scale_height =screen_height/(self.MAX_POSITION-self.MIN_POSITION)  # meter 
        scale_width = 100/(LANE_WIDTH*3)

        lane_left_boundary = 400
        lane_width = LANE_WIDTH*scale_width

        middle_lane_center = lane_left_boundary+1.5*lane_width

        carwidth= lane_width/2
        carlength= lane_width

        if self.viewer is None:
            barrier_width = lane_width

            car_track_shift = 150    

            from lib import rendering
            import pyglet
            self.viewer = rendering.Viewer(screen_width, screen_height)
            self.previous_track_pos = 0
            self.previous_car_pos = 0
            self.tracktrans = rendering.Transform()

            # Add the lane marks
            for i in range(4):
                track = rendering.make_line((lane_left_boundary+i*lane_width,0),(lane_left_boundary+i*lane_width,(len(self.barriers)+1)*1000*scale_height))
                track.set_linewidth(2)
                if i==1 or i==2:
                    track.set_linestyle(15,0x1111)
                track.add_attr(rendering.Transform(translation=(0, 0)))     
                track.add_attr(self.tracktrans)
                self.viewer.add_geom(track)

            # Add the barriers 
            for barrier in self.barriers:
                l,r,t,b = -barrier_width/2, barrier_width/2, (barrier.position+barrier.length)*scale_height, barrier.position*scale_height           
                barrier = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
                barrier.set_color(1,0,0)
                barrier.add_attr(rendering.Transform(translation=(middle_lane_center, car_track_shift)))            
                barrier.add_attr(self.tracktrans)
                self.viewer.add_geom(barrier)
            
            # Add a line at the goal position
            end_line = rendering.make_line((lane_left_boundary,(self.barriers[-1].position+self.barriers[-1].length)*scale_height+car_track_shift),
                                (lane_left_boundary+3*lane_width,(self.barriers[-1].position+self.barriers[-1].length)*scale_height+car_track_shift))
            end_line.set_linewidth(5)
            end_line.set_linestyle(3,0x3333)
            end_line.set_color(0,1,0)   
            end_line.add_attr(self.tracktrans)
            self.viewer.add_geom(end_line)

            
            # add skew lines (not in use)
            # interval = 10 
            # xs = np.array(range(interval, int((self.MAX_POSITION*2-self.MIN_POSITION)*scale+interval),interval))
            # xs1 = xs-interval            
            # for startx,endx in np.nditer([xs,xs1]):
            #     line = rendering.make_line((startx,roadheight),(endx,roadheight-10))
            #     line.set_linewidth(2)
            #     line.add_attr(self.tracktrans)
            #     self.viewer.add_geom(line)

            top_y = 400
            # Add a label and a circle for denoting whether warning issued
            warning_label = pyglet.text.Label('Warning', font_size=13,
                x=30, y=top_y, anchor_x='left', anchor_y='bottom',
                # color=(128,128,128,255))
                color=(0,0,0,255))
            self.viewer.add_label(warning_label)
            self.circle_warning = rendering.make_circle_at_pos(pos=(120,top_y+10),radius=8)
            self.circle_warning.set_color(.5,.5,.5)
            self.viewer.add_geom(self.circle_warning)
            
            # Add a label and a circle for denoting whether the driver acknowledges a warning
            driver_acknowledge_label = pyglet.text.Label('Driver Acknowledge', font_size=13,
                x=180, y=top_y, anchor_x='left', anchor_y='bottom',
                # color=(128,128,128,255))
                color=(0,0,0,255))
            self.viewer.add_label(driver_acknowledge_label)
            self.circle_driver_acknowledge = rendering.make_circle_at_pos(pos=(360,top_y+10),radius=8)
            self.circle_driver_acknowledge.set_color(.5,.5,.5)
            self.viewer.add_geom(self.circle_driver_acknowledge)

            # Add a label for total alarms
            self.label_total_alarms = pyglet.text.Label('#Total Alarms: '+ '0', font_size=13,
                x=30, y=top_y-40, anchor_x='left', anchor_y='bottom',
                # color=(128,128,128,255))
                color=(0,0,0,255))
            self.viewer.add_label(self.label_total_alarms)

            # Add a label for false positive alarms
            self.label_FP = pyglet.text.Label('#FP: '+ '0', font_size=13,
                x=200, y=top_y-40,anchor_x='left', anchor_y='bottom',
                # color=(128,128,128,255))
                color=(0,0,0,255))
            self.viewer.add_label(self.label_FP)

            # Add a label for false negative alarms
            self.label_FN = pyglet.text.Label('#FN: '+ '0', font_size=13,
                x=300, y=top_y-40, anchor_x='left', anchor_y='bottom',
                # color=(128,128,128,255))
                color=(0,0,0,255))
            self.viewer.add_label(self.label_FN)

           
            # Add a label for the number of near crashes
            self.label_num_near_crashes = pyglet.text.Label('#Near Crashes: '+ '0', font_size=13,
                x= 30, y=top_y-80, anchor_x='left', anchor_y='bottom',
                # color=(128,128,128,255))
                color=(0,0,0,255))
            self.viewer.add_label(self.label_num_near_crashes)

            # Add a label for the number of crashes
            self.label_num_crashes = pyglet.text.Label('#Crashes: '+ '0', font_size=13,
                x= 200, y=top_y-80, anchor_x='left', anchor_y='bottom',
                # color=(128,128,128,255))
                color=(0,0,0,255))
            self.viewer.add_label(self.label_num_crashes)

            # Add a label for driver mode 
            self.label_driver_mode = pyglet.text.Label('Driver Mode: '+ 'Stop', font_size=13,
                x= 30, y=top_y-120, anchor_x='left', anchor_y='bottom',
                # color=(128,128,128,255))
                color=(0,0,0,255))
            self.viewer.add_label(self.label_driver_mode)

            # Add a label for driver's trust 
            self.label_driver_trust = pyglet.text.Label('Driver\'s Trust: '+ '1.00', font_size=13,
                x= 30, y=top_y-160, anchor_x='left', anchor_y='bottom',
                # color=(128,128,128,255))
                color=(0,0,0,255))
            self.viewer.add_label(self.label_driver_trust)
            

            # Add the car
            l,r,t,b = -carwidth/2, carwidth/2, 0, -carlength           
            car = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            car.set_color(0,1,0)
            car.add_attr(rendering.Transform(translation=(middle_lane_center, car_track_shift)))            
            self.cartrans = rendering.Transform()
            car.add_attr(self.cartrans)
            self.viewer.add_geom(car)
            
            # Add a line in front of the car to denote the start of a new episode
            start_line = rendering.make_line((lane_left_boundary,car_track_shift+carlength/2+20),(lane_left_boundary+3*lane_width,car_track_shift+carlength/2+20))
            start_line.set_linewidth(5)
            start_line.set_linestyle(3,0x3333)
            start_line.set_color(0,0,1)
            # track.add_attr(rendering.Transform(translation=(0, 0)))     
            start_line.add_attr(self.tracktrans)
            self.viewer.add_geom(start_line)

        # Note that the direction of x axis in the window  is the reverse direction of the x axis in defining the lateral position
        self.cartrans.set_translation(-self.car.position_lateral*scale_width, 0)
        
        # Move the track in the reversive direction so that the car looks moving forward
        track_pos = self.previous_track_pos + (self.previous_car_pos -self.car.position)*scale_height
        self.tracktrans.set_translation(0,track_pos)
        self.previous_car_pos = self.car.position
        self.previous_track_pos =  track_pos

        # Visualize the current status
        if self.warning_action == NOT_WARN:
            self.circle_warning.set_color(.5,.5,.5)
        else:
            self.circle_warning.set_color(1,0,0)

        if self.driver.warning_acknowledged:
            self.circle_driver_acknowledge.set_color(1,0,0)
        else:
            self.circle_driver_acknowledge.set_color(.5,.5,.5)

        self.label_FP.text = '#FP: '+ str(self.driver.false_positive_warnings)
        self.label_FN.text = '#FN: '+ str(self.driver.false_negative_warnings)
        self.label_total_alarms.text = '#Total Alarms: '+ str(self.driver.total_warnings)
        self.label_driver_mode.text ='Driver Mode: '+ self.driver.driver_mode.name
        self.label_num_near_crashes.text = '#Near Crashes: '+ str(self.num_near_crashes)
        self.label_num_crashes.text = '#Crashes: '+ str(self.num_crashes)
        self.label_driver_trust.text = "Driver\'s Trust: {:.2f}".format(self.driver.probability_to_acknowledge)  #'Driver\'s Trust: '+ str(self.driver.probability_to_acknowledge)
        return self.viewer.render(return_rgb_array = mode=='rgb_array')

class Driver():
    """

    """
    def __init__(self,
                 brake_intensity =0.5, 
                 accel_intensity = 0, 
                 response_time_bounds = [1.5,2.5], 
                 maximum_intervention_ttc = 8,
                 comfort_follow_distance = 50,
                 driver_mode = DriverMode.STOP):
        
        ## TO DO: determine the brake intensity and accel intensity more reasonably 
        self.false_positive_warnings = 0
        self.false_negative_warnings = 0

        self.total_warnings = 0
        self.probability_to_acknowledge = 1

        self.brake_intensity = brake_intensity
        self.accel_intensity = accel_intensity
        self.steer_intensity = 0

        self.response_time_bounds = response_time_bounds           
        self.maximum_intervention_ttc = maximum_intervention_ttc  # Driver will not intervene if real ttc is larger than this value

        self.warning_acknowledged = False
        self.time_to_decision = float('inf')                      # Time before making a decision for the driver after acknowledging a warning signal 
        
        self.comfort_follow_distance = comfort_follow_distance
        self.driver_mode = driver_mode
        
        self.urgency_degree = UrgencyDegree.MILD
        self.target_speed = 120

    def reset(self):
        self.false_positive_warnings = 0
        self.false_negative_warnings = 0

        self.total_warnings = 0

        self.warning_acknowledged = False
        self.time_to_decision = float('inf')
        self.steer_intensity = 0
        self.driver_mode = DriverMode.STOP

    def drive_to_speed(self,car,sample_time,target_speed=120):   
        car.target_speed = None         
        self.target_speed = target_speed
        
        # calculate the brake and acceleration intensities to reach the target_speed
        self._speed_control(car.speed,target_speed)
        if abs(car.speed-target_speed) <= epislon_speed:
            if car.lane == Lane.MIDDLE:
                self.activate_autonomous_mode(car)   
            elif car.lane == Lane.LEFT:
                self.driver_mode = DriverMode.CHANGE_TO_RIGHT 
            
    def be_distracted(self,warning_action):
        if warning_action == WARN:
            self.decide_whether_to_acknowledge()
        if self.warning_acknowledged:
            self.driver_mode = DriverMode.PERCEIVE

    def perceive_and_decide(self,time_to_collision,sample_time):
        """
            Perceive the surroundings and decide whether to intervene after acknowledging a TOR
        """  
        self.time_to_decision -= sample_time
        self.time_to_decision = max(self.time_to_decision,0)

        if self.time_to_decision <=0:
            if time_to_collision > self.maximum_intervention_ttc:
                # false positive alarms
                self.false_positive_warnings += 1
                self.driver_mode = DriverMode.BE_DISTRACTED  
                # Reset the warning_acknoweledge flag so that the driver can go ahead to decide whether to acknowledge next warning signal
                self.warning_acknowledged = False
            else: 
                if time_to_collision >= 4:
                    self.urgency_degree = UrgencyDegree.MILD
                    self.driver_mode = DriverMode.SWERVE_TO_LEFT 
                elif time_to_collision >= 2.5:
                    self.urgency_degree = UrgencyDegree.URGENT
                    self.driver_mode = np.random.choice([DriverMode.SWERVE_TO_LEFT,DriverMode.EMERGENCY_BRAKE])
                else:
                    self.urgency_degree = UrgencyDegree.EXTREMELY_URGENT
                    self.driver_mode = DriverMode.EMERGENCY_BRAKE

    def follow_with_safe_distance(self,car,relative_distance,relative_velocity):
        # calculate the brake and acceleration intensities to reach the target distance
        self.brake_intensity, self.accel_intensity = self._distance_control(relative_distance, relative_velocity,self.comfort_follow_distance)
        if car.speed >= self.target_speed and relative_distance > self.comfort_follow_distance:
            self.driver_mode = DriverMode.DRIVE_TO_SPEED

    def swerve_to_left(self,car):        
        if abs(car.speed_lateral)<1e-1 and abs(car.position_lateral-LANE_WIDTH)<1e-1:
            self.warning_acknowledged = False
            self.urgency_degree = UrgencyDegree.MILD            
            self.driver_mode = DriverMode.DRIVE_TO_SPEED
            self.steer_intensity = 0
        else:
            self.steer_intensity = self._steer_control(car, target_lateral_position=LANE_WIDTH)
        
    def change_to_right(self,car):
        if abs(car.speed_lateral)<1e-1 and abs(car.position_lateral)<1e-1:
            self.driver_mode = DriverMode.DRIVE_TO_SPEED
            self.steer_intensity = 0
            # reset car's later speed and acceleration to 0
            car.speed_lateral = 0
            car.acceleration_lateral = 0
        else:   
            self.steer_intensity = self._steer_control(car,target_lateral_position=0)

    def emergy_brake(self,car):
        """        
        """
        if car.speed<=0:
            # If no collision happens after a full stop, change to the middle lane and continue
            self.accel_intensity = 0.1
            self.brake_intensity = 0
            self.urgency_degree = UrgencyDegree.MILD
            self.driver_mode = DriverMode.SWERVE_TO_LEFT
            self.swerve_to_left(car)
        else: 
            if self.urgency_degree == UrgencyDegree.URGENT:
                self.brake_intensity = -0.8*STAND_GRAVITY/MIN_ACCELERATION_LONG
            elif self.urgency_degree == UrgencyDegree.EXTREMELY_URGENT:
                self.brake_intensity = 1

    def _steer_control(self,car,target_lateral_position = LANE_WIDTH,Kp= 0.5, Kd=0.5):
        """
        Use a PD controller to control the lateral acceleration
        """
        # calculate the desired lateral acceleration
        acceleration_lateral = ((target_lateral_position-car.position_lateral)*Kp-car.speed_lateral*Kd)*self.urgency_degree.value 
        acceleration_lateral = np.clip(acceleration_lateral,-MAX_ACCELERATION_LAT,MAX_ACCELERATION_LAT)

        steer_intensity = acceleration_lateral/MAX_ACCELERATION_LAT
        return steer_intensity
                
    def _brake_to_stop(self,car):
        self.accel_intensity = 0
        self.brake_intensity = np.clip(0.2 + car.speed*0.1,0,1)

        return True if car.speed <=0 else False

    def get_action(self):
        """
        Get the action of the driver based on the accel_intensity and brake_intensity 
        """
        if self.accel_intensity !=0:
            action = DriverAction.ACCEL
        elif self.brake_intensity !=0:
            action = DriverAction.BRAKE
        elif self.steer_intensity !=0:
            action = DriverAction.STEER
        else:
            action = DriverAction.NONE
        return action
        
    def _sample_response_time(self):
        response_time = (self.response_time_bounds[1]-self.response_time_bounds[0])*np.random.sample()+ self.response_time_bounds[0] 
        return response_time

    def _speed_control(self,speed, target_speed, Kp = 0.03):
        """
        To maintain a target_speed with a proportional controller 
        """
        if speed<target_speed-epislon_speed:
            self.brake_intensity = 0
            self.accel_intensity = np.clip(Kp*(target_speed-speed),0,1)
        elif speed>target_speed+epislon_speed:
            self.accel_intensity = 0
            self.brake_intensity = np.clip(Kp*(speed-target_speed),0,1)

    def _distance_control(self, relative_distance, relative_velocity, target_distance, Kp = 0.1,Kd=0.2):
        """
        To maintain a target relative_distance with a proportional-derivative controller 
        """
       
        # calculate the desired acceleration
        acceleration = (relative_distance-target_distance)*Kp-relative_velocity*Kd 
        acceleration = np.clip(acceleration,MIN_ACCELERATION_LONG,MAX_ACCELERATION_LONG)

        # calculate the acceleration and brake intensities based the  desired acceleration
        if acceleration > 0:
            brake_intensity = 0
            accel_intensity = acceleration/MAX_ACCELERATION_LONG
        elif acceleration < 0:
            brake_intensity = acceleration/MIN_ACCELERATION_LONG
            accel_intensity = 0
        else:
            brake_intensity = 0
            accel_intensity = 0
        
        return brake_intensity,accel_intensity

    def activate_autonomous_mode(self,car):
        self.brake_intensity = 0
        self.accel_intensity = 0
        self.steer_intensity = 0

        car.driving_mode = CarDrivingMode.AUTONOMOUS
        car.target_speed = car.speed       
        self.driver_mode = DriverMode.BE_DISTRACTED

    def stop_the_car(self,car):
        self.driver_mode = DriverMode.STOP
        self._brake_to_stop(car)

    def update_probability_to_acknowledge(self):
        self.probability_to_acknowledge = 1 - (self.false_positive_warnings + self.false_negative_warnings) \
                                      /(self.total_warnings + self.false_negative_warnings+1)
    
    def update_status(self,car,time_to_collision,relative_distance,relative_velocity,warning_action,sample_time):
        """         
        """
        self.update_probability_to_acknowledge()

        if self.driver_mode == DriverMode.STOP:
            self._brake_to_stop(car)
        elif self.driver_mode == DriverMode.DRIVE_TO_SPEED:
            self.drive_to_speed(car,sample_time)
        elif self.driver_mode == DriverMode.BE_DISTRACTED:
            self.be_distracted(warning_action)  ## TO DO: think how to pass warn
        elif self.driver_mode == DriverMode.PERCEIVE:
            self.perceive_and_decide(time_to_collision,sample_time)
        elif self.driver_mode == DriverMode.SWERVE_TO_LEFT:
            self.swerve_to_left(car)
        elif self.driver_mode == DriverMode.EMERGENCY_BRAKE:
            self.emergy_brake(car)
        elif self.driver_mode == DriverMode.CHANGE_TO_RIGHT:
            self.change_to_right(car)
        elif self.driver_mode == DriverMode.FOLLOW_WITH_SAFE_DISTANCE:
            self.follow_with_safe_distance(car,relative_distance,relative_velocity)

        if self.brake_intensity >0:
            self.action = DriverAction.BRAKE
        elif self.accel_intensity >0:
            self.action = DriverAction.ACCEL
        elif self.steer_intensity >0:
            self.action = DriverAction.STEER
        else:
            self.action = DriverAction.NONE
        
    def decide_whether_to_acknowledge(self):        
        # Evaluate whether the driver will acknowledge an warning 
        # The driver can only acknowledge at most one warning at a time 
        if self.warning_acknowledged == False:
            self.total_warnings += 1
            self.warning_acknowledged = np.random.choice(VALID_DRIVER_INTENTIONS, p =[1-self.probability_to_acknowledge,self.probability_to_acknowledge])
            if self.warning_acknowledged == True:
                # Acknowledge a warning signal and reset the time it takes before the driver makes decision
                self.time_to_decision = self._sample_response_time()

class Car():
    def __init__(self, speed =30, accel_intensity = 0, brake_intensity = 0, driving_mode = CarDrivingMode.DRIVER_INTEVENTION ):
        
        self.accel_intensity = accel_intensity
        self.brake_intensity = brake_intensity

        self.speed = speed 
        self.speed_lateral = 0

        self.acceleration_longitudinal = 0
        self.acceleration_lateral = 0

        self.position = 0
        self.position_lateral = 0

        self.target_speed = speed            # target speed under autonomous driving mode 
        if driving_mode == CarDrivingMode.DRIVER_INTEVENTION:
            self.target_speed = None
        else:
            self.target_speed = speed
        
        self.driving_mode = driving_mode
        self.MAX_SPEED = 120
        
        self.lane = Lane.MIDDLE

    def reset(self):
        self.position = 0
        self.position_lateral =0

        self.speed = 0   
        self.speed_lateral = 0

        self.acceleration_longitudinal = 0
        self.acceleration_lateral = 0

        self.lane = Lane.MIDDLE 

        self.accel_intensity = 0
        self.brake_intensity = 0.2

    def update_control_intensity(self,driver):
        """
        Update the brake and acceleration intensities. A constant value is used for the autonomous mode
        """
        if  self.driving_mode == CarDrivingMode.DRIVER_INTEVENTION:
            self.brake_intensity = driver.brake_intensity
            self.accel_intensity = driver.accel_intensity
            self.steer_intensity = driver.steer_intensity
        elif self.driving_mode == CarDrivingMode.AUTONOMOUS:
            if self.target_speed > self.speed+epislon_speed:
                self.accel_intensity = 0.5
                self.brake_intensity = 0
            elif self.target_speed < self.speed - epislon_speed:
                self.accel_intensity =0
                self.brake_intensity =0.5
            else:
                self.accel_intensity = 0
                self.brake_intensity = 0
    
    def set_target_speed(self,target_speed):
        self.target_speed = target_speed
    
    def update_position(self,sample_time):
        self.position += self.speed/3.6*sample_time         # /3.6 is for converting from km/h to m/s
        self.position_lateral += self.speed_lateral/3.6*sample_time

    def update_speed(self,sample_time):        
        self.speed += self.acceleration_longitudinal*sample_time*3.6     # *3.6 is for converting from m/s to km/h
        self.speed = np.clip(self.speed,0,self.MAX_SPEED)

        self.speed_lateral += self.acceleration_lateral*sample_time*3.6
    
    def update_acceleration(self,driver):
        self.update_control_intensity(driver)
        self.acceleration_longitudinal = self.accel_intensity*MAX_ACCELERATION_LONG + self.brake_intensity*MIN_ACCELERATION_LONG
        self.acceleration_lateral = self.steer_intensity*MAX_ACCELERATION_LAT

    def update_status(self,driver,sample_time):
        self.update_acceleration(driver)

        self.update_position(sample_time)

        if self.position_lateral > 0.5*LANE_WIDTH:
            self.lane = Lane.LEFT
        elif self.position_lateral < -0.5*LANE_WIDTH:
            self.lane = Lane.RIGHT
        else:
            self.lane = Lane.MIDDLE         

        self.update_speed(sample_time)        

        if driver.get_action() != DriverAction.NONE:
            self.driving_mode = CarDrivingMode.DRIVER_INTEVENTION