from rasrobot import RASRobot

import numpy as np
import time
import math
import torch
import cv2
import random

from collections import deque


class MyRobot(RASRobot):
    def __init__(self):

        #The constructor has no parameters.

        super(MyRobot, self).__init__()

        # Initialise and resize a new window 
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        #self.yolov5_model = yolov5.load('yolov5s.pt') # Load YOLOv5s model
        self.fl=0
        self.steering_angleee=0
        cv2.namedWindow("output", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("output", 256*2, 128*2)
        cv2.namedWindow("output", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("output", 256*2, 128*2)
        self.random_turn = None
        self.queue = deque(maxlen=15)
        
    #The robot's camera is used to capture an image, and road-related pixels are removed. The cleaned-up filtered image is then subjected to a morphological procedure (opening).
    def get_road(self):
        
        #This method relies on the `get_camera_image` method from RASRobot. It takes the image, converts
        #it into HSV (Hue, Saturation, Value) and filters pixels that belong to the three following
        #ranges:
        # Hue: 0-200 (you may want to tweak this)
        # Saturation: 0-50    (you may want to tweak this)
        # Value: 0-100    (you may want to tweak this)
        
        image = self.get_camera_image()
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower = np.array([0, 0, 0], np.uint8)
        upper = np.array([200, 50, 100], np.uint8)
        mask = cv2.inRange(hsv, lower, upper).astype(bool)
        result = np.zeros(image.shape)
        result[mask] = 255
        return cv2.morphologyEx(result, cv2.MORPH_OPEN, (3,3))
                  
    #By using the location of the yellow line in the edge image, one may calculate the steering angle. The image is reduced to its bottom-quarter, and the departure from the image's centre is calculated. It also locates the centre of the white pixels (which stand in for the yellow line).    
    def yellowline(self, edges):
        
        #Follow the yellow line by turning left or right depending on where the line is in the image.
        
        # Crop the image to only look at the bottom quarter
        # restricting the view since zebra cross is in yello
        height, width = edges.shape
        cropped = edges[3*height//4:height,:]
        # Get the indices of the white pixels
        indices = np.where(cropped == 255)
        
        # Check if there are any white pixels in the image
        if len(indices[0]) == 0:
            return 0
        # Compute the center of the white pixels
        center = np.mean(indices[1])
        #print(center)
        # Compute the deviation from the center of the image
        deviation = center - width/2
        #print(deviation)
        # Compute the steering angle
        steering_angle = deviation/(width/2)
        return steering_angle
  
    #The primary loop of the robot is contained in the run method. Based on where the yellow line is located in the camera view, it continuously modifies the robot's speed and steering angle. Robot spins at random if it is unable to detect the line. By travelling to the charging station as necessary, it also controls the battery level.
    def run(self):
        
        #This function implements the main loop of the robot.
       
        turn_timer = 0
        turn_direction = 0
        while self.tick():
        
            # Check battery level and navigate to charging area when low on battery
            if self.time_to_live < 50:
                self.navigate_to_charging_area()
                self.charge_battery()
                self.navigate_to_initial_goal()
            else:
                # Get the camera image and convert it to grayscale
                image = self.get_camera_image()            
                image = cv2.resize(image, (600,600))
                image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)  # convert to LAB color space
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))  # create a CLAHE object
                image[:,:,0] = clahe.apply(image[:,:,0])  # apply CLAHE to the L channel
                image = cv2.cvtColor(image, cv2.COLOR_LAB2BGR)  # convert back to BGR color space
                # Convert to HSV color space
                hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
                
                yellowLL = np.array([20, 60, 100])
                yellowUL = np.array([30, 130, 200])
                mask_yellow = cv2.inRange(hsv, yellowLL, yellowUL)
                mask_blur = cv2.GaussianBlur(mask_yellow, (5, 5), 0)
                edges = cv2.Canny(mask_blur, 100, 200)
            
                steering_angle = self.yellowline(edges)
            
   

            if steering_angle == 0:
                #random_turn = random.choice([-0.4, 0.4])  # Randomly choose between left (-0.4) and right (0.4) turns
                
                if turn_timer <= 0:
                    turn_direction = random.choice([0.4, 0.4])  # Randomly choose between left (-0.4) and right (0.4) turns
                    turn_timer = random.randint(10, 30)  # Set a random timer for the current direction
                else:
                    turn_timer -= 1

                steering_angle = turn_direction
                speed = 30
                self.stopFlag = 0
                #default_angle=steering_angle
             # thresholding the steering angle to avoid drift
            if steering_angle < -0.3:
                steering_angle =- 0.3
                speed = 40
                if self.fl==1:
                    self.fl=0
                #default_angle=steering_angle
            elif steering_angle > 0.3:
                steering_angle = 0.3
                speed = 40 
                if self.fl==1:
                    self.fl=0
                #default_angle=steering_angle
            else:
                 steering_angle=0
                 speed = 40
            
                       
            # Set the speed and steering angle of the robot
            self.set_speed(speed)
            self.set_steering_angle(steering_angle)
            # Display the output image with the detected edges
            output = np.dstack((edges, edges, edges))
            print(f'Time to live: {self.time_to_live}')
            print(f'GPS: {self.get_gps_values()}')
            
            # Check battery level and navigate to charging area when low on battery
            # if self.time_to_live < 50:
                # self.navigate_to_charging_area()
                # self.charge_battery()
                # self.navigate_to_initial_goal()
            
            cv2.imshow('output', output)
            cv2.waitKey(1)

    #Steers the robot towards the charging area using GPS coordinates.
    def navigate_to_charging_area(self):
        while not self.is_at_charging_area():
        # Calculate the required steering angle and speed based on the current GPS position and the charging area's GPS coordinates
            steering_angle, speed = self.calculate_navigation_parameters(self.charging_area_gps)
            self.set_steering_angle(steering_angle)
            self.set_speed(speed)
            
    #Checks if the robot is close enough to the charging area based on a threshold distance.       
    def is_at_charging_area(self):
        threshold_distance = 5
        current_gps = self.get_gps_values()
        distance = self.calculate_distance(current_gps, self.charging_area_gps)
        return distance < threshold_distance
       
    #Simulates charging the robot's battery by waiting for a specified time.  
    def charge_battery(self):
        charging_time = 10  # seconds
        time.sleep(charging_time)
        
    #Steers the robot back to the initial goal using GPS coordinates.    
    def navigate_to_initial_goal(self):
        while not self.is_at_initial_goal():
        # Calculate the required steering angle and speed based on the current GPS position and the initial goal's GPS coordinates
            steering_angle, speed = self.calculate_navigation_parameters(self.initial_goal_gps)
            self.set_steering_angle(steering_angle)
            self.set_speed(speed)
    
    #Checks if the robot is close enough to the initial goal based on a threshold distance.        
    def is_at_initial_goal(self):
        threshold_distance = 5
        current_gps = self.get_gps_values()
        distance = self.calculate_distance(current_gps, self.initial_goal_gps)
        return distance < threshold_distance



# The API of the MyRobot class, is extremely simple, not much to explain.
# We just create an instance and let it do its job.
robot = MyRobot()
robot. run()