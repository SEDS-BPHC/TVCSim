#Modules

from vpython import *
from time import sleep
import matplotlib.pyplot as plt 

#Time step

dt = 0.1

setpoint = 6
setpoint_line = curve(vector(-10,setpoint,0), 
						vector(10, setpoint, 0),
						color = color.blue)

#Rocket params
initial_y_vel = 0
initial_x_vel = 0
initial_height = -10
max_thrust = 15 #newtons
mass = 1 #kg
g = -9.8
min_pos = -10
