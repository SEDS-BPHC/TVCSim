#Modules

from vpython import *
from time import sleep
from math import cos 
from math import sin 
from math import pi 
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
radius_of_gyr = 1 #m
moment_of_inertia = mass*radius_of_gyr**2/12
#PID constants
#time period = 
#KU = 

'''TVC performed using two control loops, one for X
and one for Y. Orthogonality has been assumed'''

class Rocket(object):
	"""docstring for Rocket"""
	def __init__(self):
		self.body = cylinder(
            pos=vector(0, initial_height, 0),
            color=color.red,
            length=3.5,
            radius=0.3,
            axis=vector(0, 1, 0),
            opacity=0.6,
            make_trail = True
            )

		self.velocity = vector(initial_x_vel, initial_y_velocity, 0)

        self.acc = vector(0, 0, 0)

        #Angle that the body makes with the X, Y and Z axes
        self.orientation = vector(90, 0, 0)

        self.rot_vel = vector(0, 0, 0)

        self.rot_acc = vector(0, 0, 0)

        self.trail = curve(color=color.white)

        '''Angle that the line passing through nose and Cg makes with 
        the y axis''' 
        self.theta = 0

        '''Angle that the thrust vector makes with the line passing through 
        nose and centre of gravity'''
        self.phi = 0

        def set_acc(self):
        	self.acc = vector(thrust*sin((theta+phi)*pi/180)/mass, g + thrust*cos((theta+phi)*pi/180)/mass, 0)

        def get_y_acc(self):
        	return self.acc.y

        def get_x_acc(self):
        	return self.acc.x 

        def set_vel(self):
        	self.velocity += self.acc*dt 

        def get_y_vel(self):
        	return self.velocity.y

        def get_x_vel(self):
        	return self.velocity.x 

        def set_rot_acc(self):
        	self.rot_acc = vector(0, 0, -thrust*sin(phi*pi/180)*radius_of_gyr/moment_of_inertia)

        def set_rot_vel(self):
        	self.rot_vel += self.rot_acc*dt 

        def set_orientation(self):
        	self.orientation +=  (self.rot_vel.z*dt, self.rot_vel.z*dt, 0)

        def set_pos(self):
        self.body.pos.y += self.velocity.y * dt
        if(self.body.pos.y <= min_pos):
            self.body.pos.y = min_pos

    	def get_pos(self):
        	return self.body.pos.y

