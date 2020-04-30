import numpy as np
from gym_collision_avoidance.envs.dynamics.Dynamics import Dynamics
from gym_collision_avoidance.envs.util import wrap, find_nearest
from collections import namedtuple
import math

class RobotDynamics(Dynamics):

    # all vectors expressed in global frame
    def __init__(self, agent):
        Dynamics.__init__(self, agent)
        self.mass = 10 #kg
        self.inertia = 10
        self.pos_vector_motor_1 = np.array([0.2, -0.2])
        self.pos_vector_motor_2 = np.array([-0.2, -0.2])
        self.forward = np.array([0, 1])
        self.pos = np.array[0, 0]
        self.dir = np.array[0, 0]
        self.vel = 0
        self.rotspeed = 0

        self.motor = Motor()

    def get_rotation_matrix(theta):
        return np.array([[math.cos(theta), math.sin(theta)], [-1 * math.sin(theta), math.cos(theta)]])

    def step(self, action, dt):

        self.pos_global = self.agent.pos_global_frame
        self.dir_global = get_rotation_matrix(self.agent.heading_global_frame).dot(self.forward)

        desired_speed_local = (action[0] - self.vel) * self.forward
        current_speed_local = np.transpose(get_rotation_matrix(self.agent.heading_global_frame)).dot(self.agent.speed_global_frame)
        delta_speed_local = desired_speed_local - current_speed_local
        net_force_acc_local = np.clip((dt / self.mass) * delta_speed_local, -2 * self.motor.get_max_torque(current_speed_local[1]), 2 * self.motor.get_max_torque(current_speed_local[1]))

        delta_rot_speed_global = action[1] / dt - self.rotspeed

class Motor():

    def __init__(self):
        self.maxForce = 100 #N
        self.maxSpeed = 2 #m/s

    def get_max_speed(force):
        return np.array([0, (self.maxSpeed / self.maxForce) * force])

    def get_max_torque(speed):
        return np.array((self.maxForce / self.maxSpeed) * (self.maxSpeed - speed))
