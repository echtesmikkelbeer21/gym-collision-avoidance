import numpy as np
from gym_collision_avoidance.envs.dynamics.Dynamics import Dynamics
from gym_collision_avoidance.envs.util import wrap, find_nearest
from collections import namedtuple
import math

Vector2 = namedtuple('Vector2', 'x y')

class RobotDynamics(Dynamics):

    def __init__(self, agent):
        Dynamics.__init__(self, agent)
        self.mass = 10
        self.wheeldiam = 0.1
        self.maxMotorRPM = 100

    def step(self, action, dt):

        desired_speed = action[0]
        deltaV = self.agent.speed_global_frame - desired_speed
