import numpy as np
import math
from gym_collision_avoidance.envs.dynamics.Dynamics import Dynamics
from gym_collision_avoidance.envs.util import wrap, find_nearest

class RobotDynamics(Dynamics):

    def __init__(self, agent):
        Dynamics.__init__(self, agent)
        self.mass = 12
        self.inertia = 6
        self.maxForce = 35
        self.maxTorque = 35
        self.maxSpeed = 1
        self.maxRotSpeed = np.pi*10

        self.dt = 0.1
        self.A = np.array([[0, 1, 0, 0],
                            [0, 0, 0, 0],
                            [0, 0, 0, 1],
                            [0, 0, 0, 0]])

        self.stateVector = np.array([0, 0, 0, 0])

    def B(self, omega, dt):
        return np.array([0,
                        dt * math.cos(omega)/self.mass,
                        0,
                        dt * math.sin(omega)/self.mass])

    def step(self, action, dt):
        set_speed = action[0]
        rotation = action[1]

        deltaSpeed = set_speed - math.sqrt(self.stateVector[1]**2 + self.stateVector[3]**2)
        force = np.clip(deltaSpeed * self.mass / dt, -self.maxForce, self.maxForce)

        self.stateVector = np.array([self.agent.pos_global_frame[0],
                                    self.stateVector[1],
                                    self.agent.pos_global_frame[1],
                                    self.stateVector[3]])

        maxStateVector = np.array([np.inf,
                                    self.maxSpeed * math.cos(self.agent.heading_global_frame),
                                    np.inf,
                                    self.maxSpeed * math.sin(self.agent.heading_global_frame)])

        minStateVector = -maxStateVector

        s_dot = self.A.dot(self.stateVector) + self.B(self.agent.heading_global_frame, dt) * force
        self.stateVector += s_dot * dt
        #print(self.stateVector)
        self.stateVector = np.clip(self.stateVector, minStateVector, maxStateVector)

        self.agent.pos_global_frame[0] = self.stateVector[0]
        self.agent.vel_global_frame[0] = self.stateVector[1]
        self.agent.pos_global_frame[1] = self.stateVector[2]
        self.agent.vel_global_frame[1] = self.stateVector[3]
        self.agent.speed_global_frame = math.sqrt(self.stateVector[1]**2 + self.stateVector[3]**2)

        selected_heading = wrap(action[1] + self.agent.heading_global_frame)

        self.agent.delta_heading_global_frame = wrap(selected_heading - self.agent.heading_global_frame)
        self.agent.heading_global_frame = selected_heading
