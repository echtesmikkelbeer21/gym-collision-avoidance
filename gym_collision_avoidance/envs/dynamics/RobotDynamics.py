import numpy as np
import math
from gym_collision_avoidance.envs.dynamics.Dynamics import Dynamics

class RobotDynamics(Dynamics):
    # all vectors expressed in global frame
    def __init__(self, agent):
        Dynamics.__init__(self, agent)
        self.mass = 35 #kg
        self.inertia = 35 #kgm^2
        self.forward = np.array([0, 1])
        self.max_motor_force_local = np.array([0, 10])
        self.pos_motor_a_local = np.array([0.2, -0.2])
        self.pos_motor_b_local = np.array([-0.2, -0.2])
        self.rotspeed = 0

        self.mainMatrix = np.linalg.inv(np.array([[1.0, 1.0],[self.pos_motor_a_local[0], self.pos_motor_b_local[0]]]))

    def get_rotation_matrix(self, theta):
        return np.array([[math.cos(theta), math.sin(theta)], [-1 * math.sin(theta), math.cos(theta)]])

    def crossproduct2D(self, vector1, vector2):
        return vector1[0] * vector2[1] - vector1[1] * vector2[0]

    def step(self, action, dt):
        desired_speed_local = (action[0] - self.agent.speed_global_frame) * self.forward
        current_speed_local = np.transpose(self.get_rotation_matrix(self.agent.heading_global_frame)).dot(self.agent.vel_global_frame)
        delta_speed_local = desired_speed_local - current_speed_local
        net_force_acc_local = (dt / self.mass) * delta_speed_local

        desired_rot_speed_local = action[1] / dt
        delta_rot_speed_local = desired_rot_speed_local - self.rotspeed
        net_torque_acc_local = (dt / self.inertia) * delta_rot_speed_local

        #now solve for system of vector equations using inverse matrix multiplication
        forceVector = np.dot(self.mainMatrix, np.array([net_force_acc_local[1], net_torque_acc_local]))
        motor1 = np.clip(np.array([0, forceVector[0]]), -self.max_motor_force_local, self.max_motor_force_local)
        motor2 = np.clip(np.array([0, forceVector[1]]), -self.max_motor_force_local, self.max_motor_force_local)

        #calculating new (angular) velocities and applying
        torque = self.crossproduct2D(self.pos_motor_a_local, motor1) + self.crossproduct2D(self.pos_motor_b_local, motor2)
        self.agent.delta_heading_global_frame = 0.5 * (torque / self.inertia) * dt**2 + dt * self.rotspeed
        self.agent.heading_global_frame += self.agent.delta_heading_global_frame
        self.rotspeed += (torque / self.inertia) * dt

        force_global = np.dot(self.get_rotation_matrix(self.agent.heading_global_frame), (motor1 + motor2))
        self.agent.pos_global_frame += 0.5 * (force_global / self.mass) * dt**2 + dt * self.agent.vel_global_frame
        self.agent.vel_global_frame += (force_global / self.mass) * dt
        self.agent.speed_global_frame = math.sqrt(self.agent.vel_global_frame[0]**2 + self.agent.vel_global_frame[1]**2)
