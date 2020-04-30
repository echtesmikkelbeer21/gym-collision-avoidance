import numpy as np
from gym_collision_avoidance.envs.dynamics.Dynamics import Dynamics

class RobotDynamics(Dynamics):
    # all vectors expressed in global frame
    def __init__(self, agent):
        Dynamics.__init__(self, agent)
        self.mass = 10 #kg
        self.inertia = 10 #kgm^2
        self.forward = np.array([0, 1])
        self.max_motor_force_local = np.array([0, 10])
        self.pos_motor_a_local = np.array([0.2, -0.2])
        self.pos_motor_b_local = np.array([-0.2, -0.2])
        self.pos = np.array[0, 0]
        self.dir = np.array[0, 0]
        self.vel = 0
        self.rotspeed = 0

        self.mainMatrix = np.linalg.inv(np.array([[1.0, 1.0],[self.pos_motor_a_local[0], self.pos_motor_b_local[0]]]))

    def get_rotation_matrix(theta):
        return np.array([[math.cos(theta), math.sin(theta)], [-1 * math.sin(theta), math.cos(theta)]])

    def crossproduct2D(vector1, vector2):
        return vector1[0] * vector2[1] - vector1[1] * vector2[0]

    def step(self, action, dt):
        self.pos_global = self.agent.pos_global_frame
        self.dir_global = get_rotation_matrix(self.agent.heading_global_frame).dot(self.forward)

        desired_speed_local = (action[0] - self.vel) * self.forward
        current_speed_local = np.transpose(get_rotation_matrix(self.agent.heading_global_frame)).dot(self.agent.speed_global_frame)
        delta_speed_local = desired_speed_local - current_speed_local
        net_force_acc_local = np.clip((dt / self.mass) * delta_speed_local, -2 * self.max_motor_force_local, 2 * self.max_motor_force_local)

        max_torque = crossproduct2D(pos_motor_a_local, max_motor_force_local) * 2

        desired_rot_speed_local = action[1] / dt
        delta_rot_speed_local = desired_rot_speed_local - self.rotspeed
        net_torque_acc_local = np.clip((dt / self.inertia) * delta_rot_speed_local, -max_torque, max_torque)

        #now solve for system of vector equations using inverse matrix multiplication
        forceVector = np.dot(self.mainMatrix, np.array([net_force_acc_local[1], net_torque_acc_local]))
        motor1, motor2 = np.array([0, forceVector[0]]), np.array([0, forceVector[1]])
