import rclpy as rospy
from rclpy.node import Node


from can_zoe_msgs.msg import CmdSteering
from can_zoe_msgs.msg import CmdAcceleration

from can_zoe_msgs.msg import SteeringInfo
from novatel_oem7_msgs.msg import CORRIMU
#from can_zoe_msgs.msg import IMU
from can_zoe_msgs.msg import Smotion

import traceback

from mpc_package.class_mpc import MPC_Controller_LPV
from mpc_package.utils_system import get_A_B_matrices

import os 
import math
import numpy as np


import time



class MPC_Vehicle(Node):


    def __init__(self):

        super().__init__("mpc_zoe")


        #Subscription
        self.sub_smotion = self.create_subscription(Smotion , "/can/smotion" , self.cb_smotion , 10) 
        self.sub_steer = self.create_subscription(SteeringInfo , "/can/steering_info" , self.cb_steering , 10) 
        self.sub_imu  = self.create_subscription(CORRIMU , "/novatel/oem7/corrimu" , self.cb_imu , 10) 


        #Publish
        self.pubsteering = self.create_publisher(CmdSteering , "/cmd_steering" , 10)
        self.ax_pub = self.create_publisher(CmdAcceleration , "/cmd_acceleration" , 10)

        self.Ts_inl = 0.01
        self.Ts_mpc = 0.033

        self.mpc_divider = int(self.Ts_mpc/self.Ts_inl)
        self.tick = 0

        self.nx = 3
        self.nu = 2
        self.Hp = 6


        self.Sched_Var = np.array([
            [0.95  , 18.0],
            [-1.4 , 1.4],
            [-1.4 , 1.4],
            [-0.867 , 0.867],
            [-6.0 , 22.0]
        ] , dtype=np.float64)

        self.x_min = np.array([self.Sched_Var[0,0] , self.Sched_Var[1,0] , self.Sched_Var[2,0]])
        self.x_max = np.array([self.Sched_Var[0,1] , self.Sched_Var[1,1] , self.Sched_Var[2,1]])

        self.u_min = np.array([self.Sched_Var[3,0] , self.Sched_Var[4,0]])
        self.u_max = np.array([self.Sched_Var[3,1] , self.Sched_Var[4,1]])

        self.vx = 0.5
        self.vy = 0.0
        self.psi = 0.0
        self.slip = 0.0


        self.ax = 0.0
        self.steer_fb = 0.0


        self.meas = False
        self.last_meas_ns = 0



        self.vx_ref_const = 5.0
        self.yaw_rate = 0.0


        self.ref = np.zeros((self.nx , 1 , self.Hp+1))

        self.A_vec = []
        self.B_vec = []


        self.u_last = np.array([0.0 , 0.0] , dtype=np.float64)
        self.u_minus = self.u_last
        self.XX = np.zeros((self.nx , 1 , self.Hp))


        self.K = None

        self.timer = self.create_timer(self.Ts_inl , self.main_loop)
        self.get_logger().info("Node MPC started ...")


    def cb_steering(self , msg : SteeringInfo):
        self.steer_fb = float(msg.position_dae)
        self.mark_meas_ok()

    def cb_smotion(self , msg : Smotion):

        self.vx = msg.vel_x
        self.vy = msg.vel_y
        self.slip = msg.angle

        self.mark_meas_ok()


    def cb_imu(self , msg : CORRIMU):
        
        self.psi = msg.yaw_rate
        self.mark_meas_ok()

    def mark_meas_ok(self):

        self.meas = True
        self.last_meas_ns = self.get_clock().now().nanoseconds

    def main_loop(self):

        self.tick += 1


        if (self.tick % self.mpc_divider == 0): 
            self.mpc_solve_once()

        self.publish_cmd(self.u_last)

    def mpc_solve_once(self):

        S_state = np.array([self.vx , self.vy , self.psi] , dtype=np.float64)
        Qvx     = 1/self.Sched_Var[0,1]**2
        Qvy     = 1/self.Sched_Var[1,1]**2
        Qw      = 1/self.Sched_Var[2,1]**2
        Qdelta  = 1/self.Sched_Var[3,1]**2
        Qaccel  = 1/self.Sched_Var[4,1]**2

        Q   = 0.8*np.diag([ 2.8*Qvx, 0.71*Qvy  , 5.1*Qw])
        R   = 0.2*np.diag([ 0.5*Qdelta , 0.5*Qaccel ])

        for i in range(self.Hp+1):

            self.ref[: , : , i] = np.array([self.vx_ref_const , 0.0 , 0.0] , dtype=np.float64).reshape(-1,1)

        self.A_vec.clear()
        self.B_vec.clear()


        vx_meas = self.vx if self.vx == 0 else 0.5
        vy_meas = self.vy
        delta = float(self.u_minus[0])

        for _ in range(self.Hp): 
            A , B = get_A_B_matrices(delta , vx_meas , vy_meas , self.Ts_mpc , option="Variant B")
            self.A_vec.append(A)
            self.B_vec.append(B)

        x0 = S_state.reshape(-1)

        try: 

            t0 = time.perf_counter()

            if self.K is None:

                self.K = MPC_Controller_LPV(
                    self.nx, self.nu,
                    Np=self.Hp, Nc=self.Hp,
                    x_ref=self.ref, x0=x0,
                    x_min=self.x_min, x_max=self.x_max,
                    u_minus=self.u_minus, u_ref=None,
                    u_min=self.u_min, u_max=self.u_max,
                    Qx=Q, QxN=Q, Qu=None, QDu=R,
                    Du_min=None, Du_max=None,
                    esp_feas=1e6, esp_rel=1e-3, esp_abs=1e-3
                )
                self.K.Du_ineq = False
                self.K.setup_MPC(self.A_vec, self.B_vec)
            else:
                self.K.update_MPC(
                    A_vec=self.A_vec, B_vec=self.B_vec,
                    x_ref=self.ref, x0=x0,
                    u_ref=None, u_minus=self.u_minus,
                    x_min=self.x_min, x_max=self.x_max,
                    u_min=self.u_min, u_max=self.u_max,
                    tube_int=False
                )

            self.K.solve()
            info, uMPC = self.K.output()

            solve_ms = (time.perf_counter() - t0) * 1000.0
            status = info.get('status', 'unknown')

            # Feasibility monitoring
            self.get_logger().info(f"MPC status={status} solve={solve_ms:.2f}ms  x=[{self.vx:.2f},{self.vy:.2f},{self.yaw_rate:.2f}]")

            if(info['status'] != 'solved'):

                self.u_last[:] = np.array([float(uMPC[0]) , float(uMPC[1])] , dtype=np.float64)
                self.u_minus[:] = self.u_last

        except Exception as e:
            traceback.print_exc()
            self.get_logger().warn(f"MPC Exception : {e}")
            self.u_last[:] = np.array([0.0 , 0.0])

    def publish_cmd(self , u : np.ndarray): 

        delta = float(np.clip(u[0], self.u_min[0], self.u_max[0]))
        ax    = float(np.clip(u[1], self.u_min[1], self.u_max[1]))

        msg_s = CmdSteering()

        msg_s.angle = delta

        self.pubsteering.publish(msg_s)

        msg_a = CmdAcceleration()

        msg_a.accel = ax

        self.ax_pub.publish(msg_a)



def main():
    rospy.init()
    node = MPC_Vehicle()

    try: 
        rospy.spin(node)
    finally:
        node.destroy_node()
        rospy.shutdown()


if __name__ == "__main__":
    main()


        











        




    

    

    















