# ROS 2 node for Zoe speed and lateral-dynamics control. The MPC predicts the
# next few steps, but only the first command of each solved plan is published.
# Tube mode adds margins to the prediction constraints. It does not apply a
# separate feedback correction to the published command.
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

from scipy.linalg import solve_discrete_are
from mpc_package.sets_library import zonotope 

import time
from dataclasses import dataclass


# Small assumed uncertainty in [longitudinal speed, lateral speed, yaw rate].
# R_G gives the half-width along each axis of the initial error box.
eps = 1e-6

R_c = np.zeros((3, 1))
R_G = np.diag([eps, eps, eps])
W_c = np.zeros((3, 1))


@dataclass(frozen=True)
class VehicleParameters:
    """Physical parameters used by the nonlinear bicycle plant."""

    # Only the subset used below contributes to W_G;
    # get_A_B_matrices supplies the prediction model separately.

    mass: float = 1754.0
    yaw_inertia: float = 3651.0
    front_distance: float = 1.10
    rear_distance: float = 1.45
    air_density: float = 1.225
    longitudinal_drag_area: float = 0.70
    lateral_drag_area: float = 1.50
    aerodynamic_center: float = -0.35
    rolling_resistance: float = 0.012
    gravity: float = 9.81
    minimum_speed: float = 0.50
    acceleration_gain: float = 1.0
    acceleration_bias: float = 0.0


def dlqr(A: np.ndarray, B: np.ndarray, Q: np.ndarray, R: np.ndarray) -> tuple:
    """Compute a discrete-time LQR gain and closed-loop eigenvalues.

    The returned gain follows the same positive-gain convention as MATLAB's
    ``dlqr``. Callers in this script negate it when using the correction in the
    form ``u = K x``.

    Args:
        A: Discrete-time state matrix.
        B: Discrete-time input matrix.
        Q: State-cost matrix.
        R: Input-cost matrix.

    Returns:
        A tuple containing the gain, Riccati solution, and closed-loop poles.
    """
    # Solve for the LQR gain. The returned K is used with a minus sign, so the
    # closed-loop model here is A - B @ K. The tube loop applies that sign once.
    X = solve_discrete_are(A, B, Q, R)
    K = np.linalg.inv(B.T @ X @ B + R) @ (B.T @ X @ A)
    eigVals = np.linalg.eigvals(A - B @ K)
    return K, X, eigVals


VEHICLE = VehicleParameters()
# Disturbance increments at the base sample time: slope changes vx, while
# lateral wind changes vy and creates a yaw moment around the center of mass.
Ts = 0.01
max_wind_force = (0.5 * VEHICLE.air_density * VEHICLE.lateral_drag_area * 2**2)
vx_max_e = Ts * VEHICLE.gravity * np.sin(0.1)
vy_max_e = Ts * max_wind_force / VEHICLE.mass
vw_max_e = (Ts * abs(VEHICLE.aerodynamic_center) * max_wind_force / VEHICLE.yaw_inertia)
W_G =  np.diag([vx_max_e, vy_max_e, vw_max_e])


class MPC_Vehicle(Node):


    def __init__(self):

        super().__init__("mpc_zoe")


        # Sensor callbacks cache the latest values independently. They do not
        # wait for a synchronized set of messages before a control step.
        self.sub_smotion = self.create_subscription(Smotion , "/can/smotion" , self.cb_smotion , 10) 
        self.sub_steer = self.create_subscription(SteeringInfo , "/can/steering_info" , self.cb_steering , 10) 
        self.sub_imu  = self.create_subscription(CORRIMU , "/novatel/oem7/corrimu" , self.cb_imu , 10) 
        self.z = zonotope.Zonotope(R_c, R_G)



        # Output topics consumed by the vehicle interface.
        self.pubsteering = self.create_publisher(CmdSteering , "/cmd_steering" , 10)
        self.ax_pub = self.create_publisher(CmdAcceleration , "/cmd_acceleration" , 10)

        # The timer is scheduled every 10 ms. Integer division gives three
        # ticks per solve (nominally 30 ms), but prediction uses a 33 ms step.
        self.Ts_inl = 0.01
        self.Ts_mpc = 0.033

        self.mpc_divider = int(self.Ts_mpc/self.Ts_inl)
        self.tick = 0

        # State: [vx, vy, yaw rate]. Input: [steering, acceleration].
        # Six input steps produce seven state points, including the current one.
        self.nx = 3
        self.nu = 2
        self.Hp = 6


        # Each row is [lower, upper], in state order followed by input order.
        # These are model limits, not the CAN driver's command encoding.
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

        # Initial values are placeholders until sensor callbacks arrive.
        self.vx = 0.5
        self.vy = 0.0
        self.psi = 0.0
        self.slip = 0.0

        self.ax = 0.0
        self.steer_fb = 0.0

        self.meas = False
        self.last_meas_ns = 0

        self.vx_ref_const = 8.0
        self.yaw_rate = 0.0

        # The reference and model matrices are filled for each new horizon.
        self.ref = np.zeros((self.nx , 1 , self.Hp+1))

        self.A_vec = []
        self.B_vec = []

        # Hold the last command between solves. u_minus shares the same array.
        # XX and UU store the previous solved state and input trajectories.
        self.u_last = np.array([0.0 , 0.0] , dtype=np.float64)
        self.u_minus = self.u_last
        self.XX = np.zeros((self.nx , 1 , self.Hp))
        #CHANGE: Input predictions contain nu entries per stage, not nx.
        self.UU = np.zeros((self.nu , 1 , self.Hp))

        # K holds the QP controller object, not the LQR gain K_MPC.
        self.K = None
        # True freezes A and B across the horizon. False allows later stages
        # to use the previous solution once a successful solve is available.
        self.LTI_sys = False
        # At startup or after a failed solve, there is no plan to reuse.
        self.init_step = True

        #CHANGE: Enable the experimental tube constraint tightening.
        self.tube_enabled = True
        self.tube_solution_valid = False

        #CHANGE: Build the constraint sets from the actual (possibly asymmetric) bounds.
        # The tube loop uses the equivalent x_min/x_max and u_min/u_max arrays directly.
        self.X_Set_Zon = zonotope.Zonotope.from_bounds(self.x_min, self.x_max)
        self.U_Set_Zon = zonotope.Zonotope.from_bounds(self.u_min, self.u_max)
        # W_G is an Euler disturbance increment at Ts; use the prediction step here.
        # This only models wind and slope, not all LPV or actuator uncertainty.
        self.W_Zonotope = zonotope.Zonotope(W_c, (self.Ts_mpc / Ts) * W_G)
        self.R_Zonotope = zonotope.Zonotope(R_c, R_G)
        self.timer = self.create_timer(self.Ts_inl , self.main_loop)
        self.get_logger().info("Node MPC started ...")

    def cb_steering(self , msg : SteeringInfo):
        # Cache steering feedback. The current model scheduling uses the last
        # commanded steering instead of this measurement.
        self.steer_fb = float(msg.position_dae)
        self.mark_meas_ok()

    def cb_smotion(self , msg : Smotion):
        # Cache velocity and slip as received. No unit conversion is done here.
        self.vx = msg.vel_x
        self.vy = msg.vel_y
        self.slip = msg.angle
        self.mark_meas_ok()


    def cb_imu(self , msg : CORRIMU):
        # Despite the name psi, this state stores yaw rate, not yaw angle.
        self.psi = msg.yaw_rate
        self.mark_meas_ok()

    def mark_meas_ok(self):
        # Record the arrival of any sensor message. This is not a per-sensor
        # freshness check, and the control loop does not gate on it yet.
        self.meas = True
        self.last_meas_ns = self.get_clock().now().nanoseconds

    def main_loop(self):
        # Run the solve inline on the scheduled tick. A slow solve can delay
        # publishing because both operations happen in this timer callback.
        self.tick += 1

        if (self.tick % self.mpc_divider == 0): 
            self.mpc_solve_once()

        #CHANGE: Do not publish a new command from an invalid experimental tube plan.
        # Skipping publication is not an active stop command.
        if not self.tube_enabled or self.tube_solution_valid:
            self.publish_cmd(self.u_last)

    def _tube_bounds(self, Q, R):
        # Reserve room inside the original constraints for the errors the
        # prediction expects. The result is one lower/upper pair per stage.
        #CHANGE: Keep the flat bounds used by update_MPC.
        x_min_tube = np.empty(self.nx * (self.Hp + 1))
        x_max_tube = np.empty_like(x_min_tube)
        u_min_tube = np.empty(self.nu * self.Hp)
        u_max_tube = np.empty_like(u_min_tube)
        # The current state (stage 0) keeps the original limits. Future state
        # bounds use the tube after propagating it to the corresponding stage.
        x_min_tube[:self.nx] = self.x_min
        x_max_tube[:self.nx] = self.x_max
        R_Zonotope = self.R_Zonotope

        for i in range(self.Hp):
            A, B = self.A_vec[i], self.B_vec[i]
            # This gain describes the assumed error feedback for this stage.
            # It is used to predict the tube, not added to the ROS command.
            K_MPC, _, _ = dlqr(A, B, Q, R)
            K_MPC = -K_MPC
            Acl = A + B @ K_MPC

            #CHANGE: Tighten input stage i using the current tube mapped by K_MPC.
            # If the correction can range from -0.1 to +0.1, for example,
            # the lower limit moves up by 0.1 and the upper limit down by 0.1.
            du_min, du_max = R_Zonotope.linear_mapping(K_MPC).bounds()
            u_min_tube[i*self.nu:(i+1)*self.nu] = self.u_min - du_min.ravel()
            u_max_tube[i*self.nu:(i+1)*self.nu] = self.u_max - du_max.ravel()

            #CHANGE: Propagate the tube, then tighten state stage i+1.
            # Acl carries the current error forward; W adds the disturbance
            # for this step. bounds() returns the resulting box on each axis.
            R_Zonotope = R_Zonotope.linear_mapping(Acl) + self.W_Zonotope
            dx_min, dx_max = R_Zonotope.bounds()
            x_min_tube[(i+1)*self.nx:(i+2)*self.nx] = self.x_min - dx_min.ravel()
            x_max_tube[(i+1)*self.nx:(i+2)*self.nx] = self.x_max - dx_max.ravel()

        # A lower bound above its upper bound means no value fits the margin.
        if np.any(x_min_tube > x_max_tube) or np.any(u_min_tube > u_max_tube):
            raise ValueError("Tube tightening produced an empty constraint interval")

        return x_min_tube, x_max_tube, u_min_tube, u_max_tube

    def mpc_solve_once(self):

        # Build the current state and scale the cost weights using the model
        # limits. Q weights state tracking; the QP uses R for input changes.
        S_state = np.array([self.vx , self.vy , self.psi] , dtype=np.float64)
        Qvx     = 1/self.Sched_Var[0,1]**2
        Qvy     = 1/self.Sched_Var[1,1]**2
        Qw      = 1/self.Sched_Var[2,1]**2
        Qdelta  = 1/self.Sched_Var[3,1]**2
        Qaccel  = 1/self.Sched_Var[4,1]**2

        Q   = 0.8*np.diag([ 2.8*Qvx, 0.71*Qvy  , 5.1*Qw])
        R   = 0.2*np.diag([ 0.5*Qdelta , 0.5*Qaccel ])

        # Ask for steady forward speed and zero lateral speed and yaw rate
        # at every state point, including the terminal point.
        for i in range(self.Hp+1):

            self.ref[: , : , i] = np.array([self.vx_ref_const , 0.0 , 0.0] , dtype=np.float64).reshape(-1,1)

        self.A_vec.clear()
        self.B_vec.clear()


        # Clamp speed only when building the model from the current operating
        # point, because get_A_B_matrices divides by vx. The QP still gets the
        # measured vx in x0; this clamp does not make standstill feasible.
        vx_meas = max(float(self.vx), 1.0) #self.vx if self.vx != 0 else 2.0
        vy_meas = self.vy
        delta = float(self.u_minus[0])

        # Stage 0 uses the latest velocity and last commanded steering.
        # Later stages reuse the previous plan in LPV mode. If no plan is
        # available yet, use the same current operating point for every stage.
        for i in range(self.Hp):
            if self.LTI_sys or self.init_step or i == 0:
                A , B = get_A_B_matrices(delta , vx_meas , vy_meas , self.Ts_mpc , option="Variant B")
            else:
                A , B = get_A_B_matrices(self.UU[0,0,i] , self.XX[0,0,i] , self.XX[1,0,i] , self.Ts_mpc , option="Variant B")

            self.A_vec.append(A)
            self.B_vec.append(B)

        x0 = S_state.reshape(-1)

        self.tube_solution_valid = False
        try:

            # This timing starts after model construction. It includes tube
            # bounds and the QP, but not callback delays or publishing.
            t0 = time.perf_counter()

            #CHANGE: Feed stage-wise tube bounds into both QP setup and update.
            if self.tube_enabled:
                print("Tube Based MPC enabled")
                x_min, x_max, u_min, u_max = self._tube_bounds(Q, R)
            else:
                x_min, x_max = self.x_min, self.x_max
                u_min, u_max = self.u_min, self.u_max

            # Build the solver on the first attempt or after a reset. On
            # later attempts, update the existing problem with the new horizon.
            if self.K is None:

                self.K = MPC_Controller_LPV(
                    self.nx, self.nu,
                    Np=self.Hp, Nc=self.Hp,
                    x_ref=self.ref, x0=x0,
                    x_min=x_min, x_max=x_max,
                    u_minus=self.u_minus, u_ref=None,
                    u_min=u_min, u_max=u_max,
                    Qx=Q, QxN=Q, Qu=None, QDu=R,
                    Du_min=None, Du_max=None,
                    esp_feas=1e6, esp_rel=1e-3, esp_abs=1e-3
                )
                # Input changes are penalized by QDu, but not hard-limited here.
                self.K.Du_ineq = False
                self.K.SOFT_RELAX = not self.tube_enabled  # Keep tightened state bounds hard.
                self.K.setup_MPC(self.A_vec, self.B_vec, tube_int=self.tube_enabled)
            else:
                self.K.update_MPC(
                    A_vec=self.A_vec, B_vec=self.B_vec,
                    x_ref=self.ref, x0=x0,
                    u_ref=None, u_minus=self.u_minus,
                    x_min=x_min, x_max=x_max,
                    u_min=u_min, u_max=u_max,
                    tube_int=self.tube_enabled
                )

            # Extract the first optimized input and retain the horizon for
            # scheduling the next solve if this one succeeds.
            self.K.solve()
            info, uMPC = self.K.output()

            solve_ms = (time.perf_counter() - t0) * 1000.0
            status = info.get('status', 'unknown')

            # Log the solver status and the measured computation time.
            self.get_logger().info(f"MPC status={status} solve={solve_ms:.2f}ms v={self.vx:.2f} u_ac={uMPC[1]:.2f}")

            if(info['status'] == 'solved'):
                #CHANGE: Only use a solved trajectory for the next LPV schedule.
                # Skip x_trajectory[0], which is the current state rather than a prediction.
                for j in range(self.Hp):
                    self.XX[:,:,j] = info['x_trajectory'][j+1,:].reshape(-1,1)
                    self.UU[:,:,j] = info['u_trajectory'][j,:].reshape(-1,1)
                self.init_step = False
                # Send the nominal MPC input without an extra tube feedback term.
                self.u_last[:] = np.array([float(uMPC[0]) , float(uMPC[1])] , dtype=np.float64)
                self.u_minus[:] = self.u_last
                self.tube_solution_valid = True
            else:
                # Discard the failed plan and rebuild the solver next time.
                # Tube mode suppresses publishing until another solve succeeds.
                self.u_last[:] = np.array([0.0, 0.0])
                self.K = None
                self.init_step = True

        except Exception as e:
            traceback.print_exc()
            self.get_logger().warn(f"MPC Exception : {e}")
            self.u_last[:] = np.array([0.0 , 0.0])
            self.K = None
            self.init_step = True

    def publish_cmd(self , u : np.ndarray): 

        # Clip to the original model limits. The values are copied into the
        # ROS messages directly; this is not a model-to-CAN unit conversion.
        delta = float(np.clip(u[0], self.u_min[0], self.u_max[0]))
        ax    = float(np.clip(u[1], self.u_min[1], self.u_max[1]))

        msg_s = CmdSteering()
        msg_s.angle = delta
        self.pubsteering.publish(msg_s)

        msg_a = CmdAcceleration()
        msg_a.accel = ax
        msg_a.can_publish = True
        self.ax_pub.publish(msg_a)



def main():
    # Initialize ROS 2, process callbacks, and release the node on shutdown.
    rospy.init()
    node = MPC_Vehicle()

    try: 
        rospy.spin(node)
    finally:
        node.destroy_node()
        rospy.shutdown()


if __name__ == "__main__":
    main()
