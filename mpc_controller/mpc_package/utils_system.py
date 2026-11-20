

import numpy as np


def nonlinear_DriverlessUPC_model_matlab_v2(t, states, u , d):
	"""
	Transcription of the provided MATLAB nonlinear_DriverlessUPC_model function to Python.
	Args:
		t (float): Current time (not used, for compatibility).
		states (np.ndarray): State vector [vx, vy, w].
		u (np.ndarray): Input vector [delta, a, road_slope, v_wind].
	Returns:
		np.ndarray: Derivative of states [dx_vx, dx_vy, dx_w].
	"""
	lf = 1.10
	lr = 1.45
	m = 1754.0
	I = 3651.0
	cdA = 1.64
	rho = 1.225
	g = 9.81
	Cf = 4.4458e4
	Cr = 1.6485 * Cf

	# Lateral Pacejka parameters (not used in this linear tire model)
	Df = 1.2246e+03
	Cff = 1.1804
	Bf = -17.3065
	SHyf = 0.0014
	SVyf = 1.9536

	Dr = 902.7381
	Crr = 1.1804
	Br = -19.2720
	SHyr = 0.0011
	SVyr = -2.2360

	# States
	vx = states[0]
	vy = states[1]
	w = states[2]

	# Inputs
	delta = u[0]
	a = u[1]
	road_slope = d[0]
	v_wind = d[1]

	a_F = 0.0
	a_R = 0.0

	if abs(vx) > 0.1:
		a_F = np.arctan((vy + lf * w) / vx) - delta
		a_R = np.arctan((vy - lr * w) / vx)

	# Linear tire model
	FyF = -Cf * a_F
	FyR = -Cr * a_R

	# Optionally warn for large slip angles
	pi = np.pi
	if abs(a_F) > 30.0 / 180.0 * pi or abs(a_R) > 30.0 / 180.0 * pi:
		pass  # print("WARNING: Large slip angles in simulation")

	F_drag = 1.4 * vx
	Fy_wind = 0.5 * 1.225 * 1.5 * v_wind ** 2

	dx = np.zeros(3)
	dx[0] = a - F_drag - FyF * np.sin(delta) / m + w * vy - g * np.sin(road_slope)
	dx[1] = (FyF * np.cos(delta) + FyR - Fy_wind) / m - w * vx
	dx[2] = (FyF * lf * np.cos(delta) - FyR * lr - Fy_wind * (lf - lr)) / I

	return dx








def get_A_B_matrices(delta , vx , vy , Ts , option):
	"""
	Placeholder function to compute A and B matrices.
	Replace with actual computation based on the vehicle model.
	"""
	# Vehicle parameters (added after line 99)
	lf = 1.10      # Distance from CG to front axle (meters)
	lr = 1.45      # Distance from CG to rear axle (meters)
	m = 1754.0         # Mass (kg)
	I = 3651.0       # Yaw moment of inertia (kg*m^2)
	Cf = 4.1458e4      # Front cornering stiffness (N/rad)
	Cr = 1.6845*Cf      # Rear cornering stiffness (N/rad)
	mu = 1.4 # Friction coefficient (dimensionless)

	

	if option == 'Invariant B':
		# Compute linear A and B matrices based on the vehicle model
		B = np.array([
            [0,      1],         # [delta, accel]
            [Cf/m,   0],
            [lf*Cf/I,0]
        ])
	elif option == 'Variant B':
		B = np.array([
            [-Cf * np.sin(delta) / m,      1],   # [delta, accel]
            [ Cf * np.cos(delta) / m,      0],
            [ lf * Cf * np.cos(delta) / I, 0]
        ])
	else:
		raise ValueError("Invalid option for B matrix computation.")
	
	# Discretize B matrix using the sampling time
	B = Ts * B
	# Compute elements of the continuous-time A matrix
	A11 = -mu
	A12 = (np.sin(delta) * Cf) / (m * vx)
	A13 = (np.sin(delta) * Cf * lf) / (m * vx) + vy
	A22 = -(Cr + Cf * np.cos(delta)) / (m * vx)
	
	A23 = -(lf * Cf * np.cos(delta) - lr * Cr) / (m * vx) - vx
	A32 = -(lf * Cf * np.cos(delta) - lr * Cr) / (I * vx)
	A33 = -(lf**2 * Cf * np.cos(delta) + lr**2 * Cr) / (I * vx)

	# Construct the continuous-time A matrix
	A = np.array([
		[A11, A12, A13],
		[0, A22, A23],
		[0, A32, A33]
	])
	# Discretize A matrix using the sampling time
	nx = A.shape[0]
	A = Ts * A + np.eye(nx)
	return A, B