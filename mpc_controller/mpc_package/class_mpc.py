import numpy as np
import scipy.sparse as sparse
import osqp
import warnings
import matplotlib.pyplot as plt




class MPC_Controller_LPV:
    """
    Linear Parameter-Varying Model Predictive Controller (MPC) class.

    Attributes:
        nx (int): Number of states.
        nu (int): Number of inputs.
        Np (int): Prediction horizon for states.
        Nc (int): Prediction horizon for inputs (<= Np).
        x_ref (np.ndarray): Reference state vector.
        x0 (np.ndarray): Initial state vector.
        x_min, x_max (np.ndarray): State bounds.
        u_minus (np.ndarray): Previous input (for delta u penalty).
        u_ref (np.ndarray): Reference input.
        u_min, u_max (np.ndarray): Input bounds.
        Du_min, Du_max (np.ndarray): Input rate bounds.
        Qx, QxN, Qu, QDu (np.ndarray): Cost matrices for states, terminal state, input, and input rate.
        eps_feas (float): Soft constraint relaxation weight.
        eps_rel, eps_abs (float): Solver tolerances.
        SOFT_RELAX (bool): Enable soft constraint relaxation.
    """

    def __init__(self, nx, nu, Np=10, Nc=None, x_ref=None, x0=None, x_min=None, x_max=None,
                 u_minus=None, u_ref=None, u_min=None, u_max=None, Qx=None, QxN=None,
                 Qu=None, QDu=None, Du_min=None, Du_max=None, v_affine = None , 
                 esp_feas=1e6, esp_rel=1e-3, esp_abs=1e-3):

        # Dimensions
        self.nx = nx
        self.nu = nu

        # Prediction horizons
        if Np <= 1:
            raise ValueError("Np should be > 1!")
        self.Np = Np
        self.Nc = Nc if Nc is not None else Np
        if self.Nc > self.Np:
            raise ValueError("Nc should be <= Np!")
        if self.Nc < 1:
            raise ValueError("Nc should be >= 1!")

        # State and input references
        self.x_ref = np.zeros(nx) if x_ref is None else x_ref
        self.v_affine = np.zeros(nx) if v_affine is None else v_affine
        self.x0 = np.zeros(nx) if x0 is None else x0
        self.u_minus = np.zeros(nu) if u_minus is None else u_minus
        self.u_ref = np.zeros(nu) if u_ref is None else u_ref

        # State and input bounds
        self.x_min = -np.inf * np.ones((nx, 1)) if x_min is None else x_min
        self.x_max = np.inf * np.ones((nx, 1)) if x_max is None else x_max
        self.u_min = -np.inf * np.ones((nu, 1)) if u_min is None else u_min
        self.u_max = np.inf * np.ones((nu, 1)) if u_max is None else u_max
        self.Du_min = -np.inf * np.ones((nu, 1)) if Du_min is None else Du_min
        self.Du_max = np.inf * np.ones((nu, 1)) if Du_max is None else Du_max

        # Cost matrices
        self.Qx = np.zeros((nx, nx)) if Qx is None else Qx
        self.QxN = np.zeros((nx, nx)) if QxN is None else QxN
        self.Qu = np.zeros((nu, nu)) if Qu is None else Qu
        self.QDu = np.zeros((nu, nu)) if QDu is None else QDu

        # Soft constraint relaxation
        self.eps_feas = esp_feas
        self.Q_eps = self.eps_feas * sparse.eye(nx)

        # Solver tolerances
        self.eps_abs = esp_abs
        self.eps_rel = esp_rel

        # Internal flags and OSQP problem
        self.error_mpc = False
        self.SOFT_RELAX = True  # Set to True if soft relaxation is needed
        self.prob = osqp.OSQP()
        self.res = None
        self.P = None
        self.q = None
        self.A = None
        self.l = None
        self.u = None

        self.Du_ineq = True
    
    def setup_MPC(self, A_vec, B_vec, *, tube_int=False):
        """
        Prepares the QP matrices for the MPC problem.
        """
        #CHANGE: Use stage-wise tube bounds at setup as well as on updates.
        self._compute_QP_Matrices(A_vec, B_vec, self.x_ref, self.x0, self.u_ref,
                                  self.u_minus, tube_int=tube_int)

        #CHANGE: Reserve every LPV dynamics slot, including coefficients that start at zero.
        nx, nu = self.nx, self.nu
        coo = self.A.tocoo()
        rows, cols = [coo.row], [coo.col]
        for i in range(self.Np):
            state_rows = np.arange((i + 1) * nx, (i + 2) * nx)
            state_cols = np.arange(i * nx, (i + 1) * nx)
            input_start = (self.Np + 1) * nx + min(i, self.Nc - 1) * nu
            dynamics_cols = np.concatenate((state_cols, np.arange(input_start, input_start + nu)))
            rows.append(np.repeat(state_rows, dynamics_cols.size))
            cols.append(np.tile(dynamics_cols, nx))
        rows, cols = np.concatenate(rows), np.concatenate(cols)
        pattern = sparse.csc_matrix((np.ones(rows.size), (rows, cols)), shape=self.A.shape)
        self._ax_rows = pattern.indices.copy()
        self._ax_cols = np.repeat(np.arange(pattern.shape[1]), np.diff(pattern.indptr))
        pattern.data[:] = np.asarray(self.A[self._ax_rows, self._ax_cols]).ravel()

        self.prob.setup(P=self.P, q=self.q, A=pattern, l=self.l, u=self.u,
                        warm_start=True, verbose=False,
                        eps_abs=self.eps_abs, eps_rel=self.eps_rel)
        
        print("OSQP problem setup complete.")

    def update_MPC(self, A_vec, B_vec, x_ref, x0, u_ref , u_minus ,  x_min = None ,x_max = None , u_min = None , u_max = None , v_affine = None , tube_int = False):
        """
        Updates the MPC problem with new parameters and re-computes the QP matrices.
        """
        
      
        self.x_ref = x_ref
        
        self.v_affine = v_affine if u_ref is not None else self.v_affine
        self.x0 = x0
        self.u_ref = u_ref if u_ref is not None else self.u_ref
        self.u_minus = u_minus
        #CHANGE: Use the provided stage bounds for this update without overwriting defaults.
        x_min = x_min if tube_int and x_min is not None else self.x_min
        x_max = x_max if tube_int and x_max is not None else self.x_max
        u_min = u_min if tube_int and u_min is not None else self.u_min
        u_max = u_max if tube_int and u_max is not None else self.u_max

        # Recompute the QP matrices with the updated parameters
        self._update_QP_Matrices(A_vec, B_vec, self.x_ref, self.x0, self.u_ref, self.u_minus,
                                  x_min, x_max, u_min, u_max, tube_int, self.v_affine)
        
        #CHANGE: Update OSQP's numeric dynamics entries in setup CSC order.
        # P is constant across MPC steps; stage bounds are carried by l/u.
        Ax = np.asarray(self.A[self._ax_rows, self._ax_cols]).ravel()
        self.prob.update(Ax=Ax, q=self.q, l=self.l, u=self.u)

    def _compute_QP_Matrices(self, A_vec, B_vec, x_ref, x_0, u_ref, u_minus, *, tube_int=False):
        """
        Constructs the QP cost matrices (P, q) for the MPC problem.
        """
        Np, Nc, nx, nu = self.Np, self.Nc, self.nx, self.nu
        Qx, QxN, Qu, QDu, Q_eps = self.Qx, self.QxN, self.Qu, self.QDu, self.Q_eps
 

        # --- Cost matrix P ---
        # State cost for all prediction steps except terminal
        P_X = sparse.kron(sparse.eye(Np), Qx)
        # Terminal state cost
        P_XN = QxN
        # Input cost (with special handling for last input)
        iU = np.ones(Nc)
        iU[Nc-1] = Np - Nc + 1
        P_U = sparse.kron(sparse.diags(iU), Qu)
        # Input rate cost (finite difference structure)
        iDu = 2 * np.eye(Nc) - np.eye(Nc, k=-1) - np.eye(Nc, k=1)

  
        iDu[Nc-1, Nc-1] = 1
        P_DU = sparse.kron(iDu, QDu)
        # Total input cost

   
        P_Utot = P_DU + P_U

        # --- Linear cost vector q ---
        # State cost linear term
        
        
        if(len(x_ref.shape) == 3): 
         
            q_X = np.array([-Qx@x_ref[:,0,i] for i in range(Np)]).reshape(-1)
            q_XN = -QxN @ x_ref[:,0,-1]
        
        else:
            q_X = np.kron(np.ones(Np), -Qx @ x_ref)
            q_XN = -QxN @ x_ref
        
        
        # Terminal state cost linear term

        
        # Input cost linear term
        q_U = np.kron(iU, -Qu @ u_ref)
        # Input rate cost linear term (only first input penalized by u_minus)
        q_Du = np.zeros(Nc*nu)

     
        q_Du[:nu] = (-QDu @ u_minus.reshape(-1))

    
        # Total input cost linear term
        q_Utot = q_Du + q_U

        # --- Stack cost matrices ---
        if self.SOFT_RELAX:
            # Add slack variables for soft constraints
            P_eps = sparse.kron(sparse.eye(Np + 1), Q_eps)
            q_eps = np.zeros(nx * (Np + 1))
            self.P = sparse.block_diag([P_X, P_XN, P_Utot, P_eps], format='csc')
            self.q = np.hstack([q_X, q_XN, q_Utot, q_eps])
        else:
            self.P = sparse.block_diag([P_X, P_XN, P_Utot], format='csc')
            self.q = np.hstack([q_X, q_XN, q_Utot])

        print(f"Shape of P: {self.P.shape}")
        print(f"Shape of q: {self.q.shape}")

        # --- Equality constraints (Aeq, leq, ueq) ---

        # -- Define the system dynamics matrices -- 
        # Build a block diagonal matrix with -I blocks, used for state difference constraints
        # This helps in forming the equality constraints for the system dynamics
        A_xp1 = sparse.kron(sparse.eye(Np+1), -sparse.eye(nx))

        # Construct a temporary sparse matrix to hold the A matrices for each prediction step.
        # This matrix will be used to represent the effect of the system dynamics on the state trajectory.
        # For each time step i, place A_vec[i-1] at the correct block position.
        A_xp_temp = sparse.lil_matrix((nx * (Np + 1), nx * (Np + 1)))

        for i in range(1, Np + 1):
            # Place A_vec[i-1] in the block corresponding to the transition from x(i-1) to x(i)
            A_xp_temp[i*nx:(i+1)*nx, (i-1)*nx:i*nx] = A_vec[i-1]
        # Convert to CSR format for efficient arithmetic and slicing
        A_xp = A_xp_temp.tocsr()
        
        # Initialize the input matrix B_u, which maps the control inputs to the state trajectory.
        # The matrix has (Nc+1) blocks for states and Nc blocks for inputs.
        B_u = sparse.lil_matrix((nx*(Nc+1), nu*Nc))

        # Fill in the B_u matrix with the B matrices for each control interval.
        for i in range(1, Nc+1):
            # Place B_vec[i-1] in the block corresponding to the effect of u(i-1) on x(i)
            B_u[i*nx:(i+1)*nx, (i-1)*nu:i*nu] = B_vec[i-1]

        # If the number of control intervals Nc is less than the prediction horizon Np,
        # we need to extend B_u to account for the remaining prediction steps.
        if Nc < Np:
            # Create a temporary matrix to hold the additional B blocks
            B_u_temp2 = sparse.lil_matrix((nx*(Np-Nc), nu*Nc))
            for j in range(Np - Nc):
                # For each extra prediction step, place the corresponding B matrix at the end columns
                # This handles the case where the last input is held constant after Nc steps
                B_u_temp2[j*nx : (j+1)*nx, -(Nc-1)*nu:] = B_vec[j+Nc]
            # Stack the additional rows to the original B_u matrix
            B_u = sparse.vstack([B_u, B_u_temp2])

        # --- Lower and upper bounds for equality constraints ---
        leq = np.zeros(nx * (Np + 1))
        # The first state x(0) is constrained to be equal to the initial state x_0.
        
        leq[:nx] = -x_0 # Initial state constraint
        for i in range(1,Np+1): 
            print(self.v_affine.shape)
            leq[nx*i:(nx*(i+1))] = -self.v_affine
        
        ueq = leq.copy()  
            

                        
        
        
        Aeq = sparse.hstack([A_xp1+A_xp, B_u], format='csc')
        
        # Add the soft constraint relaxation term if enabled
        if self.SOFT_RELAX:
            # Append the slack variables to the equality constraints
            Aeq = sparse.hstack([Aeq, sparse.csc_matrix((Aeq.shape[0], nx * (Np + 1)))])
        
        # --- Inequality constraints (Aineq, lineq, uineq) ---
        # State bounds inequality constraints
        Aineq_x = sparse.hstack([sparse.eye((Np+1)*nx),  sparse.csc_matrix((nx * (Np + 1), nu * Nc))])
        if self.SOFT_RELAX:
            # If soft relaxation is enabled, add slack variables to the state bounds
            Aineq_x = sparse.hstack([Aineq_x , sparse.eye(nx * (Np + 1))])
        #CHANGE: The tube supplies flattened bounds in stage-major order.
        lineq_x = self.x_min if tube_int else np.kron(np.ones(Np + 1), self.x_min)
        uineq_x = self.x_max if tube_int else np.kron(np.ones(Np + 1), self.x_max)
        # Input bounds inequality constraints
        Aineq_u = sparse.hstack([sparse.csc_matrix((Nc * nu, (Np + 1) * nx)), sparse.eye(Nc * nu)])
        if self.SOFT_RELAX:
            # If soft relaxation is enabled, add slack variables to the input bounds
            Aineq_u = sparse.hstack([Aineq_u, sparse.csc_matrix((Aineq_u.shape[0], nx * (Np + 1)))])
        lineq_u = self.u_min if tube_int else np.kron(np.ones(Nc), self.u_min)
        uineq_u = self.u_max if tube_int else np.kron(np.ones(Nc), self.u_max)

        # Input rate bounds inequality constraints
        # If input rate constraints are enabled, construct the corresponding matrices
        if self.Du_ineq: 
            Aineq_Du = sparse.vstack([
                sparse.hstack([np.zeros((nu , (Np + 1) * nx)), sparse.eye(nu), np.zeros((nu, nu * (Nc - 1)))]), #Constraints input rate u(0)-u(-1) and use it for the bound
                sparse.hstack([np.zeros((nu * Nc , (Np + 1) * nx)), -sparse.eye(nu*Nc) + sparse.eye(nu*Nc, k=1)]) #Constraints input rate u(kk)-u(k-1) and use it for the bound
            ])

            if self.SOFT_RELAX:
                # If soft relaxation is enabled, add slack variables to the input rate bounds
                Aineq_Du = sparse.hstack([Aineq_Du, sparse.csc_matrix((Aineq_Du.shape[0], nx * (Np + 1)))])

            lineq_Du = np.kron(np.ones(Nc+1), self.Du_min)
            uineq_Du = np.kron(np.ones(Nc+1), self.Du_max)

            lineq_Du[0:nu] += u_minus.reshape(-1)  # Initial input rate constraint minimum bound
            uineq_Du[0:nu] += u_minus.reshape(-1) # Initial input rate constraint maximum bound


        # Stack all inequality constraints
            Aineq = sparse.vstack([Aineq_x, Aineq_u , Aineq_Du], format='csc')
            lineq = np.hstack([lineq_x, lineq_u, lineq_Du])
            uineq = np.hstack([uineq_x, uineq_u, uineq_Du])
        # If input rate constraints are not enabled, we only stack the state and input inequality constraints
        else:
            Aineq = sparse.vstack([Aineq_x, Aineq_u ], format='csc')
            lineq = np.hstack([lineq_x, lineq_u])
            uineq = np.hstack([uineq_x, uineq_u]) 

       
        # Set the OSQP problem matrices
        self.A = sparse.vstack([Aeq, Aineq], format='csc')
        self.l = np.hstack([leq, lineq])
        self.u = np.hstack([ueq, uineq])
        print("OSQP problem matrices size :")
        print(f"Shape of P: {self.P.shape}")
        print(f"Shape of q: {self.q.shape}")
        print(f"Shape of Aeq: {Aeq.shape}")
        print(f"Shape of Aineq: {Aineq.shape}")
        print(f"Shape of leq: {self.l.shape}")
        print(f"Shape of ueq: {self.u.shape}")
        print(f"Shape of A: {self.A.shape}")
        print(f"Shape of l: {self.l.shape}")
        print(f"Shape of u: {self.u.shape}")

    def _update_QP_Matrices(self, A_vec, B_vec, x_ref, x_0, u_ref, u_minus , x_min , x_max , u_min , u_max , tube_int , v_affine):
        """
        Updates the QP matrices based on new references and initial state.
        """
        Np, Nc, nx, nu = self.Np, self.Nc, self.nx, self.nu
        Qx, QxN, Qu, QDu, Q_eps = self.Qx, self.QxN, self.Qu, self.QDu, self.Q_eps


        
        # As the linear vector depend s on the references, we need to recompute the QP matrices
        # --- Linear cost vector q ---
        # State cost linear term and Terminal state cost linear term

        if(len(x_ref.shape) == 3): 
         
            q_X = np.array([-Qx@x_ref[:,0,i] for i in range(Np)]).reshape(-1)
            q_XN = -QxN @ x_ref[:,0,-1]
        
        else:
            q_X = np.kron(np.ones(Np), -Qx @ x_ref)
            q_XN = -QxN @ x_ref
       

        # Input cost linear term
        iU = np.ones(Nc)
        iU[Nc-1] = Np - Nc + 1
        q_U = np.kron(iU, -Qu @ u_ref)
        # Input rate cost linear term (only first input penalized by u_minus)
        q_Du = np.zeros(Nc*nu)

        q_Du[:nu] = (-QDu @ u_minus.reshape(-1))[:nu]

  
    
        q_Utot = q_Du + q_U
        # --- Stack cost matrices ---
        if self.SOFT_RELAX:
            # Add slack variables for soft constraints
            q_eps = np.zeros(nx * (Np + 1))
            self.q = np.hstack([q_X, q_XN, q_Utot, q_eps])
        else:
            self.q = np.hstack([q_X, q_XN, q_Utot])
        
        # --- Equality constraints (Aeq, leq, ueq) ---

        # -- Define the system dynamics matrices -- 
        
        # Build a block diagonal matrix with -I blocks, used for state difference constraints
        # This helps in forming the equality constraints for the system dynamics
        A_xp1 = sparse.kron(sparse.eye(Np+1), -sparse.eye(nx))

        # Update the temporary sparse matrix to hold the A matrices for each prediction step.
        A_xp_temp = sparse.lil_matrix((nx * (Np + 1), nx * (Np + 1)))

        for i in range(1, Np + 1):
            # Place A_vec[i-1] in the block corresponding to the transition from x(i-1) to x(i)
            A_xp_temp[i*nx:(i+1)*nx, (i-1)*nx:i*nx] = A_vec[i-1]
        # Convert to CSR format for efficient arithmetic and slicing
        A_xp = A_xp_temp.tocsr()
        
        # Update the input matrix B_u, which maps the control inputs to the state trajectory.
        # The matrix has (Nc+1)*nx blocks for states and Nc blocks for inputs.
        B_u = sparse.lil_matrix((nx*(Nc+1), nu*Nc))

        # Fill in the B_u matrix with the B matrices for each control interval.
        for i in range(1, Nc+1):
            # Place B_vec[i-1] in the block corresponding to the effect of u(i-1) on x(i)
            B_u[i*nx:(i+1)*nx, (i-1)*nu:i*nu] = B_vec[i-1]

        # If the number of control intervals Nc is less than the prediction horizon Np,
        # we need to extend B_u to account for the remaining prediction steps.
        if Nc < Np:
            # Create a temporary matrix to hold the additional B blocks
            B_u_temp2 = sparse.lil_matrix((nx*(Np-Nc), nu*Nc))
            for j in range(Np - Nc):
                # For each extra prediction step, place the corresponding B matrix at the end columns
                # This handles the case where the last input is held constant after Nc steps
                B_u_temp2[j*nx : (j+1)*nx, -(Nc-1)*nu:] = B_vec[j+Nc]
            # Stack the additional rows to the original B_u matrix
            B_u = sparse.vstack([B_u, B_u_temp2])

        # --- Lower and upper bounds for equality constraints ---
        leq = np.zeros(nx * (Np + 1))
        # The first state x(0) is constrained to be equal to the initial state x_0.
        leq[:nx] = -x_0  # Initial state constraint
        ueq = leq.copy()  
            

        Aeq = sparse.hstack([A_xp1+A_xp, B_u], format='csc')

        # Add the soft constraint relaxation term if enabled
        if self.SOFT_RELAX:
            # Append the slack variables to the equality constraints
            Aeq = sparse.hstack([Aeq, sparse.csc_matrix((Aeq.shape[0], nx * (Np + 1)))])
        
        # --- Inequality constraints (Aineq, lineq, uineq) ---
        # State bounds inequality constraints
        Aineq_x = sparse.hstack([sparse.eye((Np+1)*nx),  sparse.csc_matrix((nx * (Np + 1), nu * Nc))])
        if self.SOFT_RELAX:
            # If soft relaxation is enabled, add slack variables to the state bounds
            Aineq_x = sparse.hstack([Aineq_x , sparse.eye(nx * (Np + 1))])

        if not(tube_int): 

            lineq_x = np.kron(np.ones(Np + 1), x_min)
            uineq_x = np.kron(np.ones(Np + 1), x_max)
        else:
            lineq_x = x_min
            uineq_x = x_max
        # Input bounds inequality constraints
        Aineq_u = sparse.hstack([sparse.csc_matrix((Nc * nu, (Np + 1) * nx)), sparse.eye(Nc * nu)])
        if self.SOFT_RELAX:
            # If soft relaxation is enabled, add slack variables to the input bounds
            Aineq_u = sparse.hstack([Aineq_u, sparse.csc_matrix((Aineq_u.shape[0], nx * (Np + 1)))])

        if not(tube_int):
            lineq_u = np.kron(np.ones(Nc), u_min)
            uineq_u = np.kron(np.ones(Nc), u_max)
        else:
            lineq_u = u_min
            uineq_u = u_max


        # Input rate bounds inequality constraints
        # If input rate constraints are enabled, construct the corresponding matrices
        if self.Du_ineq:
            Aineq_Du = sparse.vstack([
                sparse.hstack([np.zeros((nu , (Np + 1) * nx)), sparse.eye(nu), np.zeros((nu, nu * (Nc - 1)))]), #Constraints input rate u(0)-u(-1) and use it for the bound
                sparse.hstack([np.zeros((nu * Nc , (Np + 1) * nx)), -sparse.eye(nu*Nc) + sparse.eye(nu*Nc, k=1)]) #Constraints input rate u(0)-u(-1) and use it for the bound
            ])

            if self.SOFT_RELAX:
                # If soft relaxation is enabled, add slack variables to the input rate bounds
                Aineq_Du = sparse.hstack([Aineq_Du, sparse.csc_matrix((Aineq_Du.shape[0], nx * (Np + 1)))])

            lineq_Du = np.kron(np.ones(Nc+1), self.Du_min)
            uineq_Du = np.kron(np.ones(Nc+1), self.Du_max)

    
            lineq_Du[0:nu] += u_minus  # Initial input rate constraint minimum bound
            uineq_Du[0:nu] += u_minus  # Initial input rate constraint maximum bound

            Aineq = sparse.vstack([Aineq_x, Aineq_u , Aineq_Du], format='csc') #We stack the inequality constraints for states, inputs, and input rates
            # Stack the lower and upper bounds for the inequality constraints
            # This combines the lower and upper bounds for states, inputs, and input rates
            lineq = np.hstack([lineq_x, lineq_u, lineq_Du])
            uineq = np.hstack([uineq_x, uineq_u, uineq_Du])
        # If input rate constraints are not enabled, we only stack the state and input inequality constraints
        else:

            Aineq = sparse.vstack([Aineq_x, Aineq_u ], format='csc')
            lineq = np.hstack([lineq_x, lineq_u])
            uineq = np.hstack([uineq_x, uineq_u])
       
       
    
        # Set the OSQP problem matrices
        self.A = sparse.vstack([Aeq, Aineq], format='csc')
        self.l = np.hstack([leq, lineq])
        self.u = np.hstack([ueq, uineq])

    def  solve(self):
        """
        Solves the MPC problem using OSQP.
        Returns the optimal control input and state trajectory.
        """
        
        # Solve the QP problem
        self.res = self.prob.solve()

        if self.res.info.status != 'solved':
            warnings.warn(f"OSQP did not solve the problem: {self.res.info.status}")
    
    def  output(self):
        """
        Returns the optimal control input and state trajectory.
        """
        
        Nc = self.Nc
        Np = self.Np
        nx = self.nx
        nu = self.nu


        info = {}
        if self.res.info.status == 'solved':

            uMPC = self.res.x[(Np + 1) * nx: (Np + 1) * nx +nu]
        else:
            uMPC = np.zeros(nu)

        # Extract the state trajectory from the solution
        seq_X = self.res.x[0:(Np + 1) * nx]
        seq_X = seq_X.reshape(-1,nx)
        info['x_trajectory'] = seq_X

        #Extract the input trajectory from the solution
        seq_U = self.res.x[(Np + 1) * nx: (Np + 1) * nx + Nc * nu]
        seq_U = seq_U.reshape(-1, nu)
        info['u_trajectory'] = seq_U
        # Extract the slack variables if soft relaxation is enabled
        if self.SOFT_RELAX:
            seq_eps = self.res.x[(Np + 1) * nx + Nc * nu:]
            seq_eps = seq_eps.reshape(-1, nx)
            info['eps_trajectory'] = seq_eps

        info['status'] = self.res.info.status


        return info , uMPC


        



        
           
        


           
if __name__ == '__main__':
    

    # System parameters
    Ts = 0.2  # Sampling time (s)
    M = 2     # Mass (kg)
    b = 0.3   # Friction coefficient (N*s/m)

    # Continuous-time system matrices
    Ac = np.array([[0.0, 1.0], [0, -b/M]])
    Bc = np.array([[0.0], [1/M]])
    nx, nu = Bc.shape

    # Discretize system (Euler)
    Ad = np.eye(nx) + Ac * Ts
    Bd = Bc * Ts

    # Reference and constraints
    xref = np.array([7.0, 1.2])
    uref = np.array([0.0])
    uminus1 = np.array([0.0])
    xmin = np.array([-10, -10.0])
    xmax = np.array([7.0, 10.0])
    umin = np.array([-1.2])
    umax = np.array([1.2])
    Dumin = np.array([-0.2])
    Dumax = np.array([0.2])

    # Cost matrices
    Qx = 1*sparse.diags([0.5, 0.1])
    QxN = 1*sparse.diags([0.5, 0.1])
    Qu = 2 * sparse.eye(1)
    QDu = 20.0 * sparse.eye(1)

    # Initial state
    x0 = np.array([0.1, 0.2])

    # Horizons
    Np = 25
    Nc =  10

    l = []
    bl = []
    for i in range(Np): 

        l.append(Ad)
        bl.append(Bd)


    # Instantiate MPC controller
    K = MPC_Controller_LPV(
        nx, nu, Np=Np, Nc=Nc, x_ref=xref, x0=x0, x_min=xmin, x_max=xmax,
        u_minus=uminus1, u_ref=uref, u_min=umin, u_max=umax,
        Qx=Qx, QxN=QxN, Qu=Qu, QDu=QDu,
        Du_min=Dumin, Du_max=Dumax,
        esp_feas=1e6, esp_rel=1e-3, esp_abs=1e-3 , x0_optimal=False
    )


    K.setup_MPC(l, bl)

    # Solve the MPC problem
    [nx , nu] = K.nx, K.nu
    len_sim = 50
    nsim = int(len_sim/Ts)
    x = np.zeros((nsim, nx))
    u = np.zeros((nsim, nu))
    
    tsim = np.arange(0,nsim) * Ts

    x_step = x0.copy()
    for i in range(nsim):

        
        K.solve()
        info, uMPC = K.output()

        x_step = Ad @ x_step + Bd @ uMPC
       
        x[i, :] = x_step
        u[i, :] = uMPC
        K.update_MPC(l, bl, x_ref=xref, x0=x_step, u_ref=uref, u_minus=uMPC)

        if i < 2 :
            print(info)

       
        

    # Plot results
    fig , ax = plt.subplots(4,1,figsize=(10,10))
    ax[0].plot(tsim , x[:,0] , label=r'$x_1(k)$')
    ax[0].plot(tsim , xref[0]*np.ones(len(tsim)) , ls='--' , c = 'red' ,label=r'$x_1 ~ ref$')
    ax[0].legend()
    ax[0].grid(True)
    ax[1].plot(tsim , x[:,1] , label=r'$x_2(k)$')
    ax[1].plot(tsim , xref[1]*np.ones(len(tsim)) , ls='--' , c = 'red' ,label=r'$x_2 ~ ref$')
    ax[1].legend()
    ax[1].grid(True)
    ax[2].plot(tsim , u[:] , label=r'$u_1(k)$')
    ax[2].legend()
    ax[2].grid(True)
    ax[3].plot(tsim[1:] , u[1:]-u[:-1] , label=r'$u_1(k)$')
    ax[3].legend()
    ax[3].grid(True)

plt.show()
