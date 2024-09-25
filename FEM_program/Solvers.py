import taichi as ti 
import numpy as np  
from time import time
from ti_utils import mult_array_scalar

@ti.data_oriented
class CG_solver:
    def __init__(self,index_ij:ti.template(),pointers:ti.template(), b:int,eps:float=1e-10): #type:ignore 

        self.index_ij = index_ij
        self.pointers = pointers
        self.x = ti.ndarray(dtype=ti.f64,shape=b)
        self.r = ti.field(dtype=ti.f64,shape=b)
        self.d = ti.field(dtype=ti.f64,shape=b)
        self.M = ti.field(dtype=ti.f64,shape=b) #Preconditioner
        self.Ad = ti.field(dtype=ti.f64,shape=b)
        self.eps = eps
       
    def initialize(self,A_mat:ti.template(),b:ti.template()): #type:ignore 
        self.x.fill(0.0)
        self.A = A_mat
        self.b = b 
        self.M_init()

    @ti.func
    def get_element(self,row:int,col:int,):
        start_idx = self.pointers[row]
        end_idx = self.pointers[row + 1]
        out = 0
        for i in range(start_idx, end_idx):
            if self.index_ij[i] == col:
                out = i
                break
        return out    

    @ti.kernel
    def M_init(self):  # initialize the precondition diagonal matrix
        for i in range(self.M.shape[0]):
            val=1.0
            ind = self.get_element(i,i)
            if self.A[ind]!=0:
                val = 1 / self.A[ind]  
            self.M[i] = val

    @ti.kernel
    def r_d_init(self,b:ti.types.ndarray()):  # initial residual r and direction d #type:ignore 
        for i in self.r:  # r0 = b - Ax0 = b
            self.r[i] = b[i]
        for i in self.d:
            self.d[i] = self.M[i] * self.r[i]  # d0 = M^(-1) * r
    
    @ti.kernel
    def compute_Ad(self):  # compute A multiple d
        self.Ad.fill(0.0)
        for i in range(self.pointers.shape[0]-1):
            start_idx = self.pointers[i]
            end_idx = self.pointers[i + 1]
            for j in range(start_idx,end_idx):
                self.Ad[i] += self.A[j] * self.d[self.index_ij[j]]
    @ti.kernel
    def compute_rMr(self) -> float:  
        rMr = 0.
        for i in self.r:
            rMr += self.r[i] * self.M[i] * self.r[i]
        return rMr
    
    @ti.kernel 
    def update_x_r(self,x:ti.types.ndarray(),alpha:float): #type:ignore 
        for j in x:
            x[j] += alpha * self.d[j]
            self.r[j] -= alpha * self.Ad[j]
            
    @ti.kernel 
    def update_d(self, beta: float):
        for j in self.d:
            self.d[j] = self.M[j] * self.r[j] + beta * self.d[j]
    
    @staticmethod
    @ti.kernel
    def dot_product(y: ti.template(), z: ti.template()) -> float: #type:ignore 
        res = 0.
        for i in y:
            res += y[i] * z[i]
        return res

    def solve(self):
        self.r_d_init(self.b)
        r0 = np.sqrt(self.dot_product(self.r, self.r))
        
        print("\033[32;1m the initial residual scale is {} \033[0m".format(r0))
        for i in range(self.b.shape[0]):  # CG will converge within at most b.shape[0] loops
            t0 = time()
            self.compute_Ad()
            rMr = self.compute_rMr()
            alpha = rMr / self.dot_product(self.d, self.Ad)
            self.update_x_r(self.x,alpha)
            beta = self.compute_rMr() / rMr
            self.update_d(beta)
            r_norm = np.sqrt(self.dot_product(self.r, self.r))
            t1 = time()

            if i % 100 == 0:
                print("\033[35;1m the {}-th loop, norm of residual is {}, in-loop time is {} s\033[0m".format(
                    i, r_norm, t1 - t0
                ))
            if r_norm < self.eps :  
                print("\033[35;1m the {}-th loop, norm of residual is {}, in-loop time is {} s\033[0m".format(
                    i, r_norm, t1 - t0
                ))
                break
        return self.x

class SolutionAlgorithm:
    def __init__(self,
                 control:str,
                 tolerance:float,
                 NR_max_iter:int,
                 tot_increments:int,
                 total_load:float,
                 dtype):
        
        self.control_type = control
        self.NR_tol = tolerance
        self.NR_max_iter = NR_max_iter
        self.tot_increments = tot_increments
        self.tot_load = total_load

        self.load_factor = 1
        self.tot_load_factor = 1
        self.dtype = dtype 
    
    def Initialize_CG_solver(self,index_ij:ti.template(),pointers:ti.template(),b:int): #type:ignore

        assert self.solver_type=='CG'
        self.CG = CG_solver(index_ij=index_ij,pointers=pointers,b=b)

    def Direct_solve(self,K_stiff:ti.types.template(),F_ext:ti.types.ndarray()): #type:ignore
        solver = ti.linalg.SparseSolver(solver_type="LLT",dtype=self.dtype)
        solver.analyze_pattern(K_stiff)
        solver.factorize(K_stiff)
        sol = solver.solve(F_ext)
        return sol
    
    def CG_solve(self,K_stiff:ti.types.template(),F_ext:ti.types.ndarray()):    #type:ignore 
        self.CG.initialize(K_stiff,F_ext)
        out = self.CG.solve()
        return out

class LoadControl(SolutionAlgorithm):
    def __init__(self,
                 type,
                 solver_type,
                 NR_tolerance,
                 NR_max_iter,
                 total_increments,
                 total_load,
                 dtype):
        super().__init__(type,NR_tolerance, NR_max_iter, total_increments, total_load,dtype)

        self.load_factor = total_load/total_increments
        self.solver_type = solver_type
    
    def Find_displacement(self,K_stiff:ti.template(),F_ext:ti.types.ndarray(),iter:int): #type:ignore 
        
        if self.solver_type=='Direct':
            sol = self.Direct_solve(K_stiff,F_ext)
        elif self.solver_type=='CG':
            sol = self.CG_solve(K_stiff,F_ext)
        return sol
    
class DisplacementControl(SolutionAlgorithm):
    def __init__(self,
                 control:str,
                 type:str,
                 NR_tolerance:float,
                 NR_max_iter:int,
                 total_increments:int, 
                 total_load:int,
                 controlled_dof:int,
                 dtype):
        
        super().__init__(control,NR_tolerance, NR_max_iter, total_increments, total_load,dtype)

        self.controlled_dof = controlled_dof
        self.target_disp = total_load/total_increments
        self.solver_type = type
    
    def Solve_system(self,K_stiff:ti.template(),F_ext:ti.types.ndarray()):  #type:ignore 
        
        if self.solver_type=='Direct':
            sol = self.Direct_solve(K_stiff,F_ext)
        elif self.solver_type=='CG':
            sol = self.CG_solve(K_stiff,F_ext)
        return sol    

    def Find_displacement(self,K_stiff:ti.template(),F_residual:ti.types.ndarray(),iter:int): #type:ignore 
        if iter ==1:
            #Solve linear system
            U_d = self.Solve_system(K_stiff,F_residual)
            #Find controlled dof value
            self.u_a1 = U_d[self.controlled_dof[0]]
            self.load_factor = self.target_disp/self.u_a1
            mult_array_scalar(U_d,self.load_factor)
        elif iter >1:
            ub = self.Solve_system(K_stiff,F_residual)
            u_b1 = ub[self.controlled_dof[0]]
            self.load_factor -= u_b1/self.u_a1 #Total incremental load factor
            F_residual[self.controlled_dof[0]] -= u_b1/self.u_a1   
            U_d = self.Solve_system(K_stiff,F_residual)
        return U_d
    

    