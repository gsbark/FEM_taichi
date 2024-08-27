import taichi as ti 
import numpy as np 
from FEM_utils import generate_sparse_ij,get_Sparse_Kss,generate_row_pointers,WriteVTK
from ti_utils import add,max_norm,normalized_norm

@ti.data_oriented 
class FEM_program:
    def __init__(self,Input):
        
        self.Input = Input
        n_DoFs = Input.n_free_DoFs

        #Initialize Taichi global fields
        self.F_residual = ti.ndarray(dtype=Input.dtype,shape=(n_DoFs))
        self.P_ext = ti.ndarray(dtype=Input.dtype,shape=(n_DoFs))
        self.U_disp = ti.ndarray(dtype=Input.dtype,shape=(n_DoFs)) 
        self.F_int = ti.field(dtype=Input.dtype,shape=(n_DoFs))
        self.Kss_U = ti.field(dtype=Input.dtype,shape=(n_DoFs))    
        self.load_dof = ti.field(dtype=ti.i32,shape=(len(Input.controlled_dof)))
        
        #Rearrange Fields
        self.all_dofs_index_ti = ti.field(dtype=ti.i32,shape=(len(Input.all_dofs_map)))
        self.dofs_bool  = ti.field(dtype=ti.i32,shape=(Input.Domain['Elements_DoFs'].shape))
        
        if Input.Kinematic_constraints == 1:
            self.Kin_con = ti.field(dtype=ti.i32,shape=Input.kinem_constraints.shape)
        self.Load_from_Input()
        
        #Initialize Solver
        if Input.Solution_algorithm['solver_type']=='Direct':
            '''
            Taichi direct solver, taichi sparse matrix datastructure 
            (need to specify the max number of triplets)
            '''
            self.K = ti.linalg.SparseMatrixBuilder(n_DoFs,n_DoFs,max_num_triplets=n_DoFs*2000,dtype=Input.dtype) #Adjust the max_num_triplets
            if Input.Kinematic_constraints == 1:
                #Create KSS sparse (penalty method for fixed nodes)
                C_mat = ti.linalg.SparseMatrixBuilder(Input.kinem_constraints.shape[1],n_DoFs,max_num_triplets=Input.kinem_constraints.shape[1]*n_DoFs,dtype=Input.dtype) 
                self.Assemble_Cmat_sparse(C_mat)
                Con_mat = C_mat.build()
                self.KSS = 1e8*Con_mat.transpose()@Con_mat

        elif Input.Solution_algorithm['solver_type']=='CG':
            '''
            CG solver with sparse matrix CRS (row pointers, column indices)
            3 sets of pointers and C_indices (K_stiff,KSS,K_total)
            '''
            #TODO : generate sparse faster !!!!
            Kstif_ij = np.unique(generate_sparse_ij(Input.Domain['Elements_DoFs'],Input.Dofs_boolean_mask,Input.all_dofs_map),axis=0).astype(np.int32)
            #Constrainted matrix
            if Input.Kinematic_constraints == 1: 
                KSS_ij,KSS_val = get_Sparse_Kss(Input.kinem_constraints,n_DoFs)
                Ktot_ij =  np.unique(np.concatenate((Kstif_ij,KSS_ij)),axis=0)
                self.KSS_ij= ti.Vector.field(2,ti.i32,shape=KSS_ij.shape[0])
                self.KSS_val = ti.field(dtype=Input.dtype,shape=KSS_val.shape[0])
                self.KSS_ij.from_numpy(KSS_ij)
                self.KSS_val.from_numpy(KSS_val)
            else:
                Ktot_ij = Kstif_ij
            Ktot_pointers,Ktot_sort_ij = generate_row_pointers(Ktot_ij)
            self.Ktot_Jindex = ti.field(ti.i32,shape=Ktot_sort_ij.shape[0])
            self.Ktot_pointers = ti.field(ti.i32,shape=Ktot_pointers.shape[0])
            self.K_stiff = ti.field(dtype=Input.dtype,shape=Ktot_sort_ij.shape[0])
    
            self.Ktot_pointers.from_numpy(Ktot_pointers)
            self.Ktot_Jindex.from_numpy(Ktot_sort_ij[:,1])
            self.Input.Solution_alg.Initialize_CG_solver(self.Ktot_Jindex,self.Ktot_pointers,self.P_ext.shape[0])

        #Initialize PF fields
        if Input.Analysis_type==4 or Input.Analysis_type==3:
            '''
            Only direct solver support for phase field analysis
            '''
            self.K_PF = ti.linalg.SparseMatrixBuilder(Input.num_nodes, Input.num_nodes, max_num_triplets=Input.num_nodes*1000,dtype=Input.dtype) 
            self.F_PF = ti.ndarray(dtype=Input.dtype,shape=(Input.num_nodes))
            self.C_PF = ti.ndarray(dtype=Input.dtype,shape=(Input.num_nodes))

    def Load_from_Input(self):
        self.load_dof.from_numpy(self.Input.controlled_dof)
        self.dofs_bool.from_numpy(self.Input.Dofs_boolean_mask)
        self.all_dofs_index_ti.from_numpy(self.Input.all_dofs_map)
        if self.Input.Kinematic_constraints == 1:
            self.Kin_con.from_numpy(self.Input.kinem_constraints)
    #--------------------------------------------------------------------------------

    #For taichi direct solver (TODO:maybe reduce the loop K_stiff)
    #Taichi sparse matrix builder (precision float64 for CPU, float32 for CUDA)
    #--------------------------------------------------------------------------------
    @ti.kernel
    def Assemble_Cmat_sparse(self,B: ti.types.sparse_matrix_builder()): #type:ignore
        for i in range(self.Input.kinem_constraints.shape[1]):
            ref = self.Kin_con[1,i]
            base =  self.Kin_con[0,i]
            B[i,ref] += 1
            if base!=ref:
                B[i,base] += -1

    @ti.kernel
    def Build_sparse_Kstiff(self,A: ti.types.sparse_matrix_builder()): #type:ignore
        #Calculate Global stiffness matrix
        for iel in range(self.Input.num_elements):
            #Get dofs of each element
            d = self.Input.El_type.ti_dofs[iel]
            #Calculate Local stiffness
            k_loc = self.Input.El_type.Local_stiffness(iel)
            #Assemble to global 
            for i in range((self.Input.dim)*self.Input.el_nodes):
                if self.dofs_bool[iel,i]==1: #free dof
                    idx_i = self.all_dofs_index_ti[d[i]]
                    for j in range((self.Input.dim)*self.Input.el_nodes):
                        if self.dofs_bool[iel,j]==1: #free dof
                            idx_j = self.all_dofs_index_ti[d[j]]
                            A[idx_i,idx_j] += k_loc[i,j]
    # Taichi CG Solver      
    #--------------------------------------------------------------------------------
    @ti.func
    def get_element(self,row:int,col:int,):
        start_idx = self.Ktot_pointers[row]
        end_idx = self.Ktot_pointers[row + 1]
        out = 0
        for i in range(start_idx, end_idx):
            if self.Ktot_Jindex[i] == col:
                out = i
                break
        return out
    
    @ti.kernel
    def Build_Kstiff(self):
        #Calculate Global stiffness matrix
        for iel in range(self.Input.num_elements):
            #Get dofs of each element
            d = self.Input.El_type.ti_dofs[iel]
            #Calculate Local stiffness
            k_loc = self.Input.El_type.Local_stiffness(iel)
            #Assemble to global 
            for i in range(self.Input.dim*self.Input.el_nodes):
                if self.dofs_bool[iel,i]==1: #free dof
                    idx_i = self.all_dofs_index_ti[d[i]]
                    for j in range(self.Input.dim*self.Input.el_nodes):
                        if self.dofs_bool[iel,j]==1: #free dof
                            idx_j = self.all_dofs_index_ti[d[j]]
                            idx = self.get_element(idx_i,idx_j)
                            self.K_stiff[idx]+=k_loc[i,j] 

    @ti.kernel
    def Add_Kstiff_KSS(self):
        for i in self.KSS_ij:
            idx = self.get_element(self.KSS_ij[i][0],self.KSS_ij[i][1])
            self.K_stiff[idx] += self.KSS_val[i]

    @ti.kernel
    def KSS_mult_U(self,U_disp:ti.types.ndarray()): #type: ignore
        self.Kss_U.fill(0.0)
        for i in self.KSS_ij:
            self.Kss_U[self.KSS_ij[i][0]] +=  self.KSS_val[i] * U_disp[self.KSS_ij[i][1]]
    #--------------------------------------------------------------------------------

    #FEM functions
    #--------------------------------------------------------------------------------        
    @ti.kernel
    def reset_step(self,P_ext:ti.types.ndarray(),F_res:ti.types.ndarray()): #type: ignore
        #Initialize
        P = 1
        for d in self.load_dof:
            P_ext[self.load_dof[d]] = P #* 1e-5
            F_res[self.load_dof[d]] = P #* 1e-5
        #Initialize NR iterations
        for i in range(self.Input.num_elements):
            for j in range(self.Input.el_nodes):
                    self.Input.El_type.stress_gp_prev[i,j] = self.Input.El_type.stress_gp_curr[i,j]
                    self.Input.El_type.strain_gp_prev[i,j] = self.Input.El_type.strain_gp_curr[i,j]
    
    @ti.kernel
    def Get_Fint(self,U_disp:ti.types.ndarray()): #type: ignore
        self.F_int.fill(0.0)
        #Calculate stresses/internal_forces
        for iel in range(self.Input.num_elements):
            d = self.Input.El_type.ti_dofs[iel]
            u_local = ti.Vector([0.0]*self.Input.dim*self.Input.el_nodes,dt=self.Input.dtype)
            for i in range(self.Input.dim*self.Input.el_nodes):
                if self.dofs_bool[iel,i]==1: #free dof
                    idx_i = self.all_dofs_index_ti[d[i]]
                    u_local[i] = U_disp[idx_i]
            Fint = self.Input.El_type.Get_fint(u_local,iel)
            for k in range(self.Input.dim*self.Input.el_nodes):
                if self.dofs_bool[iel,k]==1: #free dof
                    idx_k = self.all_dofs_index_ti[d[k]]
                    self.F_int[idx_k] += Fint[k]
    @ti.kernel
    def Extrapolate(self):
        for iel in range(self.Input.num_elements):
            self.Input.El_type.ExtrapolateGP(iel)

    @ti.kernel
    def Residual(self,Pext:ti.types.ndarray(),Res:ti.types.ndarray(),fact:float): #type: ignore
        for i in range(Pext.shape[0]):
            Res[i] = fact*Pext[i] - self.F_int[i] - self.Kss_U[i]

    def check_residual(self):
        fact = self.Input.Solution_alg.load_factor
        if self.Input.Kinematic_constraints == 1: 
            if self.Input.Solution_algorithm['solver_type']=='Direct':
                self.Kss_U.from_numpy(self.KSS@self.U_disp.to_numpy()) #Need fix 
            elif self.Input.Solution_algorithm['solver_type']=='CG':
                self.KSS_mult_U(self.U_disp)
            self.Residual(self.P_ext,self.F_residual,fact)
        else:
            self.Residual(self.P_ext,self.F_residual,fact)

    def Average_node_data(self):
        #Average node stress
        Av_stress= np.zeros(self.Input.Domain['Node_coords'].shape[0])
        np.add.at(Av_stress,
                  self.Input.Domain['Elements_connectivity'],
                  self.Input.El_type.node_stress.to_numpy())
        a,b=np.unique(self.Input.Domain['Elements_connectivity'].reshape(-1,1),
                      return_counts=True)
        Av_stress/=b
        return Av_stress
    #--------------------------------------------------------------------------------

    #Phase field functions
    #--------------------------------------------------------------------------------
    @ti.kernel
    def Initialize_PF(self):
        for iel in range(self.Input.num_elements):
            self.Input.El_type.Energy_update(iel)
            for i in range(self.Input.El_type.ngp):

                if self.Input.El_type.History_psi_mode1[iel,i] < self.Input.El_type.psi_plus_mode_1[iel,i]:
                    self.Input.El_type.History_psi_mode1[iel,i] = self.Input.El_type.psi_plus_mode_1[iel,i]

                if self.Input.El_type.History_psi_mode2[iel,i] < self.Input.El_type.psi_plus_mode_2[iel,i]:
                    self.Input.El_type.History_psi_mode2[iel,i] = self.Input.El_type.psi_plus_mode_2[iel,i]
    
    @ti.kernel
    def Assemble_PF(self,PF:ti.sparse_matrix_builder(),F_PF:ti.types.ndarray()): #type: ignore
    
        for iel in range(self.Input.num_elements):
            con = self.Input.El_type.ti_elements[iel]       
            k_loc,F = self.Input.El_type.PhaseField_K(iel)
            for i in range(self.Input.el_nodes):
                for j in range(self.Input.el_nodes):
                    PF[con[i],con[j]] += k_loc[i,j]
                F_PF[con[i]] += F[i]
    
    @ti.kernel
    def Update_PF_params(self,C_PF:ti.types.ndarray()): #type: ignore
        #Assign fracture parameters to gp 
        for iel in range(self.Input.num_elements):
            con = self.Input.El_type.ti_elements[iel]
            c_local = ti.Vector([0.0]*self.Input.el_nodes,dt=self.Input.dtype)
            for i in range(self.Input.el_nodes):
                c_local[i]= C_PF[con[i]]
            self.Input.El_type.Update_fracture_params(c_local,iel)

    def solve_PF(self,K_stiff:ti.template(),F_ext:ti.types.ndarray()): #type: ignore
        
        PF_solver = ti.linalg.SparseSolver(solver_type="LLT",dtype=self.Input.dtype)
        PF_solver.analyze_pattern(K_stiff)
        PF_solver.factorize(K_stiff)
        sol = PF_solver.solve(F_ext)
        return sol
    
    #--------------------------------------------------------------------------------
    def Run_FEM(self):
        inc = 0
        error_flag=0
        while inc< self.Input.Solution_alg.tot_increments:
            iter = 1
            self.F_residual.fill(0.0)
            self.U_disp.fill(0.0)
            self.reset_step(self.P_ext,self.F_residual)
            Res_norm,Res_norm_normal = 1e8,1e8
    
            if self.Input.Analysis_type==4 or self.Input.Analysis_type==3:
                self.F_PF.fill(0.0)
                self.Initialize_PF()
                self.Assemble_PF(self.K_PF,self.F_PF)
                K_stiff_PF = self.K_PF.build()
                C_PF = self.solve_PF(K_stiff_PF,self.F_PF)
                self.Update_PF_params(C_PF)
                print('Phase Field OK')
                
            while Res_norm > self.Input.Solution_alg.NR_tol and iter<self.Input.Solution_alg.max_iter:
                print('inc:',inc,'iter:',iter)
                if self.Input.Solution_algorithm['solver_type']=='CG':
                    self.K_stiff.fill(0.0)
                    self.Build_Kstiff()
                    if self.Input.Kinematic_constraints == 1: 
                        self.Add_Kstiff_KSS()

                elif self.Input.Solution_algorithm['solver_type']=='Direct':
                    self.Build_sparse_Kstiff(self.K)
                    self.K_stiff = self.K.build()
                    if self.Input.Kinematic_constraints == 1:  
                        self.K_stiff = self.K_stiff + self.KSS
                        
                if self.Input.Solution_algorithm['type']=='load':
                    U_d = self.Input.Solution_alg.Solve_system(self.K_stiff,self.F_residual,self.U_disp) 
                elif self.Input.Solution_algorithm['type']=='displacement':
                    U_d = self.Input.Solution_alg.Find_displacement(self.K_stiff,self.F_residual,iter)
                print('Solution OK')
                add(self.U_disp,U_d)
                #Calculate internal forces
                self.Get_Fint(self.U_disp)
                #Check Residual
                self.check_residual()
                Res_norm_new = max_norm(self.F_residual)
                if Res_norm_new >Res_norm:
                    print('N-R_diverging',Res_norm_new)
                    # break
                if np.isnan(Res_norm_new).any() or np.isinf(Res_norm_new).any():
                    error_flag=1
                    break
                Res_norm = Res_norm_new
                Res_norm_normal = normalized_norm(self.F_residual,self.P_ext)
                print('Residual norm:',Res_norm,Res_norm_normal,'\n',"-"*30)
                iter +=1

            self.Extrapolate()
            self.Input.Tot_U[self.Input.free_dofs] += self.U_disp.to_numpy()
            self.Input.Tot_Fext[self.Input.free_dofs] += self.Input.Solution_alg.load_factor*self.P_ext.to_numpy()
            self.Input.Plot_U.append(np.abs(self.Input.Tot_U[self.Input.controlled_dof_1[0]]))
            self.Input.Plot_F.append(self.Input.Tot_Fext[self.Input.controlled_dof_1[0]]*1000)

            Av_stress = self.Average_node_data()
            #Export field
            export_data = {'Vm_stress':Av_stress[:,None],
                           'Displacement':self.Input.Tot_U[1::self.Input.dim,None]}
            
            if self.Input.Analysis_type==4 or self.Input.Analysis_type==3:
                export_data['Phase_field']=C_PF.to_numpy()[:,None]

            #Export VTK file 
            # GaussPoints_export(self.Input.El_type.coords_gp.to_numpy(),
            #                    self.Input.El_type.stress_gp_curr.to_numpy()[:,:,1],
            #                    inc)
                
            WriteVTK(self.Input.Domain['Elements_connectivity'],
                     self.Input.Domain['Node_coords']+self.Input.Tot_U.reshape(-1,self.Input.dim),   
                     export_data,
                     file=f"./Analysis_res/vtk/{inc}.vtk")
                    #self.Input.Domain['Node_coords']+self.Input.Tot_U.reshape(-1,self.Input.dim),        
            inc +=1

        # Load/Displacement curve
        if error_flag==0:    
            np.savetxt(f'./Analysis_res/load_disp.csv',
                       np.stack((np.array(self.Input.Plot_U),np.array(np.abs(self.Input.Plot_F))),axis=1),
                       header='disp,force',delimiter=',',fmt="%.4e",comments='')