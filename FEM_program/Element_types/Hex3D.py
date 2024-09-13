import taichi as ti 

'''
TODO : ADD elastoplastic and ductile
'''

@ti.data_oriented
class HexElement:
    def __init__(self,E,n,ngp,elements_con,node_coords,dofs,dtype) -> None:
        
        self.dtype = dtype
        self.E = E                 #Young modulus
        self.n = n                 #Poisson ratio
        self.ngp = ngp             #Integration points
        self.G = E/(2*(1+n))       #Shear modulus
        self.K = E/(3*(1-2*n))     #Bulk modulus

        self.lamda = (E*n)/((1+n)*(1-2*n)) 
        self.num_elements = len(elements_con)

        #Structure
        self.ti_node = ti.Vector.field(3,dtype=self.dtype,shape=((node_coords.shape[0])))
        self.node_stress = ti.Vector.field(8,dtype=self.dtype,shape=(self.num_elements))

        self.ti_elements = ti.Vector.field(8,dtype=ti.i32,shape=((elements_con.shape[0]))) 
        self.ti_dofs =  ti.Vector.field(24,dtype=ti.i32,shape=((dofs.shape[0]))) 

        self.ti_node.from_numpy(node_coords)
        self.ti_elements.from_numpy(elements_con)
        self.ti_dofs.from_numpy(dofs)
        
        #Gauss points attributes
        self.coords_gp = ti.Vector.field(3,dtype=self.dtype,shape=(self.num_elements,8))
        self.stress_gp_prev = ti.Vector.field(7,dtype=self.dtype,shape=(self.num_elements,ngp)) #s_xx,s_yy,s_zz,s_xy,s_xz,s_yz,s_vm
        self.stress_gp_curr = ti.Vector.field(7,dtype=self.dtype,shape=(self.num_elements,ngp))
        
        self.strain_gp_prev = ti.Vector.field(7,dtype=self.dtype,shape=(self.num_elements,ngp)) #e_xx,e_yy,e_zz,g_xy,g_xz,g_yz,e_vm
        self.strain_gp_curr = ti.Vector.field(7,dtype=self.dtype,shape=(self.num_elements,ngp))

        #Functions for MatPoints
        self.B_mat = ti.Matrix.field(6,24,dtype=self.dtype,shape=(self.num_elements,ngp))
        self.BB_mat = ti.Matrix.field(3,8,dtype=self.dtype,shape=(self.num_elements,ngp))

        self.N_func = ti.Vector.field(8,dtype=self.dtype,shape=(self.num_elements,ngp))
        self.vol = ti.field(dtype=self.dtype,shape=(self.num_elements,ngp))
        #Initialize FEM functions
        self.Initialize()

    @ti.kernel
    def Initialize(self):
        '''
        Initialize Functions for GPs
        '''
        for iel in self.ti_elements:
            con = self.ti_elements[iel]
            coord_v = ti.Matrix([[0.0,0.0,0.0]]*8,dt=self.dtype)
            for i in range(8):
                coord_v[i,:] = self.ti_node[con[i]]
            self.Initialize_func(coord_v,iel)
    
    @ti.func
    def Initialize_func(self,
                        coords,
                        iel):
        
        gp,w = self.Gauss_quad()
        for i in range(self.ngp):
            eta = gp[i,0]
            ksi = gp[i,1]
            zeta = gp[i,2]
            N,dN = self.shape_fun(eta,ksi,zeta)
            B_mat,detJ,BB = self.get_B_mat(coords,dN)    

            self.coords_gp[iel,i] = N @ coords
            self.B_mat[iel,i] = B_mat
            self.N_func[iel,i] = N 
            self.BB_mat[iel,i]= BB
            self.vol[iel,i] = w[i]*detJ

    @ti.func
    def shape_fun(self,eta,ksi,zeta):
       
        N1 = 1/8*(1-ksi)*(1-eta)*(1-zeta)
        N2 = 1/8*(1+ksi)*(1-eta)*(1-zeta)
        N3 = 1/8*(1+ksi)*(1+eta)*(1-zeta)
        N4 = 1/8*(1-ksi)*(1+eta)*(1-zeta)
        N5 = 1/8*(1-ksi)*(1-eta)*(1+zeta)
        N6 = 1/8*(1+ksi)*(1-eta)*(1+zeta)
        N7 = 1/8*(1+ksi)*(1+eta)*(1+zeta)
        N8 = 1/8*(1-ksi)*(1+eta)*(1+zeta)

        N = ti.Vector([N1,N2,N3,N4,N5,N6,N7,N8],dt=self.dtype)

        dN = 1/8 * ti.Matrix([[-(1-eta)*(1-zeta),(1-eta)*(1-zeta),(1+eta)*(1-zeta),-(1+eta)*(1-zeta),
                               -(1-eta)*(1+zeta),(1-eta)*(1+zeta),(1+eta)*(1+zeta),-(1+eta)*(1+zeta)],
                              [-(1-ksi)*(1-zeta),-(1+ksi)*(1-zeta),(1+ksi)*(1-zeta),(1-ksi)*(1-zeta),
                               -(1-ksi)*(1+zeta),-(1+ksi)*(1+zeta),(1+ksi)*(1+zeta),(1-ksi)*(1+zeta)],                               
                               [-(1-ksi)*(1-eta),-(1+ksi)*(1-eta),-(1+ksi)*(1+eta),-(1-ksi)*(1+eta),
                                (1-ksi)*(1-eta),(1+ksi)*(1-eta),(1+ksi)*(1+eta),(1-ksi)*(1+eta)]],dt=self.dtype)

        return N,dN
    
    @ti.func
    def Gauss_quad(self):
        a = 1.0/ti.sqrt(3.0)
        g_loc = ti.Matrix([[-a, -a, -a],
                            [-a,  a, -a],
                            [a,   a, -a],
                            [a,  -a, -a],
                            [-a, -a, a],
                            [-a,  a, a],
                            [a,   a, a],
                            [a,  -a, a]],dt=self.dtype)
        
        w = ti.Vector([1,1,1,1,1,1,1,1],dt=self.dtype)
        return g_loc,w
    
    @ti.func
    def get_B_mat(self,coords,dN):

        #Calculate Jacobian
        J = dN@coords
        detJ = ti.math.determinant(J)
        # #InvJacobian
        inv_J = 1/detJ*ti.Matrix([[J[2,2]*J[1,1]-J[2,1]*J[1,2],-(J[2,2]*J[0,1]-J[2,1]*J[0,2]),J[1,2]*J[0,1]-J[1,1]*J[0,2]],
                                  [-(J[2,2]*J[1,0]-J[2,0]*J[1,2]),J[2,2]*J[0,0]-J[2,0]*J[0,2],-(J[1,2]*J[0,0]-J[1,0]*J[0,2])],
                                  [J[2,1]*J[1,0]-J[2,0]*J[1,1],-(J[2,1]*J[0,0]-J[2,0]*J[0,1]),J[1,1]*J[0,0]-J[1,0]*J[0,1]]],dt=self.dtype)
        BB = inv_J@dN
        
        B_mat = ti.Matrix([[BB[0,0],0,0,BB[0,1],0,0,BB[0,2],0,0,BB[0,3],0,0,BB[0,4],0,0,BB[0,5],0,0,BB[0,6],0,0,BB[0,7],0,0],
                           [0,BB[1,0],0,0,BB[1,1],0,0,BB[1,2],0,0,BB[1,3],0,0,BB[1,4],0,0,BB[1,5],0,0,BB[1,6],0,0,BB[1,7],0],
                           [0,0,BB[2,0],0,0,BB[2,1],0,0,BB[2,2],0,0,BB[2,3],0,0,BB[2,4],0,0,BB[2,5],0,0,BB[2,6],0,0,BB[2,7]],
                           [BB[1,0],BB[0,0],0,BB[1,1],BB[0,1],0,BB[1,2],BB[0,2],0,BB[1,3],BB[0,3],0,BB[1,4],BB[0,4],0,BB[1,5],BB[0,5],0,BB[1,6],BB[0,6],0,BB[1,7],BB[0,7],0],
                           [0,BB[2,0],BB[1,0],0,BB[2,1],BB[1,1],0,BB[2,2],BB[1,2],0,BB[2,3],BB[1,3],0,BB[2,4],BB[1,4],0,BB[2,5],BB[1,5],0,BB[2,6],BB[1,6],0,BB[2,7],BB[1,7]],
                           [BB[2,0],0,BB[0,0],BB[2,1],0,BB[0,1],BB[2,2],0,BB[0,2],BB[2,3],0,BB[0,3],BB[2,4],0,BB[0,4],BB[2,5],0,BB[0,5],BB[2,6],0,BB[0,6],BB[2,7],0,BB[0,7]]],dt=self.dtype)
        
        return B_mat,detJ,BB
    
    #Subclasses will overwrite this
    @ti.func
    def D_matrix(self):
        pass

    @ti.func
    def Stress_update(self):
        pass
    
    @ti.func
    def Calc_BT_D_B(self,B_mat,D_matrix):

        result = ti.Matrix.zero(dt=self.dtype,n=24,m=24)
    
        # Perform the calculation A^TBA
        for i in range(24):  # Iterate over the columns of A (transposed)
            for j in range(6):  # Iterate over the rows of B
                for k in range(6):  # Iterate over the columns of B
                    result[i][k] += B_mat[j][i] * D_matrix[j][k]  # Update the result

        return result

    @ti.func
    def Local_stiffness(self,iel):
        k_local = ti.Matrix.zero(dt=self.dtype,n=24,m=24)
        for i in range(self.ngp):

            D_matrix = self.D_matrix(iel,i)
            B_mat = self.B_mat[iel,i]
            vol = self.vol[iel,i]
            k_local += B_mat.transpose() @ D_matrix @ B_mat *vol
        return k_local

    @ti.func
    def Get_fint(self,U_inc,iel):
        F_int = ti.Vector.zero(dt=self.dtype,n=24)
        for i in range(self.ngp):

            B_mat = self.B_mat[iel,i]
            vol = self.vol[iel,i]
            inc_strain = B_mat @ U_inc
           
            self.strain_gp_curr[iel,i] = ti.Vector([inc_strain[0],inc_strain[1],inc_strain[2],inc_strain[3],inc_strain[4],inc_strain[5], 0.0]) + self.strain_gp_prev[iel,i]
            #Get stress
            self.Stress_update(iel,i)        
            #internal_forces
            F_int += vol * B_mat.transpose() @ (self.stress_gp_curr[iel,i]-self.stress_gp_prev[iel,i])[:6]
        return F_int
    
    @ti.func
    def ExtrapolateGP(self,iel):
        gp_val = ti.Vector([self.stress_gp_curr[iel,i][6] for i in range(self.ngp)])
        a = ti.sqrt(3.0)
        natCoords = ti.Matrix([[-a, -a, -a],
                               [-a,  a, -a],
                               [a,   a, -a],
                               [a,  -a, -a],
                               [-a, -a, a],
                               [-a,  a, a],
                               [a,   a, a],
                               [a,  -a, a]],dt=self.dtype)
        
        for node in range(8):
            self.node_stress[iel][node] = ((self.shape_fun(natCoords[node,0],natCoords[node,1],natCoords[node,2])[0])*gp_val).sum()
        pass
    
@ti.data_oriented
class Elastic_3D(HexElement):
    def __init__(self,E,n,ngp,elements_con,node_coords,dofs,dtype):
        super().__init__(E,n,ngp,elements_con,node_coords,dofs,dtype)

    @ti.func
    def D_matrix(self,iel,i):
        
        c1 = self.E/((1+self.n)*(1-2*self.n))

        D = c1*ti.Matrix([[1-self.n,self.n,self.n,0,0,0],
                          [self.n,1-self.n,self.n,0,0,0],
                          [self.n,self.n,1-self.n,0,0,0],
                          [0,0,0,(1-2*self.n)/2,0,0],
                          [0,0,0,0,(1-2*self.n)/2,0],
                          [0,0,0,0,0,(1-2*self.n)/2]],dt=self.dtype)  
        return D
    
    @ti.func
    def Stress_update(self,iel,i):

        strain = self.strain_gp_curr[iel,i]
        K = self.K
        G = self.G
        #Volumentric Strain
        e_vol = strain[0] + strain[1] + strain[2]
        P = K * e_vol
        #Deviatoric Strain
        e_dev_1 = strain[0] - e_vol/3
        e_dev_2 = strain[1] - e_vol/3
        e_dev_3 = strain[2] - e_vol/3
        gamma_12 = strain[3]/2
        gamma_23 = strain[4]/2
        gamma_31 = strain[5]/2

        stress_11 = 2*G*e_dev_1 + P 
        stress_22 = 2*G*e_dev_2 + P 
        stress_33 = 2*G*e_dev_3 + P 
        stress_12 = 2*G*gamma_12
        stress_23 = 2*G*gamma_23
        stress_31 = 2*G*gamma_31

        stress_vm = ti.sqrt((2*G*e_dev_1)**2+
                            (2*G*e_dev_2)**2+
                            (2*G*e_dev_3 )**2+
                            2*(2*G*gamma_12)**2+
                            2*(2*G*gamma_23)**2+
                            2*(2*G*gamma_31)**2) 

        ST = ti.Vector([stress_11,stress_22,stress_33,stress_12,stress_23,stress_31,stress_vm],dt=self.dtype)
        self.stress_gp_curr[iel,i] = ST
        pass

@ti.data_oriented
class Transverse_Elastic_3D(HexElement):
    def __init__(self,E,n,ngp,elements_con,node_coords,dofs,dtype):
        super().__init__(E,n,ngp,elements_con,node_coords,dofs,dtype)

    @ti.func
    def D_matrix(self,iel,i):
        
        lamda_1 = self.lamda
        alpha = 0
        beta = 0
        mu1 = self.G
        mu2 = mu1/2

        # c1 = self.E/((1+self.n)*(1-2*self.n))

        # D = c1*ti.Matrix([[1-self.n,self.n,self.n,0,0,0],
        #                   [self.n,1-self.n,self.n,0,0,0],
        #                   [self.n,self.n,1-self.n,0,0,0],
        #                   [0,0,0,(1-2*self.n)/2,0,0],
        #                   [0,0,0,0,(1-2*self.n)/2,0],
        #                   [0,0,0,0,0,(1-2*self.n)/2]],dt=self.dtype)  
        
        D = ti.Matrix([[lamda_1+2*mu2,lamda_1+alpha,lamda_1,0,0,0],
                       [lamda_1+alpha,lamda_1+2*alpha+4*mu1-2*mu2+beta,lamda_1+alpha,0,0,0],
                       [lamda_1,lamda_1+alpha,lamda_1+2*mu2,0,0,0],
                       [0,0,0,mu1,0,0],
                       [0,0,0,0,mu1,0],
                       [0,0,0,0,0,mu2]],dt=self.dtype) 
        return D
    
    @ti.func
    def Stress_update(self,iel,i):

        strain = self.strain_gp_curr[iel,i]
        Ce = self.D_matrix(iel,i)
        stress = Ce@strain[:6]
        #Volumentric Stress
        s_vol = stress[0] + stress[1] + stress[2] 
        # #Deviatoric Stress
        s_dev_1 = stress[0] - s_vol/3
        s_dev_2 = stress[1] - s_vol/3
        s_dev_3 = stress[2] - s_vol/3
        s_dev_12 = stress[3]
        s_dev_23 = stress[4]
        s_dev_31 = stress[5]

        stress_vm = ti.sqrt((s_dev_1)**2+
                            (s_dev_2)**2+
                            (s_dev_3)**2+
                            2*(s_dev_12)**2+
                            2*(s_dev_23)**2+
                            2*(s_dev_31)**2) 
        ST = ti.Vector([stress[0],stress[1],stress[2],stress[3],stress[4],stress[5],stress_vm],dt=self.dtype)
        self.stress_gp_curr[iel,i] = ST
        pass    

@ti.data_oriented
class BrittleFracture_3D(HexElement):
    def __init__(self,E,n,l0,Gc,k,Fc,ngp,elements_con,node_coords,dofs,dtype):
        super().__init__(E,n,ngp,elements_con,node_coords,dofs,dtype) 
        
        self.l0 = l0
        self.Gc = Gc
        self.k = k
        self.Fc = Fc
        

        self.thres_psi_plus = ti.field(dtype=self.dtype,shape=(self.num_elements,ngp)) 
        self.psi_plus = ti.field(dtype=self.dtype,shape=(self.num_elements,ngp))        # Positive part of energy
        self.psi_total = ti.field(dtype=self.dtype,shape=(self.num_elements,ngp))       # Total energy
        self.History_psi = ti.field(dtype=self.dtype,shape=(self.num_elements,ngp))     # History variable
      
        self.c_PF = ti.field(dtype=self.dtype,shape=(self.num_elements,ngp))            #Phase field variable
        self.g_PF = ti.field(dtype=self.dtype,shape=(self.num_elements,ngp))            #Degradation variable
        self.g_PF.fill(1.0)
    
    @ti.func
    def PhaseField_K(self,iel):

        k_local = ti.Matrix.zero(dt=self.dtype,n=8,m=8)
        F = ti.Vector([0.0]*8)
        for i in range(self.ngp):

            N = self.N_func[iel,i]
            BB = self.BB_mat[iel,i]
            vol = self.vol[iel,i]
                    
            a = N*((4*self.l0*(1-self.k)*self.History_psi[iel,i]/(self.Gc))+1)
            A1 = a.outer_product(N)
            A2 = BB.transpose()*(4*self.l0**2)@BB
            k_local += (A1+A2)*vol
            F += N*vol
        return k_local,F
    
    @ti.func
    def Update_fracture_params(self,c_local,iel):

        for i in range(self.ngp):
            N = self.N_func[iel,i]
            self.c_PF[iel,i] = ti.math.dot(c_local,N)
            self.g_PF[iel,i] = self.c_PF[iel,i]**2
        pass
    
    @ti.func               
    def D_matrix(self,iel,i):

        D = ti.Vector([[0.0,0.0,0.0,0.0,0.0,0.0],
                       [0.0,0.0,0.0,0.0,0.0,0.0],
                       [0.0,0.0,0.0,0.0,0.0,0.0],
                       [0.0,0.0,0.0,0.0,0.0,0.0],
                       [0.0,0.0,0.0,0.0,0.0,0.0],
                       [0.0,0.0,0.0,0.0,0.0,0.0]])

        if self.c_PF[iel,i]<0.999:
            g_deg = self.g_PF[iel,i]
            
            devprj=ti.Vector([[2/3,-1/3,-1/3,0.0,0.0,0.0],
                              [-1/3,2/3,-1/3,0.0,0.0,0.0],
                              [-1/3,-1/3,2/3,0.0,0.0,0.0],
                              [0.0,0.0,0.0,1/2,0.0,0.0],
                              [0.0,0.0,0.0,0.0,1/2,0.0],
                              [0.0,0.0,0.0,0.0,0.0,1/2]])
                
            soid = ti.Vector(([1],[1],[1],[0],[0],[0]))
            
            strain = self.strain_gp_curr[iel,i]
            trace_e = strain[0] + strain[1] + strain[2]

            trace_e_plus = (trace_e + ti.abs(trace_e))/2
            trace_e_minus = (trace_e - ti.abs(trace_e))/2
            
            if trace_e_plus!=0:
                trace_e_plus = 1
            if trace_e_minus!=0:
                trace_e_minus = 1

            dt_plus = trace_e_plus
            dt_minus = trace_e_minus

            D = 2*g_deg*self.G*devprj + self.K*(g_deg*dt_plus+dt_minus)*soid @ soid.transpose()  
           
        else:
            c1 = self.E/((1+self.n)*(1-2*self.n))

            D = c1*ti.Vector([[1-self.n,self.n,self.n,0,0,0],
                            [self.n,1-self.n,self.n,0,0,0],
                            [self.n,self.n,1-self.n,0,0,0],
                            [0,0,0,(1-2*self.n)/2,0,0],
                            [0,0,0,0,(1-2*self.n)/2,0],
                            [0,0,0,0,0,(1-2*self.n)/2]])    
        return D
   
    @ti.func
    def Stress_update(self,iel,i):

        strain = self.strain_gp_curr[iel,i]
        g_deg = self.g_PF[iel,i]
        #Volumetric Strain
        e_vol = strain[0] + strain[1] + strain[2]
        #Deviatoric Strain
        e_dev_1 = strain[0] - e_vol/3
        e_dev_2 = strain[1] - e_vol/3
        e_dev_3 = strain[2] - e_vol/3
        gamma_12 = strain[3]/2
        gamma_23 = strain[4]/2
        gamma_31 = strain[5]/2

        stress_11 = 2*g_deg*self.G*e_dev_1  
        stress_22 = 2*g_deg*self.G*e_dev_2  
        stress_33 = 2*g_deg*self.G*e_dev_3 
        stress_12 = 2*g_deg*self.G*gamma_12
        stress_23 = 2*g_deg*self.G*gamma_23
        stress_31 = 2*g_deg*self.G*gamma_31
        
        ST_dev = ti.Vector((stress_11,stress_22,stress_33,stress_12,stress_23,stress_31),dt=self.dtype)
        I = ti.math.eye(3)
        trace_e = strain[0] + strain[1] + strain[2]

        trace_e_plus = (trace_e + ti.abs(trace_e))/2
        trace_e_minus = (trace_e - ti.abs(trace_e))/2

        ST_vol = self.K*(g_deg*trace_e_plus + trace_e_minus)*I
    
        stress_vm = ti.sqrt((stress_11)**2+
                            (stress_22)**2+
                            (stress_33)**2+
                            2*(stress_12)**2+
                            2*(stress_23)**2+
                            2*(stress_31)**2) 
        
        ST = ti.Vector([ST_dev[0]+ST_vol[0,0],
                        ST_dev[1]+ST_vol[1,1],
                        ST_dev[2]+ST_vol[2,2],
                        ST_dev[3],
                        ST_dev[4],
                        ST_dev[5],
                        stress_vm],dt=self.dtype)
        
        self.stress_gp_curr[iel,i] = ST
        self.strain_gp_curr[iel,i] = strain
        pass
    
    @ti.func
    def Energy_update(self,iel):
        for i in range(self.ngp):
            strain = self.strain_gp_curr[iel,i]
            g_deg = self.g_PF[iel,i]
            
            e = ti.Vector([[strain[0], strain[3]/2,   strain[5]/2],
                           [strain[3]/2, strain[1],   strain[4]/2],
                           [strain[5]/2, strain[4]/2,   strain[2]]])
            
            I = ti.math.eye(3)
            trace_e_plus = (e.trace() + ti.abs(e.trace()))/2
            trace_e_minus = (e.trace() - ti.abs(e.trace()))/2

            e_dev = e - 1/3*e.trace()*I
            trace_strain_dev = (e_dev@e_dev).trace()

            psi_plus = 1/2*self.K*trace_e_plus**2 + self.G*trace_strain_dev
            
            psi_minus = 1/2*self.K*trace_e_minus**2


            psi_total = g_deg*psi_plus + psi_minus
            self.psi_plus[iel,i],self.psi_total[iel,i] = psi_plus,psi_total

            a = psi_plus/self.Fc -1
            self.thres_psi_plus[iel,i] = (a + ti.abs(a))/2
        pass
