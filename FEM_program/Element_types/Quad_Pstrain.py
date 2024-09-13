import taichi as ti 

@ti.data_oriented
class Element2D:
    def __init__(self,E,n,ngp,elements_con,node_coords,dofs,dtype):
        
        self.dtype = dtype
        self.E = E                 #Young modulus
        self.n = n                 #Poisson ratio
        self.ngp = ngp             #Integration points
        self.G = E/(2*(1+n))       #Shear modulus
        self.K = E/(3*(1-2*n))     #Bulk modulus

        self.lamda = (E*n)/((1+n)*(1-2*n)) 
        self.num_elements = len(elements_con)

        #Structure
        self.ti_node = ti.Vector.field(2,dtype=self.dtype,shape=((node_coords.shape[0]))) 
        self.ti_elements = ti.Vector.field(4,dtype=ti.int32,shape=((elements_con.shape[0]))) 
        self.ti_dofs =  ti.Vector.field(8,dtype=ti.int32,shape=((dofs.shape[0]))) 

        self.ti_node.from_numpy(node_coords)
        self.ti_elements.from_numpy(elements_con)
        self.ti_dofs.from_numpy(dofs)
        
        #Gauss points attributes
        self.coords_gp = ti.Vector.field(2,dtype=self.dtype,shape=(self.num_elements,4))
        
        #s_xx,s_yy,t_xy,s_zz,s_vm
        self.stress_gp_prev = ti.Vector.field(5,dtype=self.dtype,shape=(self.num_elements,ngp))
        self.stress_gp_curr = ti.Vector.field(5,dtype=self.dtype,shape=(self.num_elements,ngp))
        #e_xx,e_yy,g_xy,e_zz,e_vm
        self.strain_gp_prev = ti.Vector.field(5,dtype=self.dtype,shape=(self.num_elements,ngp)) 
        self.strain_gp_curr = ti.Vector.field(5,dtype=self.dtype,shape=(self.num_elements,ngp))

        #Functions for MatPoints
        self.B_mat = ti.Matrix.field(3,8,dtype=self.dtype,shape=(self.num_elements,ngp))
        self.BB_mat = ti.Matrix.field(2,4,dtype=self.dtype,shape=(self.num_elements,ngp))
        self.N_func = ti.Vector.field(4,dtype=self.dtype,shape=(self.num_elements,ngp))
        self.vol = ti.field(dtype=self.dtype,shape=(self.num_elements,ngp))
        #Initialize FEM functions
        self.Initialize()

        self.node_stress = ti.Vector.field(4,dtype=self.dtype,shape=(self.num_elements))

    @ti.kernel
    def Initialize(self):
        for iel in self.ti_elements:
            con = self.ti_elements[iel]
            coord_v = ti.Matrix([[0.0,0.0]]*4,dt=self.dtype)
            for i in range(4):
                coord_v[i,:2]= self.ti_node[con[i]]
            self.Initialize_func(coord_v,iel)
    
    @ti.func
    def Initialize_func(self,
                        coords,
                        iel):
        gp,w = self.Gauss_quad(self.ngp)
        for i in range(self.ngp):
            eta = gp[i,0]
            ksi = gp[i,1]
            N,dN = self.shape_fun(eta,ksi)
            B_mat,detJ,BB = self.get_B_mat(coords,dN)

            #Gauss point coordinates
            self.coords_gp[iel,i] = N @ coords
            self.B_mat[iel,i] = B_mat
            self.N_func[iel,i] = N 
            self.BB_mat[iel,i]= BB
            self.vol[iel,i] = w[i]*detJ

    @ti.func
    def shape_fun(self,eta,ksi):
       
        N1 = 1/4*(1-ksi)*(1-eta)
        N2 = 1/4*(1+ksi)*(1-eta)
        N3 = 1/4*(1+ksi)*(1+eta)
        N4 = 1/4*(1-ksi)*(1+eta)

        N = ti.Vector([N1, N2, N3, N4],dt=self.dtype)
        dN = 1/4 * ti.Matrix([[eta-1,1-eta,1+eta,-eta-1],
                              [ksi-1,-ksi-1,1+ksi,1-ksi]],dt=self.dtype)
        return N,dN
    
    @ti.func
    def Gauss_quad(self,ngp):
        a = 1.0/ti.sqrt(3.0)
        g_loc = ti.Matrix([[-a,  -a],
                           [-a,  a],
                           [a,  a],
                           [a,  -a]],dt=self.dtype)
        w = ti.Vector([1,1,1,1],dt=self.dtype)
        return g_loc,w
    @ti.func
    def get_B_mat(self,
                  coords,
                  dN):

        #Calculate Jacobian
        J = dN@coords
        detJ = ti.math.determinant(J)
        #InvJacobian
        # # BB = ti.linalg.solve(J,dN)
        inv_J = 1/detJ*ti.Matrix([[J[1,1],-J[0,1]],
                                 [-J[1,0],J[0,0]]],dt=self.dtype)
        BB = inv_J@dN
        B_mat = ti.Matrix([[BB[0,0], 0 ,BB[0,1],0,BB[0,2],0,BB[0,3],0],
                           [0 ,BB[1,0],0,BB[1,1],0,BB[1,2],0,BB[1,3]],
                           [BB[1,0], BB[0,0],BB[1,1],BB[0,1],BB[1,2],BB[0,2],BB[1,3],BB[0,3]]],dt=self.dtype)
        
        return B_mat,detJ,BB
    
    #Subclasses will overwrite this
    @ti.func
    def D_matrix(self):
        pass

    @ti.func
    def Stress_update(self):
        pass
    
    @ti.func
    def Local_stiffness(self,iel):
        k_local = ti.Matrix.zero(dt=self.dtype,n=8,m=8)
        for i in range(self.ngp):

            D_matrix = self.D_matrix(iel,i)
            B_mat = self.B_mat[iel,i]
            vol = self.vol[iel,i]
            k_local += B_mat.transpose() @ D_matrix @ B_mat *vol
        return k_local

    @ti.func
    def Get_fint(self,U_inc,iel):
        F_int = ti.Vector.zero(dt=self.dtype,n=8)
        for i in range(self.ngp):

            B_mat = self.B_mat[iel,i]
            vol = self.vol[iel,i]
            inc_strain = B_mat @ U_inc
            self.strain_gp_curr[iel,i] = ti.Vector([inc_strain[0],inc_strain[1],inc_strain[2], 0.0, 0.0],dt=self.dtype) + self.strain_gp_prev[iel,i]
            self.Stress_update(iel,i)           
            F_int += vol * B_mat.transpose() @ (self.stress_gp_curr[iel,i]-self.stress_gp_prev[iel,i])[:3]
        return F_int

    @ti.func
    def ExtrapolateGP(self,iel):
        a = 1/2*ti.sqrt(3)
        b = -1/2
        ext_mat = ti.Matrix([[1+a,b,1-a,b],
                            [b,1+a,b,1-a],
                            [1-a,b,1+a,b],
                            [b,1-a,b,1+a]],dt=self.dtype)
        
        gp_val = ti.Vector([self.stress_gp_curr[iel,i][4] for i in range(self.ngp)])
        self.node_stress[iel] = ext_mat@gp_val
        pass
    
@ti.data_oriented
class P_strain_elastic(Element2D):
    def __init__(self,E,n,ngp,Elements_connectivity,Node_coords,Elements_DoFs,dtype):
        super().__init__(E,n,ngp,Elements_connectivity,Node_coords,Elements_DoFs,dtype)

    @ti.func
    def D_matrix(self,iel,i):
        
        c1 = self.E/((1+self.n)*(1-2*self.n))

        D = c1* ti.Matrix([[1-self.n,self.n,0],
                            [self.n,1-self.n,0],
                            [0,0,(1-2*self.n)/2]],dt=self.dtype)        
        return D
    
    @ti.func
    def Stress_update(self,iel,i):
        #e_xx,e_yy,g_xy,e_zz,epl_vm
        strain = self.strain_gp_curr[iel,i]
        K = self.K
        G = self.G
        #Volumentric Strain
        e_vol = strain[0] + strain[1] + strain[3]
        P = K * e_vol
        #Deviatoric Strain
        e_dev_11 = strain[0] - e_vol/3
        e_dev_22 = strain[1] - e_vol/3
        gamma_12 = strain[2]/2
        e_dev_33 = strain[3] - e_vol/3
       
        stress_11 = 2*G*e_dev_11 + P 
        stress_22 = 2*G*e_dev_22 + P 
        stress_12 = 2*G*gamma_12
        stress_33 = 2*G*e_dev_33 + P

        stress_vm = ti.sqrt((2*G*e_dev_11)**2+
                            (2*G*e_dev_22)**2+
                            (2*G*e_dev_33 )**2+
                            2*(2*G*gamma_12)**2)
        
        ST = ti.Vector([stress_11,stress_22,stress_12,stress_33,stress_vm],dt=self.dtype)  #s_xx,s_yy,t_xy,s_zz,s_vm
        self.stress_gp_curr[iel,i] = ST
        pass

@ti.data_oriented
class P_strain_elastoplastic(Element2D):
    def __init__(self,E,n,ngp,H,sy,Elements_connectivity,Node_coords,Elements_DoFs,dtype):
        super().__init__(E,n,ngp,Elements_connectivity,Node_coords,Elements_DoFs,dtype)

        self.sy = sy
        self.H = H
        self.max_pl_iter = 10
        self.pl_tolerance = 1e-8

        self.Dg = ti.field(dtype=self.dtype,shape=(self.num_elements,ngp))      #Plastic multiplier derivative
        self.iYield = ti.field(dtype=ti.int32,shape=(self.num_elements,ngp))    #Check if gp is yielding

    @ti.func               
    def D_matrix(self,iel,i):
         
        stress = self.stress_gp_curr[iel,i]
        Dg = self.Dg[iel,i]

        D = ti.Vector([[0.0,0.0,0.0],
                       [0.0,0.0,0.0],
                       [0.0,0.0,0.0]])
        if self.iYield[iel,i]==1:
            #Elastoplastic
            devprj=ti.Vector([[0.666666667,-0.333333333, 0.00000000],
                            [-0.333333333,0.666666667,0.00000000],
                            [0.000000000,0.000000000,0.50000000]])
            
            soid = ti.Vector(([1],[1],[0]))
            P = (stress[0]+stress[1]+stress[3])*1/3
            s_dev_11 = stress[0]-P
            s_dev_22 = stress[1]-P
            s_dev_12 = stress[2]
            s_dev_33 = stress[3]-P

            ST = ti.Vector(([s_dev_11],[s_dev_22],[s_dev_12]))
            #Von mises stress
            s_vm = ti.sqrt(s_dev_11**2+
                           s_dev_22**2+
                           2*(s_dev_12)**2+
                           s_dev_33**2)
            
            q = ti.sqrt(3/2)*s_vm
            q_trial = q+3*self.G*Dg

            a_fact = 2*self.G*(1-3*self.G*Dg/q_trial)

            b_fact = 6*(self.G**2)*((Dg/q_trial)-(1/(3*self.G+self.H)))/(s_vm**2)
            
            D = a_fact*devprj+b_fact*ST@ST.transpose() + self.K* soid @ soid.transpose()
           
        else:
        #Elastic
            c1 = self.E/((1+self.n)*(1-2*self.n))
            D = c1* ti.Vector([[1-self.n,self.n,0],
                            [self.n,1-self.n,0],
                            [0,0,(1-2*self.n)/2]])
            
        return D
    
    @ti.func
    def Stress_update(self,iel,i):

        strain = self.strain_gp_curr[iel,i]
        self.Dg[iel,i] = 0.0
        K = self.K
        G = self.G
        #Volumentric Strain
        e_vol = strain[0] + strain[1] + strain[3]
        P = K * e_vol
        #Deviatoric Strain
        e_dev_11 = strain[0] - e_vol/3
        e_dev_22 = strain[1] - e_vol/3
        gamma_12 = strain[2]/2
        e_dev_33 = strain[3] - e_vol/3
       
        #Plastic strain
        e_pl = strain[4]
        #Deviatoric stress 
        s_dev = ti.Vector([2*G*e_dev_11,2*G*e_dev_22,2*G*e_dev_33,2*G*gamma_12])
        q_trial = ti.sqrt(3/2*(s_dev[0]**2+s_dev[1]**2+s_dev[2]**2+2*s_dev[3]**2))
        sy = self.sy + self.H * e_pl
        phi = q_trial-sy
        ST = ti.Vector([0.0,0.0,0.0,0.0,0.0]) 
        
        if phi<0:
            self.iYield[iel,i] =0
            #Elastic step
            stress_11 = 2*G*e_dev_11 + P 
            stress_22 = 2*G*e_dev_22 + P 
            stress_12 = 2*G*gamma_12
            stress_out = 2*G*e_dev_33 + P 
            stress_vm = ti.sqrt(s_dev[0]**2+
                                s_dev[1]**2+
                                s_dev[2]**2+
                                2*s_dev[3]**2)
            
            ST = ti.Vector((stress_11,stress_22,stress_12,stress_out,stress_vm))
        else:
            self.iYield[iel,i] =1
            Ddg = phi/(3*G+self.H)
            e_pl += Ddg  
            #Update stress
            factor_S = 2*G*(1-3*G*Ddg/q_trial)
            factor_E = factor_S/(2*G)
            s_vm = ti.sqrt((factor_S*e_dev_11)**2+
                           (factor_S*e_dev_22)**2+
                           2*(factor_S*gamma_12)**2+
                           (factor_S*e_dev_33)**2)

            ST = ti.Vector((factor_S*e_dev_11+P,
                            factor_S*e_dev_22+P,
                            factor_S*gamma_12,
                            factor_S*e_dev_33+P,
                            s_vm))
            
            strain = ti.Vector((factor_E*e_dev_11+e_vol/3,
                                factor_E*e_dev_22+e_vol/3,
                                factor_E*gamma_12*2,
                                factor_E*e_dev_33+e_vol/3,
                                e_pl))
            
            self.Dg[iel,i] = Ddg
        
        self.stress_gp_curr[iel,i] = ST
        self.strain_gp_curr[iel,i] = strain
        pass

@ti.data_oriented
class P_strain_BrittleFracture(Element2D):
    def __init__(self,E,n,ngp,l0,Gc1,Gc2,k,Elements_connectivity,Node_coords,Elements_DoFs,dtype):
        super().__init__(E,n,ngp,Elements_connectivity,Node_coords,Elements_DoFs,dtype)
        
        self.l0 = l0
        self.Gc1 = Gc1
        self.Gc2 = Gc2
        self.k = k

        self.psi_plus_mode_1 = ti.field(dtype=self.dtype,shape=(self.num_elements,ngp))  # Positive part of energy for mode 1 
        self.psi_plus_mode_2 = ti.field(dtype=self.dtype,shape=(self.num_elements,ngp))  # Positive part of energy for mode 2

        self.psi_total = ti.field(dtype=self.dtype,shape=(self.num_elements,ngp))  # Total energy
        
        self.History_psi_mode1 = ti.field(dtype=self.dtype,shape=(self.num_elements,ngp))  # History variable for mode 1 
        self.History_psi_mode2 = ti.field(dtype=self.dtype,shape=(self.num_elements,ngp))  # History variable for mode 2 
      
        self.c_PF = ti.field(dtype=self.dtype,shape=(self.num_elements,ngp))  #Phase field variable
        self.g_PF = ti.field(dtype=self.dtype,shape=(self.num_elements,ngp))  #Degradation variable
        self.g_PF.fill(1.0)
    
    @ti.func
    def PhaseField_K(self,iel):

        k_local = ti.Matrix.zero(dt=self.dtype,n=4,m=4)
        Force_v = ti.Vector([0.0]*4)
        for i in range(self.ngp):

            N = self.N_func[iel,i]
            BB = self.BB_mat[iel,i]
            vol = self.vol[iel,i]
            
            F = self.History_psi_mode1[iel,i]/self.Gc1 + self.History_psi_mode2[iel,i]/self.Gc2

            a = N*((4*self.l0*(1-self.k)*F)+1)
            
            A1 = a.outer_product(N)
            A2 = BB.transpose()*(4*self.l0**2)@BB
            k_local += (A1+A2)*vol
            Force_v += N*vol
        return k_local,Force_v
    
    @ti.func
    def Update_fracture_params(self,c_local,iel):

        for i in range(self.ngp):
            N = self.N_func[iel,i]
            self.c_PF[iel,i] = ti.math.dot(c_local,N)
            self.g_PF[iel,i] = self.c_PF[iel,i]**2
        pass
    
    @ti.func               
    def D_matrix(self,iel,i):

        D = ti.Vector([[0.0,0.0,0.0],
                       [0.0,0.0,0.0],
                       [0.0,0.0,0.0]])

        if self.c_PF[iel,i]<0.9999:
            g_deg = self.g_PF[iel,i]
            
            devprj=ti.Vector([[0.666666667,-0.333333333, 0.00000000],
                              [-0.333333333,0.666666667, 0.00000000],
                              [0.000000000,0.000000000,  0.50000000]])
                
            soid = ti.Vector(([1],[1],[0]))
            
            strain = self.strain_gp_curr[iel,i]
            trace_e = strain[0] + strain[1] + strain[3]

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

            D = c1* ti.Vector([[1-self.n,self.n,0],
                                [self.n,1-self.n,0],
                                [0,0,(1-2*self.n)/2]])  
        return D
   
    @ti.func
    def Stress_update(self,iel,i):

        strain = self.strain_gp_curr[iel,i]
        g_deg = self.g_PF[iel,i]
        G = self.G

        #Volumentric Strain
        e_vol = strain[0] + strain[1] + strain[3]
        #Deviatoric Strain
        e_dev_1 = strain[0] - e_vol/3
        e_dev_2 = strain[1] - e_vol/3
        e_dev_3 = strain[3] - e_vol/3
        gamma_12 = strain[2]/2

        stress_11 = 2*g_deg*G*e_dev_1  
        stress_22 = 2*g_deg*G*e_dev_2  
        stress_12 = 2*g_deg*G*gamma_12
        stress_out = 2*g_deg*G*e_dev_3  
        ST_dev = ti.Vector((stress_11,stress_22,stress_12,stress_out))

        stress_vm = ti.sqrt(ST_dev[0]**2+
                            ST_dev[1]**2+
                            ST_dev[3]**2+
                            2*ST_dev[2]**2)

        I = ti.math.eye(3)
        trace_e = strain[0] + strain[1] + strain[3]

        trace_e_plus = (trace_e + ti.abs(trace_e))/2
        trace_e_minus = (trace_e - ti.abs(trace_e))/2

        ST_vol = self.K*(g_deg*trace_e_plus + trace_e_minus)*I
        
        ST = ti.Vector([ST_dev[0]+ST_vol[0,0],ST_dev[1]+ST_vol[1,1],ST_dev[2],ST_dev[3]+ST_vol[2,2],stress_vm])
        
        self.stress_gp_curr[iel,i] = ST
        self.strain_gp_curr[iel,i] = strain
        pass
    
    @ti.func
    def Energy_update(self,iel):

        for i in range(self.ngp):
            strain = self.strain_gp_curr[iel,i]
            g_deg = self.g_PF[iel,i]
            
            e = ti.Vector([[strain[0],   strain[2]/2,    0.0    ],
                           [strain[2]/2, strain[1],      0.0    ],
                           [    0.0,          0.0,     strain[3]]])
            
            I = ti.math.eye(3)
        
            trace_e_plus = (e.trace() + ti.abs(e.trace()))/2
            trace_e_minus = (e.trace() - ti.abs(e.trace()))/2

            e_dev = e - 1/3*e.trace()*I
            trace_strain_dev = (e_dev@e_dev).trace()

            psi_plus_mode1 = 1/2*self.K*trace_e_plus**2  
            psi_plus_mode2 = self.G*trace_strain_dev
            
            psi_plus = psi_plus_mode1 + psi_plus_mode2

            psi_minus = 1/2*self.K*trace_e_minus**2
            psi_total = g_deg*psi_plus + psi_minus
            
            
            self.psi_plus_mode_1[iel,i] = psi_plus_mode1
            self.psi_plus_mode_2[iel,i] = psi_plus_mode2
            self.psi_total[iel,i] = psi_total
        pass

@ti.data_oriented
class P_strain_DuctileFracture(Element2D):
    def __init__(self,E,n,sy,H,l0,Gc1,Gc2,k,ngp,Elements_connectivity,Node_coords,Elements_DoFs,dtype):
        super().__init__(E,n,ngp,Elements_connectivity,Node_coords,Elements_DoFs,dtype)
        
        self.l0 = l0
        self.Gc = Gc1
        self.k = k

        self.epcrit = 0.0
        self.a = 0.08
        self.b = 1e-5

        self.sy = sy
        self.H = H
        self.max_pl_iter = 10
        self.pl_tolerance = 1e-8

        self.Dg = ti.field(dtype=self.dtype,shape=(self.num_elements,ngp))        # Plastic multiplier der
        self.psi_plus_mode_1 = ti.field(dtype=self.dtype,shape=(self.num_elements,ngp))  # Positive part of energy
        self.psi_plus_mode_2 = ti.field(dtype=self.dtype,shape=(self.num_elements,ngp))  # Positive part of energy

        self.psi_total = ti.field(dtype=self.dtype,shape=(self.num_elements,ngp)) # Total energy
        
        self.History_psi_mode1 = ti.field(dtype=self.dtype,shape=(self.num_elements,ngp))  # History variable
        self.History_psi_mode2 = ti.field(dtype=self.dtype,shape=(self.num_elements,ngp))  # History variable
      
        self.c_PF = ti.field(dtype=self.dtype,shape=(self.num_elements,ngp))      #Phase field variable
        self.g_PF = ti.field(dtype=self.dtype,shape=(self.num_elements,ngp))      #Degradation variable
        self.g_PF.fill(1.0)
        
        self.iYield = ti.field(dtype=ti.int32,shape=(self.num_elements,ngp))      #Check if gp is yielding

    @ti.func
    def PhaseField_K(self,iel):

        k_local = ti.Matrix.zero(dt=self.dtype,n=4,m=4)
        F = ti.Vector([0.0]*4)
        for i in range(self.ngp):

            N = self.N_func[iel,i]
            BB = self.BB_mat[iel,i]
            vol = self.vol[iel,i]
            e_acc_pl = self.strain_gp_curr[iel,i][4]
            
            f=1.0
            if e_acc_pl<=self.epcrit:
                f=1.0
            elif e_acc_pl>self.epcrit+self.a:
                f=self.b
            elif e_acc_pl>self.epcrit:
                f=((1-self.b)/(self.a**2))*(e_acc_pl-self.epcrit-self.a)**2+self.b
        
        
            a = N*(2*self.History_psi_mode1[iel,i]/(f*self.Gc) +1/self.l0)
            A1 = a.outer_product(N)
            A2 = BB.transpose()*(self.l0)@BB
            k_local += (A1+A2)*vol
            A3=(2*self.History_psi_mode1[iel,i]/(f*self.Gc))*N

            F += A3*vol
        return k_local,F
    
    @ti.func
    def Update_fracture_params(self,c_local,iel):

        for i in range(self.ngp):
            N = self.N_func[iel,i]
            self.c_PF[iel,i] = ti.max(ti.math.dot(c_local,N),1)
            self.g_PF[iel,i] = (1-self.c_PF[iel,i])**2
        pass
    
    @ti.func               
    def D_matrix(self,iel,i):
        
        stress = self.stress_gp_curr[iel,i]
        Dg = self.Dg[iel,i]
        g_deg = self.g_PF[iel,i]

        D = ti.Vector([[0.0,0.0,0.0],
                       [0.0,0.0,0.0],
                       [0.0,0.0,0.0]])
        
        devprj=ti.Vector([[0.666666667,-0.333333333, 0.00000000],
                          [-0.333333333,0.666666667, 0.00000000],
                          [0.000000000,0.000000000,  0.50000000]])
            
        soid = ti.Vector(([1],[1],[0]))
        
        strain = self.strain_gp_curr[iel,i]
        trace_e = strain[0] + strain[1] + strain[3]

        trace_e_plus = (trace_e + ti.abs(trace_e))/2
        trace_e_minus = (trace_e - ti.abs(trace_e))/2
        
        if trace_e_plus!=0:
            trace_e_plus = 1
        if trace_e_minus!=0:
            trace_e_minus = 1

        dt_plus = trace_e_plus
        dt_minus = trace_e_minus

        if self.iYield[iel,i] ==1:
            
            #Volumetric stress
            P = (stress[0]+stress[1]+stress[3])*1/3
            #Deviatoric stress
            s_dev_1 = stress[0]-P   #σxx
            s_dev_2 = stress[1]-P   #σyy
            s_dev_3 = stress[2]     #τxy
            s_dev_4 = stress[3]-P   #σzz out of plane 

            ST_dev = ti.Vector(([s_dev_1],[s_dev_2],[s_dev_3]))
            #Last Von mises stress
            s_vm = ti.sqrt(s_dev_1**2+
                           s_dev_2**2+
                           2*(s_dev_3)**2+
                           s_dev_4**2)

            q = ti.sqrt(3/2)*s_vm
            q_trial = q+3*g_deg*self.G*Dg

            a_fact = 2*g_deg*self.G*(1-3*g_deg*self.G*Dg/q_trial)
           
            b_fact = 6*((g_deg**2)*(self.G**2))*((Dg/q_trial)-(1/(3*g_deg*self.G+self.H)))/(s_vm**2)
            if ti.abs(g_deg)<1e-6:
                b_fact=1e-6
            D = a_fact*devprj+b_fact*ST_dev@ST_dev.transpose() + self.K*(g_deg*dt_plus+dt_minus)*soid @ soid.transpose()         
        
        else:
            D = 2*g_deg*self.G*devprj + self.K*(g_deg*dt_plus+dt_minus)*soid @ soid.transpose()

        for i in ti.static(range(3)):
            if D[i,i]<1e-6:
                D[i,i]=1e-6
        return D
    
    @ti.func
    def Stress_update(self,iel,i):

        strain = self.strain_gp_curr[iel,i]
        self.Dg[iel,i] = 0.0
        g_deg = self.g_PF[iel,i]

        G = self.G
        #Volumentric Strain
        e_vol = strain[0] + strain[1] + strain[3]
        #Deviatoric Strain
        e_dev_1 = strain[0] - e_vol/3
        e_dev_2 = strain[1] - e_vol/3
        e_dev_3 = strain[3] -e_vol/3
        gamma_12 = strain[2]/2
        #Plastic strain
        e_pl = strain[4]
        #Deviatoric stress 
        s_dev = ti.Vector([2*g_deg*G*e_dev_1,
                           2*g_deg*G*e_dev_2,
                           2*g_deg*G*e_dev_3,
                           2*g_deg*G*gamma_12])
        
        q_trial = ti.sqrt(3/2*(s_dev[0]**2+
                               s_dev[1]**2+
                               s_dev[2]**2+
                               2*s_dev[3]**2))

        sy = self.sy + self.H * e_pl
        phi = q_trial-sy
        ST = ti.Vector([0.0,0.0,0.0,0.0,0.0]) 
        ST_dev = ti.Vector([0.0,0.0,0.0,0.0]) 
        
        if phi<0:
            self.iYield[iel,i] = 0 
            #Elastic step
            stress_11 = 2*g_deg*G*e_dev_1  
            stress_22 = 2*g_deg*G*e_dev_2  
            stress_12 = 2*g_deg*G*gamma_12
            stress_out = 2*g_deg*G*e_dev_3  
            ST_dev = ti.Vector((stress_11,stress_22,stress_12,stress_out))
            
        else:
            self.iYield[iel,i] = 1 
            Ddg = phi/(3*G*g_deg+self.H)
            e_pl += Ddg  
            #Update stress
            factor_S = 2*G*g_deg*(1-3*G*g_deg*Ddg/q_trial)
            factor_E = factor_S/(2*G*g_deg)

            ST_dev = ti.Vector((factor_S*e_dev_1,
                                factor_S*e_dev_2,
                                factor_S*gamma_12,
                                factor_S*e_dev_3))
            
            strain = ti.Vector((factor_E*e_dev_1+e_vol/3,
                                factor_E*e_dev_2+e_vol/3,
                                factor_E*gamma_12*2,
                                factor_E*e_dev_3+e_vol/3,
                                e_pl))
            
            self.Dg[iel,i] = Ddg
        
        stress_vm = ti.sqrt(ST_dev[0]**2+
                            ST_dev[1]**2+
                            ST_dev[2]**2+
                            2*ST_dev[3]**2)

        I = ti.math.eye(3)
        trace_e = strain[0] + strain[1] + strain[3]

        trace_e_plus = (trace_e + ti.abs(trace_e))/2
        trace_e_minus = (trace_e - ti.abs(trace_e))/2

        ST_vol = self.K*(g_deg*trace_e_plus + trace_e_minus)*I
        
        ST = ti.Vector([ST_dev[0]+ST_vol[0,0],
                        ST_dev[1]+ST_vol[1,1],
                        ST_dev[2],
                        ST_dev[3]+ST_vol[2,2],
                        stress_vm])
        
        self.stress_gp_curr[iel,i] = ST
        self.strain_gp_curr[iel,i] = strain
        pass
    
    @ti.func
    def Energy_update(self,iel):

        for i in range(self.ngp):
            strain = self.strain_gp_curr[iel,i]
            g_deg = self.g_PF[iel,i]
            
            e = ti.Vector([[strain[0],   strain[2]/2,    0.0    ],
                           [strain[2]/2, strain[1],      0.0    ],
                           [    0.0,          0.0,     strain[3]]])
            
            I = ti.math.eye(3)

            trace_e_plus = (e.trace() + ti.abs(e.trace()))/2
            trace_e_minus = (e.trace() - ti.abs(e.trace()))/2

            e_dev = e - 1/3*e.trace()*I
            trace_strain_dev = (e_dev@e_dev).trace()

            psi_plus = 1/2*self.K*trace_e_plus**2 + self.G*trace_strain_dev
            psi_minus = 1/2*self.K*trace_e_minus**2
            psi_total = g_deg*psi_plus + psi_minus
            self.psi_plus_mode_1[iel,i],self.psi_total[iel,i] = psi_plus,psi_total
        pass