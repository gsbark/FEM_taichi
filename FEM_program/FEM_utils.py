from Element_types.Hex3D import *
from Element_types.Quad_Pstrain import *
from Geometry import *
from Solvers import LoadControl,DisplacementControl
import taichi as ti 
import numpy as np 
import meshio
import vtk
import shutil
import os 
from scipy.sparse import csr_matrix,coo_array

class Preprocess_FEM:
    def __init__(self,
                 Analysis_type:int,
                 Element:dict,
                 Elastic_prop:dict,
                 Plasticity:dict,
                 Fracture:dict,
                 Mesh_type:dict,
                 Solver:dict,
                 Kinematic_constraints:bool,
                 dtype:str):
        
        self.Analysis_type = Analysis_type
        self.Element = Element
        self.Elastic_prop = Elastic_prop
        self.Plasticity = Plasticity
        self.Fracture = Fracture
        self.Mesh_type = Mesh_type
        self.Solver = Solver
        self.Kinematic_constraints = Kinematic_constraints
        
        if dtype =='float32':
            self.dtype = ti.f32
            self.dtype_np = np.float32
        elif dtype =='float64':
            self.dtype = ti.f64
            self.dtype_np = np.float64

        if Element['type']=='8-Hex':
            self.dim = 3
            self.el_nodes = 8
        elif Element['type']=='4-Quad':
            self.dim = 2
            self.el_nodes = 4  

        self.Initialize_Mesh()
        self.Initialize_element()
        self.Initialize_BCs()

        shutil.rmtree('./Analysis_res',ignore_errors=True)
        os.mkdir('./Analysis_res')
        os.mkdir('./Analysis_res/vtk')
        
    def Initialize_Mesh(self):
        '''
        Create mesh for the domain 
        '''
        if self.Mesh_type['type']==1:
            self.Domain = Load_mesh(Domain=self.Mesh_type['Domain'],
                                    dim=self.Mesh_type['dim'],
                                    dtype=self.dtype_np)
            
        elif self.Mesh_type['type']==2:
            self.Domain = Create_mesh2D(nel_x=self.Mesh_type['nel_x'],
                                        nel_y= self.Mesh_type['nel_y'],
                                        l_x=1,l_y=1,
                                        dtype=self.dtype_np)
            
        elif self.Mesh_type['type']==3:
            self.Domain = Create_mesh3D(nel_x=self.Mesh_type['nel_x'],
                                        nel_y= self.Mesh_type['nel_y'],
                                        nel_z=self.Mesh_type['nel_z'],
                                        l_x=1,l_y=1,l_z=1,
                                        dtype=self.dtype_np)
        elif self.Mesh_type['type']==4:
            self.Domain = Make_mesh(Domain=self.Mesh_type['Domain'],
                                    dim=self.Mesh_type['dim'],
                                    dtype=self.dtype_np)

        self.num_elements = self.Domain['Elements_connectivity'].shape[0]
        self.num_nodes = self.Domain['Node_coords'].shape[0]
        assert self.dim==self.Domain['Node_coords'].shape[1], 'error in dim'

    def Initialize_element(self):
        '''
        Initialize element type 
        '''
        if self.Analysis_type ==1:
            if self.Element['type']=='8-Hex':
                self.El_type = Elastic_3D(**self.Elastic_prop,
                                          **self.Domain,
                                          ngp=self.Element['ngp'],    
                                          dtype=self.dtype) 
            elif self.Element['type']=='4-Quad':
                self.El_type = P_strain_elastic(**self.Elastic_prop,
                                                **self.Domain,
                                                ngp=self.Element['ngp'],
                                                dtype=self.dtype) 
        elif self.Analysis_type ==2:
            if self.Element['type']=='8-Hex':
                raise NotImplementedError('Change analysis type') 
            
            elif self.Element['type']=='4-Quad':
                self.El_type = P_strain_elastoplastic(**self.Elastic_prop,
                                                      **self.Plasticity,
                                                      **self.Domain,
                                                      ngp=self.Element['ngp'],
                                                      dtype=self.dtype) 
        elif self.Analysis_type ==3:
            if self.Element['type']=='8-Hex':
               self.El_type = BrittleFracture_3D(**self.Elastic_prop,
                                                 **self.Fracture,
                                                 **self.Domain,
                                                 ngp=self.Element['ngp'],
                                                 dtype=self.dtype) 
               
            elif self.Element['type']=='4-Quad':
                self.El_type = P_strain_BrittleFracture(**self.Elastic_prop,
                                                        **self.Fracture,
                                                        **self.Domain,
                                                        ngp=self.Element['ngp'],
                                                        dtype=self.dtype)
        elif self.Analysis_type ==4:
            if self.Element['type']=='8-Hex':
                raise NotImplementedError('Ductile Fracture for Hex element not implemented yet') 
            
            elif self.Element['type']=='4-Quad':
                self.El_type = P_strain_DuctileFracture(**self.Elastic_prop,
                                                        **self.Plasticity,
                                                        **self.Fracture,
                                                        **self.Domain,
                                                        ngp=self.Element['ngp'],
                                                        dtype=self.dtype)
        else:
            raise NotImplementedError('Choose anaysis type : 1: Elastic \n \
                                          2: Elastoplastic J2 \n \
                                          3: PF Brittle fracture \n \
                                          4: PF Ductile fracture') 

    def Initialize_BCs(self):
      
        nodes = self.Domain['Node_coords']
        Dofs = self.Domain['Elements_DoFs']
        #2D case 
        if self.dim==2:
            all_dofs = np.arange(np.max(Dofs.reshape(-1,1))+1)

            top_nodes = np.where((nodes[:, 1] == np.max(nodes[:, 1])))[0]
            bot_nodes = np.where((nodes[:, 1] == np.min(nodes[:, 1])))[0]
            right_nodes = np.where((nodes[:, 0] == np.max(nodes[:, 0])))[0]
            left_nodes = np.where((nodes[:, 0] == np.min(nodes[:, 0])))[0]
            
            fixed_dofs_x = 2 * bot_nodes
            fixed_dofs_y = 2 * bot_nodes + 1

            fixed_dofs = np.concatenate((fixed_dofs_x,fixed_dofs_y),axis=0)
            free_dofs =  np.delete(all_dofs,fixed_dofs)

            self.Dofs_boolean_mask = np.isin(Dofs,free_dofs).astype(np.int32)
            #All_dofs_map to rearrange the free DoFs
            self.all_dofs_map = all_dofs.astype(np.int32)       
            self.all_dofs_map[np.isin(all_dofs,fixed_dofs)] = np.array(-1,np.int32)
            self.all_dofs_map[np.isin(all_dofs,free_dofs)] = np.arange(free_dofs.shape[0]).astype(np.int32)

            self.n_free_DoFs = free_dofs.shape[0]
            self.free_dofs = free_dofs
            if self.Solver['control']=='displacement':
                load_dof = np.array([2*top_nodes[0]+1])
                self.controlled_dof = self.all_dofs_map[load_dof]
                self.controlled_dof_1 = load_dof
                self.Solver['controlled_dof'] = self.controlled_dof
                assert (self.controlled_dof != -1).all()
                if self.Kinematic_constraints==True:
                    ind = np.array(2*top_nodes[1:]+1)
                    kinem_constraints = np.vstack((ind, np.full(len(ind), load_dof)))
                    #rearrange 
                    self.kinem_constraints = self.all_dofs_map[kinem_constraints]
                    assert (self.kinem_constraints != -1).all()
            else:
                #Load control without kinematic constraints 
                load_dof = 2*top_nodes+1
        #3D case 
        else:
            #TODO: Add BCs for 3D case 
            raise NotImplementedError('Add BCs') 
        
        
#---------------------------------------------
#Calculate Constraint matrix 
def Get_ConstraintMatrix(const_dof:np.ndarray,dofs:int):
    #Initialize Constraint matrix
    C_mat = np.zeros((const_dof.shape[1],dofs),dtype=np.float32)
    for i in range(const_dof.shape[1]):
        if const_dof[0,i] != -1:
            C_mat[i,const_dof[0,i]] = -1       
        C_mat[i,const_dof[1,i]] = 1 
    return C_mat

def generate_sparse_ij(DoFs:np.ndarray,mask:np.ndarray,map:np.ndarray):
    ind = np.zeros((DoFs.shape[0]*DoFs.shape[1]**2,2),dtype=np.int32)
    gap = 0
    for i in range(DoFs.shape[0]):
        d = DoFs[i]
        idx = map[d[mask.astype(bool)[i]]]
        ind1 = np.array(np.meshgrid(idx,idx)).T.reshape(-1,2)
        ind[gap:gap+ind1.shape[0]] = ind1
        gap += ind1.shape[0] 
    return np.unique(ind[:gap],axis=0)

def generate_row_pointers(indices,values=None):
    # Sort the array based on row indices
    sorted_indices = indices[indices[:, 0].argsort()]
    
    row_pointers = np.zeros(max(indices[:, 0]) + 2, dtype=np.int32)
    for idx in sorted_indices:
        row_pointers[idx[0] + 1] += 1
    # Accumulate counts to obtain row pointers
    row_pointers = np.cumsum(row_pointers)
    n_col = sorted_indices[:,1].max()+1
    id = np.array(n_col*sorted_indices[:,0]+sorted_indices[:,1])
    sorted_indices = sorted_indices[:,1][id.argsort()]
    if values is not None:
        return row_pointers,sorted_indices,values[indices[:, 0].argsort()]
    else:
        return row_pointers.astype(np.int32),sorted_indices.astype(np.int32)
    
def get_CSR(sparse_idx):
    A = coo_array((np.arange(sparse_idx.shape[0]),
                   (sparse_idx[:,0], sparse_idx[:,1]))).tocsr()
    
    pointers,col_indx = A.indptr,A.indices
    return pointers,col_indx

def get_Sparse_Kss(kinematic_cons,dofs):
    C = Get_ConstraintMatrix(kinematic_cons,dofs)
    cc = csr_matrix(C)
    KSS= (1e8*cc.transpose()@cc).tocoo()
    KSS_sparse_ij = np.stack((KSS.row,KSS.col),axis=1)      
    return KSS_sparse_ij,KSS.data
#---------------------------------------------



#Export functions
#------------------------------------------------
def WriteVTK(El_con:np.ndarray,
             node_coords:np.ndarray,
             node_data:np.ndarray,
             file:str):

    Connectivity = El_con
    if node_coords.shape[1] <3:
        points = np.concatenate((node_coords,np.zeros((node_coords.shape[0],1))),axis=1)
        cells = {'quad':Connectivity}
    else:
        points = node_coords
        cells = {'hexahedron':Connectivity}
    point_data = node_data
    mesh = meshio.Mesh(points=points,
                       cells=cells,
                       point_data=point_data)
    meshio.write(file,mesh=mesh)

def GaussPoints_export(Particles:np.ndarray,
                       Val:np.ndarray,
                       inc:int):
    
    Val = Val.reshape(-1,1)
    points = vtk.vtkPoints()
    values_array1 = vtk.vtkDoubleArray()
    values_array1.SetNumberOfComponents(1)
    values_array1.SetName("Values1")

    for Id, particle in enumerate(Particles.reshape(-1,3)):
        points.InsertNextPoint(particle[0], particle[1], particle[2])
        values_array1.InsertNextValue(Val[Id])

    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)
    polydata.GetPointData().AddArray(values_array1)

    writer = vtk.vtkPolyDataWriter()
    fname = f'./Analysis_res/vtk/particle_data{inc}.vtk'
    writer.SetFileName(fname)
    writer.SetInputData(polydata)
    writer.Write()
#------------------------------------------------