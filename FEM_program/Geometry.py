import gmsh
import sys
import numpy as np 
import matplotlib.pyplot as plt
import numpy.matlib as matlib
import meshio
import time 

'''
Create Geometry of the domain
Supports : 
1) 2D structured mesh
2) 3D structured mesh
3) load mesh from .msh file 
'''

#2D quad sturctured mesh 
def Create_mesh2D(nel_x:int,nel_y:int,
                  l_x:int,l_y:int,
                  dtype:np.dtype):

    #Create Nodes TODO: change this 
    node_coords = np.zeros(((nel_x+1)*(nel_y+1),2),dtype=dtype)
    for i,y in enumerate(np.linspace(0,l_y,nel_y+1)):
        for j,x in enumerate(np.linspace(0,l_x,nel_x+1)):
            node_coords[(nel_x+1)*i+j] = x,y
    #Create Elements/Dofs
    element_con = np.zeros((nel_x*nel_y,4),dtype=np.int32)
    dofs = np.zeros((nel_x*nel_y,2*4),dtype=np.int32)
    iEl = 0
    for j in range(nel_y):
        for i in range(nel_x):
            n0 = i + j*(nel_x+1)
            element_con[iEl] = [n0,n0+1,n0+1+(nel_x+1),n0+(nel_x+1)]      
            dofs[iEl] = [2*n0,2*n0+1,2*(n0+1),2*(n0+1)+1,2*(n0+1+(nel_x+1)),2*(n0+1+(nel_x+1))+1,2*(n0+(nel_x+1)),2*(n0+(nel_x+1))+1]
            iEl +=1
    
    Domain = {'Node_coords':node_coords,
              'Elements_connectivity':element_con,
              'Elements_DoFs':dofs}
    return Domain

#3D hex structured mesh 
def Create_mesh3D(nel_x:int,nel_y:int,nel_z:int,
                  l_x:int,l_y:int,l_z:int,
                  dtype:np.dtype):

    mesh_vertices = []
    vertex_dict = {}
    def get_or_add_vertex_index(vertex):
        if vertex not in vertex_dict:
            vertex_dict[vertex] = len(mesh_vertices)
            mesh_vertices.append(vertex)
        return vertex_dict[vertex]
    
    #Create Elements/Dofs
    element_con = np.zeros((nel_x*nel_y*nel_z,8),dtype=np.int32)
    dofs = np.zeros((nel_x*nel_y*nel_z,3*8),dtype=np.int32)
    iEl = 0
    for y in range(nel_y):
        for x in range(nel_x):
            for z in range(nel_z):
                cube_vertices = [
                    (x, y, z),
                    (x + 1, y, z),
                    (x + 1, y + 1, z),
                    (x, y + 1, z),
                    (x, y, z+1),
                    (x + 1, y, z+1),
                    (x + 1, y + 1, z+1),
                    (x, y + 1, z+1)]
                
                con  = [get_or_add_vertex_index(v) for v in cube_vertices]
                element_con[iEl]=np.array(con)
                el_dofs = []
                for i in range(8):
                    el_dofs.append(con[i]*3)
                    el_dofs.append(con[i]*3+1)
                    el_dofs.append(con[i]*3+2)
                dofs[iEl] = el_dofs
                iEl +=1
    node_coords = np.array(mesh_vertices,dtype=dtype)*np.array([l_x/max(mesh_vertices)[0],
                                                                l_y/max(mesh_vertices)[1],
                                                                l_z/max(mesh_vertices)[2]],dtype=dtype)
    
    Domain = {'Node_coords':node_coords,
              'Elements_connectivity':element_con,
              'Elements_DoFs':dofs}
    return Domain

#Load mesh from gmsh
def Load_mesh(Domain:str,
              dim:int,
              dtype:np.dtype):
    '''
    gmsh element type
    1: Line elements
    2: Triangle elements
    3: Quadrilateral elements
    4: Tetrahedral elements
    5: Hexahedral elements
    '''
    file = f'./Domains/{Domain}'   
    np.set_printoptions(threshold=sys.maxsize)
    gmsh.initialize()
    gmsh.open(file)

    if dim ==2:
        el_type = 3 #Quad
        NodePerEl = 4
    elif dim ==3:
        el_type = 5 #Hex
        NodePerEl = 8

    NumberOfElements = gmsh.model.mesh.get_elements_by_type(el_type)[0].reshape(-1,1)
    ElementConnectivity = (gmsh.model.mesh.get_elements_by_type(el_type)[1]).reshape(-1,NodePerEl)-1

    coordsId = gmsh.model.mesh.get_nodes()[0]-1
    #Nodes that are part of elements (to check the mesh)
    included_Nodes = np.isin(coordsId, ElementConnectivity.reshape(-1,)) 
    Node_coords = np.array((gmsh.model.mesh.get_nodes()[1]).reshape(-1,3)[:,:dim],dtype=dtype)
    
    DoFs = np.zeros((ElementConnectivity.shape[0],dim*NodePerEl),dtype=np.int32)
    Element_con = np.array(ElementConnectivity,dtype=np.int32)


    # msh = meshio.read("./Domains/DogBoneNotch_Coarse.msh")
    # Nodes = msh.points
    # Element_c = msh.cells['hexahedron']

    if dim ==2:
        for iEl in range(Element_con.shape[0]):
            a = [[2*Element_con[iEl][i],2*Element_con[iEl][i]+1] for i in range(4)]
            DoFs[iEl] = np.array((a)).reshape(-1,)
    elif dim==3:
        for iEl in range(Element_con.shape[0]):
            a = [[3*Element_con[iEl][i],3*Element_con[iEl][i]+1,3*Element_con[iEl][i]+2] for i in range(8)]
            DoFs[iEl] = np.array((a)).reshape(-1,)
    
    Domain = {'Node_coords':Node_coords,
              'Elements_connectivity':Element_con,
              'Elements_DoFs':DoFs}
    return Domain

#Old version (problem for 3D case )
def Make_mesh(Domain:str,
              dim:int,
              dtype:np.dtype):
    '''
    gmsh element type
    1: Line elements
    2: Triangle elements
    3: Quadrilateral elements
    4: Tetrahedral elements
    5: Hexahedral elements
    '''
    file = f'./microstructure_dataset/{Domain}.geo'
    np.set_printoptions(threshold=sys.maxsize)
    gmsh.open(file)
    gmsh.initialize()
    gmsh.model.occ.synchronize()
    gmsh.model.mesh.generate(dim)

    if dim ==2:
        el_type = 3 #Quad
        NodePerEl = 4
    elif dim ==3:
        el_type = 5 #Hex
        NodePerEl = 8

    NumberOfElements = gmsh.model.mesh.get_elements_by_type(el_type)[0].reshape(-1,1)
    ElementConnectivity = (gmsh.model.mesh.get_elements_by_type(el_type)[1]).reshape(-1,NodePerEl)-1

    coordsId = gmsh.model.mesh.get_nodes()[0]-1
    #Nodes that are part of elements (to check the mesh)
    included_Nodes = np.isin(coordsId, ElementConnectivity.reshape(-1,)) 
    Node_coords = np.array((gmsh.model.mesh.get_nodes()[1]).reshape(-1,3)[:,:dim],dtype=dtype)

    DoFs = np.zeros((ElementConnectivity.shape[0],dim*NodePerEl),dtype=np.int32)
    Element_con = np.array(ElementConnectivity,dtype=np.int32)
    
    if dim ==2:
        for iEl in range(Element_con.shape[0]):
            a = [[2*Element_con[iEl][i],2*Element_con[iEl][i]+1] for i in range(4)]
            DoFs[iEl] = np.array((a)).reshape(1,-1)
    elif dim==3:
        for iEl in range(Element_con.shape[0]):
            a = [[3*Element_con[iEl][i],3*Element_con[iEl][i]+1,3*Element_con[iEl][i]+2] for i in range(8)]
            DoFs[iEl] = np.array((a)).reshape(1,-1)
    
    Domain = {'Node_coords':Node_coords,
              'Elements_connectivity':Element_con,
              'Elements_DoFs':DoFs}
    
    gmsh.clear()
    return Domain


#Helper functions
#------------------------------------------------
#Get the number of triplets for sparse matrix 
def get_max_num_triplets(node_coords:np.ndarray,
                         dofs:np.ndarray,
                         dim:int):
    #Check sparsity
    time_s = time.perf_counter()
    Sparsity = np.zeros((dim*node_coords.shape[0],dim*node_coords.shape[0]))
    for dof in dofs:
        Sparsity[np.ix_(dof,dof)] = np.ones((len(dof),len(dof)))
    max_triplets = np.count_nonzero(Sparsity)
    index_ij = np.stack(np.where(Sparsity==1),axis=1).astype(np.int32)
    print(time.perf_counter()-time_s)
    return max_triplets,index_ij

#Visualization for testing 
def visualize(coords,edges=None):
    fig, ax = plt.subplots(figsize=(6,4), subplot_kw={'projection': '3d'})
    if coords is not None:
        ax.scatter(coords[:,0],coords[:,1],coords[:,2],alpha=1,s=20,c='red')
    if edges is not None:
        for idx,i in enumerate(edges):
            col = 'black'
            x = np.append(coords[i,0],coords[i[0],0])
            y =  np.append(coords[i,1],coords[i[0],1])
            z = np.append(coords[i,2],coords[i[0],2])
            line, = ax.plot(x,y,z,c=col)
  #------------------------------------------------