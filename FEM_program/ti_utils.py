import taichi as ti 

#Taichi helper functions
#------------------------------------
@ti.kernel
def max_norm(arr:ti.types.ndarray()) -> float:  #type: ignore
    rm = 0.0
    for i in range(arr.shape[0]):
        ti.atomic_max(rm, ti.abs(arr[i]))
    return rm

@ti.kernel
def normalized_norm(arr:ti.types.ndarray(),force:ti.types.ndarray()) -> float:  #type: ignore
    num = 0.0
    denom = 1.0
    for i in range(arr.shape[0]):
        num += arr[i]**2
        denom += force[i]**2
    out = num/denom
    return out

@ti.kernel
def add(arr1:ti.types.ndarray(),arr2:ti.types.ndarray()): #type: ignore
    for i in range(arr1.shape[0]):
        arr1[i] += arr2[i]

@ti.kernel
def mult_array_scalar(arr:ti.types.ndarray(),scalar:float): #type: ignore
    for i in range(arr.shape[0]):
        arr[i] *= scalar
#------------------------------------