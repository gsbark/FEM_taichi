import taichi as ti
from FEM_utils import Preprocess_FEM
from ti_FEM import FEM_program
import yaml

# ti.init(arch=ti.cpu,default_fp=ti.f64,default_ip=ti.i32,cpu_max_num_threads=8,kernel_profiler=True)
ti.init(arch=ti.cpu,default_fp=ti.f64,default_ip=ti.i32)

with open('input.yaml', 'r') as file:
    config = yaml.safe_load(file)

Input = Preprocess_FEM(**config)
FEM = FEM_program(Input)
FEM.Run_FEM()

# ti.profiler.print_kernel_profiler_info()