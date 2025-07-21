# The Gaia AVU-GSR code

Please, to cite this code, use the following bibtex entry:
```bibtex
@article{Malenza25,
	title        = {Performance Portability Assesment in Gaia},
	author       = {Giulio Malenza, Valentina Cesare, Marco Edoardo Santimaria, Robert Birke, Alberto Vecchiato, Ugo Becciani, Marco Aldinucci},
	year         = 2025,
	journal      = {Transactions on Parallel and Distributed Systems (TPDS)},
	doi          = {10.1109/TPDS.2025.3591452}
}
```

Instructions for compilation and execution of the simulator used to test LSQR algorithm for the ESA Gaia mission. 

The folder is organized in the following way:
- src   -> lsqr versions
- include   -> include files needed to compile the code;
- Makefile  -> A Makefile than should be modified to compile the code;
- Makefile.examples -> some exmaples of Makefile that we used to compile the code on different systems;

You should use ```gcc>=12.2.0``` <br />

To run cuda code you can use ```nvcc >=11.8``` NVIDIA compiler <br />
To run hip code you can use ```hipcc (cuda>=11.7,rocm-5.6)``` AMD compiler <br />
To run openmp code you can use ```nvc++>=23.11``` NVIDIA compiler on NVIDIA architecture and ```AMD clang++>=16.0.0``` compiler on AMD architecture <br />
To run sycl code you can use ```acpp>=23.10.0``` AdaptiveCpp  compiler <br />
To run cpp pstl code you can use ```nvc++>=23.11``` compiler on NVIDIA architecture and ```clang++>=18.0.0``` compiler on AMD architecture (roc-stdpar) <br />

## Compilation

### NVIDIA GPU
In the Makefile you need to specify:
- GPUARCH       -> gpu architecture
- MPI_HOME      -> path to MPI 
- GNUTOOLCHAIN  -> path to GCC
- KOKKOS_HOME   -> path to Kokkos compiler

To compile cuda code: ```make clean && make cuda -j``` <br />
To compile hip code: ```make clean && make hip -j``` <br />
To compile sycl code: ```make clean && make sycl -j``` <br />
To compile openmp code: ```make clean && make ompG -j``` <br />
To compile pstl code: ```make clean && make stdparG -j``` <br />



### AMD GPU
In the Makefile you need to specify:
- GPUARCH       -> gpu architecture (possible values: gfx90a, gfx908)
- MPI_HOME      -> path to MPI 
- GNUTOOLCHAIN  -> path to GCC
- ROCM_PATH     -> path to ROCM
- KOKKOS_HOME   -> path to Kokkos compiler

To compile hip code: ```make clean && make hip -j``` <br />
To compile sycl code: ```make clean && make sycl -j``` <br />
To compile openmp code: ```make clean && make ompG -j``` <br />
To compile pstl code: ```make clean  && make stdparG -j``` <br />

### MPI
To compile code enabling MPI set USE_MPI=ON, i.e. ```make cuda USE_MPI=ON```

## TEST
```
mpirun -np 1 ./GaiaGsrParSim.x -memGlobal 2 -IDtest 0 -itnlimit 100 
```
Here:
- memGlobal specifies approximately how much memory the system occupies in GB
- IDtest 0 specifies that the test, if run up to convergence, reaches the identity solution
- itnlimit specifies the maximum number of iterations run by LSQR. This number is not reached if confergence is reached before.

In the scripts directory, some example scripts can be found for reproducibility.

