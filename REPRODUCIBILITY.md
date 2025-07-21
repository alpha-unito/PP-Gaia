# Reproducibility initiative

Here is the desciption on the software and hardware configuration used for the experiments.

### Table 1: Hardware and software architectures considered
 
| **Attribute**                            | **H100**                  | **A100**                      | **V100**                   | **T4**                     | **MI250X**                             |
|------------------------------------------|---------------------------|-------------------------------|----------------------------|----------------------------|----------------------------------------|
| **CPU**                                  | NVIDIA Grace CPU          | Intel Xeon Platinum 8358      | Intel Xeon Gold 6230 CPU   | Intel Xeon Gold 6230 CPU   | AMD EPYC 7A53 64-Core Processor        |
| **Main Memory**                          | 573 GB                    | 512 GB                        | 1.5 TB                     | 1.5 TB                     | 512GB                                  |
| **OS**                                   | Ubuntu 22.04              | RHEL 8.8                      | Ubuntu 20.04               | Ubuntu 20.04               | SUSE Linux Enterprise Server           |
| **Kernel**                               | 6.2.0-1008-nvidia-64k     | 4.18.0-477.27.1.el8_8.x86_64  | 5.4.0-177-generic          | 5.4.0-177-generic          | 5.14.21-150500.55.83_13.0.62-cray_shasta_c |
| **GPU**                                  | NVIDIA Hopper GPU         | NVIDIA Ampere                 | NVIDIA Volta               | NVIDIA Tesla               | AMD Instinct CDNA 2                    |
| **Global GPU Memory**                    | 96 GB HBM3                | 64 GB HBM2e                   | 32 GB HBM2                 | 16 GB GDDR6                | 64 GB HBM2e (per GCD)                  |
| **GPU L2 Cache**                         | 60 MB                     | 32 MB                         | 6 MB                       | 4 MB                       | 8 MB (per GCD)                         |
| **GPU Max Clock Rate**                   | 1980 MHz                  | 1395 MHz                      | 1597 MHz                   | 1590 MHz                   | 1700 MHz                               |
| **CUDA Cores / Compute Units**           | 16,896 Cores              | 7,936 Cores                   | 5,120 Cores                | 2,560 Cores                | 110 CU (per GCD)                       |
| **GPU Memory Clock Rate**                | 2619 MHz                  | 1593 MHz                      | 1107 MHz                   | 5001 MHz                   | 1593 MHz                               |
| **GPU Bus Width**                        | 6144-bit                  | 4096-bit                      | 4096-bit                   | 256-bit                    | 4096-bit                               |
| **GPU Memory Channels**                  | 6                         | 4                             | 4                          | 6                          | 4                                      |
| **GPU Driver Version**                   | 545.23.08                 | 530.30.02                     | 550.54.15                  | 550.54.15                  | ROCm 6.0.3                             |
| **CUDA / HIP Version**                   | CUDA 12.3                 | CUDA 12.1                     | CUDA 12.4                  | CUDA 12.4                  | HIP 6.0.32831-204d35d16                |
| **GPU Peak Memory Bandwidth [TB/s]**     | 0.32                      | 1.13                          | 4.02                       | 1.63                       | 1.60                                   |
| **Theoretical FP64 Peak Perf. [TFLOP/s]**| 4.09                      | 8.18                          | 33.5                       | 11.1                       | 23.9                                   |


---

### Table 2: Framework details for NVIDIA and AMD

#### NVIDIA Compilation Flags

| **Framework** | **Compiler** | **Compilation Flags (NVIDIA)** |
|---------------|--------------|---------------------------------|
| CUDA          | `nvc++`      | `-gencode=arch=compute_XX,code=sm_XX` |
| HIP           | `hipcc`      | `--gpu-architecture=sm_XX`     |
| SYCL+A        | `acpp`       | `--acpp-platform=cuda --acpp-targets=cuda:sm_XX --acpp-gpu-arch=sm_XX` |
| SYCL+I        | `clang++`    | `-fsycl -fsycl-targets=nvptx64-nvidia-cuda -Xsycl-target-backend --cuda-gpu-arch=sm_XX` |
| OpenACC       | `nvc++`      | `-acc -gpu=ccXX`               |
| OMP+V         | `nvc++`      | `-mp=gpu -gpu=ccXX,sm_XX`      |
| OMP+LLVM      | `clang++`    | `-fopenmp -fopenmp-targets=nvptx64-nvidia-cuda -march=sm_XX` |
| KOKKOS        | `nvcc`       | `--offload-arch=sm_XX  -fopenmp` |
| PSTL+A        | `acpp`       | `--acpp-platform=cuda --acpp-stdpar --acpp-targets=cuda:sm_XX --acpp-stdpar-unconditional-offload --acpp-gpu-arch=sm_XX` |
| PSTL+V        | `nvc++`      | `-stdpar=gpu -gpu=ccXX,sm_XX`  |

#### AMD Compilation Flags

| **Framework** | **Compiler** | **Compilation Flags (AMD)** |
|---------------|--------------|------------------------------|
| HIP           | `hipcc`      | `--offload-arch=gfxXXX -munsafe-fp-atomics` |
| SYCL+A        | `acpp`       | `--acpp-platform=rocm --acpp-targets=hip --acpp-gpu-arch=gfxXXX -munsafe-fp-atomics` |
| SYCL+I        | `clang++`    | `-fsycl -fsycl-targets=amdgcn-amd-amdhsa -Xsycl-target-backend --offload-arch=gfxXXX -mllvm --amdgpu-oclc-unsafe-fp-atomics=true` |
| OMP+V         | `hipcc`      | `-fopenmp --offload-arch=gfxXXX -munsafe-fp-atomics` |
| OMP+LLVM      | `clang++`    | `-fopenmp -fopenmp-targets=x86_64,amdgcn-amd-amdhsa -march=gfxXXX` |
| KOKKOS        | `hipcc`      | `--offload-arch=gfxXXX  -fopenmp -munsafe-fp-atomics` |
| PSTL+A        | `acpp`       | `--acpp-platform=rocm --acpp-stdpar --acpp-targets=hip:gfxXXX --acpp-stdpar-unconditional-offload --acpp-gpu-arch=gfxXXX -munsafe-fp-atomics` |
| PSTL+V        | `hipcc`      | `--hipstdpar --hipstdpar-path=$(HIPSTDAR_ROOT) --offload-arch=gfxXXX -munsafe-fp-atomics` |

---

### Table 3: Main features of parallel frameworks

|                                | **Low-level** | **Pragma-based**    | **C++ Abstraction** |
|--------------------------------|---------------|----------------------|----------------------|
| **Kernel Tuning**              | Manual        | Automatic/Manual     | Automatic            |
| **Memory Management**          | Manual        | Automatic/Manual     | Automatic            |
| **Shared Memory**              | Manual        | Automatic            | Automatic            |
| **Synchronization**            | Manual        | Automatic            | Automatic            |
| **GPU Streams**                | Available     | Not Available        | Not Available        |
| **Code lines**                 | High          | Medium               | Low                  |
