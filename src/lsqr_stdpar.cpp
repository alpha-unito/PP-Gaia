/* 

CPP PSTL Version

*/


#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stdbool.h>
#include <math.h>
#include <time.h>

#include "util.h"
#include <cassert>

#include <limits>
#include <sys/time.h>

#include "lsqr.h"


#include <execution>
#include <ranges>
#include <algorithm>
#include <numeric>
#include <iostream>


#ifdef USE_MPI
    #include <mpi.h>
#endif


#if defined(TBB) 
#include "counting_iterator.hpp"
#endif

#if defined(__NVIDIA90__) || defined(__NVIDIA80__) || defined(__NVIDIA70__)
    #if defined(KERNELTIME) 
        #include <cuda_runtime.h>
    #endif
#elif defined(__MI250X__) || defined(__MI100X__)
    #if defined(KERNELTIME) 
        #include <hip/hip_runtime.h>
    #endif
#endif


#ifdef _NVHPC_STDPAR_GPU
    #include <cuda/atomic>
    #include <thrust/iterator/counting_iterator.h>
    template<typename T>
    using counting_iterator = thrust::counting_iterator<T>;
    #define atomicAdd(x, y) (atomicAdd(x, y))
#elif defined(__HIPSTDPAR__) || defined(__HIPCC__)
    #include "hip/hip_runtime.h"
    #include "hip/amd_detail/amd_hip_atomic.h"
    #include "hip/amd_detail/amd_hip_unsafe_atomics.h"
    #include <thrust/iterator/counting_iterator.h>
    template<typename T>
    using counting_iterator = thrust::counting_iterator<T>;
    #define atomicAdd(x, y) (__hip_atomic_fetch_add(x, y, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT))

#else
    #include "counting_iterator.hpp"
    #include <atomic>
    #include <CL/sycl.hpp>
    #define atomicAdd(x, y) (cl::sycl::atomic_ref<double, cl::sycl::memory_order::relaxed, cl::sycl::memory_scope::device, cl::sycl::access::address_space::global_space>(*(x)) +=(y))
#endif


// execution policies
#define POL std::execution::par_unseq


#define ZERO   0.0
#define ONE    1.0
#define MONE   -1.0


#if defined(__HIPSTDPAR__) || defined(__HIPCC__)

inline
hipError_t checkHip(hipError_t result)
{
#if defined(DEBUG) || defined(_DEBUG)
  if (result != hipSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", 
            hipGetErrorString(result));
    assert(result == hipSuccess);
  }
#endif
  return result;
}

#endif

//-----------------------------------------------------------------------------------------------------
//-----------------------------------------------------------------------------------------------------
//-----------------------------------------------------------------------------------------------------
inline double d2norm(const double& a, const double& b){

    double scale=std::fabs(a)+std::fabs(b);
    return (!scale) ? ZERO : scale * std::sqrt((a/scale)*(a/scale)+(b/scale)*(b/scale)); 

}

template<typename L>
inline void dload(double* x, const L n, const double alpha){

    std::fill_n(POL,x,n,alpha);


}

template<typename L>
inline void dscal(double* array, const double val, const L N, const double sign ){
    std::transform(POL,
                        array,
                        array+N,
                        array,
                        [=](auto arrayi){
                            return sign*(arrayi*val);
                        });
}

template<typename L>
inline double maxCommMultiBlock_double(const double* __restrict__ gArr, const L arraySize){

        double red=std::transform_reduce(
                    POL,
                    gArr, gArr+arraySize,
                    -std::numeric_limits<double>::infinity(), 
                    [=](double a, double b) {
                        return std::max(a,b); 
                    },
                    [=](double it) {
                        return std::fabs(it); 
                    }
                );
        return std::fabs(red);

}

template<typename L>
inline double sumCommMultiBlock_double(const double* __restrict__ gArr, const L arraySize, const double max){

    const double d{1/max};
    return std::transform_reduce(POL,
                                gArr,
                                gArr+arraySize,
                                ZERO,
                                std::plus<double>{},
                                [=](const auto gArri){
                                    return (gArri*d)*(gArri*d);
                                });
}

template<typename L>
void vVect_Put_To_Zero_Kernel (double* vVect_dev, const L localAstroMax, const L nunkSplit)
{
    std::fill_n(POL, vVect_dev+localAstroMax, nunkSplit-localAstroMax, ZERO);

}

void cblas_dcopy_kernel(double* __restrict__ wVect_dev, const double* __restrict__ vVect_dev, const long nunkSplit)
{
    std::copy(POL,vVect_dev, vVect_dev + nunkSplit,wVect_dev);    
}

template<typename L, typename I>
void kAuxcopy_Kernel (double* __restrict__ knownTerms_dev, double* __restrict__ kAuxcopy_dev, const L nobs, const I N)
{
    std::copy_n(POL,knownTerms_dev+nobs,N,kAuxcopy_dev);
    std::fill_n(POL,knownTerms_dev+nobs,N,ZERO);
}

template<typename L>
void vAuxVect_Kernel (double* __restrict__ vVect_dev, double* __restrict__ vAuxVect_dev, const L N)
{

    std::copy(POL, vVect_dev, vVect_dev+N, vAuxVect_dev);
    std::fill(POL, vVect_dev, vVect_dev+N, ZERO);

}

//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
//                                                                                    APROD1
//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
void aprod1_Kernel_astro(double* __restrict__ knownTerms_dev, 
                        const double* __restrict__ systemMatrix_dev, 
                        const double* __restrict__ vVect_dev, 
                        const long*  __restrict__ matrixIndexAstro, 
                        const long offLocalAstro, 
                        const long nobs, 
                        const short nAstroPSolved){

    const auto start=counting_iterator<long>(0);
    const auto end=counting_iterator<long>(nobs);
    
    std::for_each(POL, start, end,[=](const auto ix){
            double sum{ZERO};
            const long jstartAstro = matrixIndexAstro[ix] - offLocalAstro;
            
            for(short jx = 0; jx <   nAstroPSolved; ++jx) {
                double matrixVal = systemMatrix_dev[ix * nAstroPSolved + jx];  
                double vectVal = vVect_dev[jstartAstro + jx];                  
                sum += matrixVal * vectVal;  
            }
            knownTerms_dev[ix]+=sum;
        });
}


void aprod1_Kernel_att_AttAxis(double*  __restrict__ knownTerms_dev, 
                                const double*  __restrict__ systemMatrix_dev, 
                                const double*  __restrict__ vVect_dev, 
                                const long*  __restrict__ matrixIndexAtt, 
                                const long  nAttP, 
                                const long nobs, 
                                const long nDegFreedomAtt, 
                                const long offLocalAtt, 
                                const short nAttParAxis)
{
    const auto start=counting_iterator<long>(0);
    const auto end=counting_iterator<long>(nobs);

    
    std::for_each(POL, start, end,[=](const auto& ix){
        double sum{ZERO};
        long jstartAtt{matrixIndexAtt[ix] + offLocalAtt}; 
        long tmp2=static_cast<long>(ix*nAttP);

        
        for(short inpax = 0;inpax<nAttParAxis; ++inpax)
            sum += systemMatrix_dev[tmp2 + inpax ] * vVect_dev[jstartAtt + inpax];
        jstartAtt += nDegFreedomAtt;
        tmp2+=nAttParAxis;
        
        for(short inpax = 0;inpax<nAttParAxis;inpax++)
            sum+=  systemMatrix_dev[tmp2+inpax] * vVect_dev[jstartAtt+inpax];
        jstartAtt += nDegFreedomAtt;
        tmp2+=nAttParAxis;
        
        for(short inpax = 0;inpax<nAttParAxis;inpax++)
            sum+= systemMatrix_dev[tmp2+inpax] * vVect_dev[jstartAtt+inpax];

        knownTerms_dev[ix]+=sum;

    });
}




void aprod1_Kernel_instr (double*  __restrict__ knownTerms_dev, 
                            const double*  __restrict__ systemMatrix_dev, 
                            const double*  __restrict__ vVect_dev, 
                            const int*  __restrict__ instrCol_dev, 
                            const long nobs, 
                            const long offLocalInstr, 
                            const short nInstrPSolved){

    const auto start=counting_iterator<long>(0);
    const auto end=counting_iterator<long>(nobs);
    
    std::for_each(POL, start, end ,[=](const auto& ix){
        double sum{ZERO};
        const long iiVal{ix*nInstrPSolved};
        long ixInstr{0};
        
        for(short inInstr=0;inInstr<nInstrPSolved;inInstr++){
            ixInstr=offLocalInstr+instrCol_dev[iiVal+inInstr];
            sum += systemMatrix_dev[ix * nInstrPSolved + inInstr]*vVect_dev[ixInstr];
        }
        knownTerms_dev[ix]+=sum;
    });
}


void aprod1_Kernel_glob(double*  __restrict__ knownTerms_dev, 
                        const double*  __restrict__ systemMatrix_dev, 
                        const double*  __restrict__ vVect_dev, 
                        const long offLocalGlob, 
                        const long nobs, 
                        const short nGlobP){

    const auto start=counting_iterator<long>(0);
    const auto end=counting_iterator<long>(nobs);

    std::for_each(POL, start, end ,[=](const auto& ix){
        double sum{ZERO};
        
        for(short inGlob=0;inGlob<nGlobP;inGlob++){
            sum+=systemMatrix_dev[ix * nGlobP + inGlob]*vVect_dev[offLocalGlob+inGlob];
        }
        knownTerms_dev[ix]+=sum;
    });
}




/// ExtConstr
void aprod1_Kernel_ExtConstr(double*   knownTerms_dev, 
                            const double*   systemMatrix_dev, 
                            const double*   vVect_dev, 
                            const long VrIdAstroPDimMax, 
                            const long nobs, 
                            const long nDegFreedomAtt, 
                            const int startingAttColExtConstr, 
                            const int nEqExtConstr, 
                            const int nOfElextObs, 
                            const int numOfExtStar, 
                            const int numOfExtAttCol, 
                            const short nAstroPSolved, 
                            const short nAttAxes){

    const long offExtAttConstr{VrIdAstroPDimMax*nAstroPSolved+startingAttColExtConstr};

    const auto start=counting_iterator<int>(0);
    const auto end=counting_iterator<int>(nEqExtConstr);
    
    std::for_each(POL, start, end,[=](const auto& iexc){

            double sum{ZERO};
            const long offExtConstr{iexc*nOfElextObs};
            for(long j3=0;j3<numOfExtStar*nAstroPSolved;j3++){
                sum += systemMatrix_dev[offExtConstr+j3]*vVect_dev[j3];
            }
            for (int nax = 0; nax < nAttAxes; nax++) {
                const long offExtAtt{offExtConstr + numOfExtStar*nAstroPSolved + nax*numOfExtAttCol};
                const long vVIx{offExtAttConstr+nax*nDegFreedomAtt};

                for(long j3=0;j3<numOfExtAttCol;j3++){
                    sum += systemMatrix_dev[offExtAtt+j3]*vVect_dev[vVIx+j3];
                }
            }
            atomicAdd(&knownTerms_dev[nobs+iexc], sum);

    });
}


 
// /// BarConstr
void aprod1_Kernel_BarConstr(double*   knownTerms_dev, 
                            const double*   systemMatrix_dev, 
                            const double*   vVect_dev, 
                            const int nOfElextObs, 
                            const int nOfElBarObs, 
                            const int nEqExtConstr, 
                            const long nobs, 
                            const int nEqBarConstr, 
                            const int numOfBarStar, 
                            const short nAstroPSolved){

    const long ktIx{nobs + nEqExtConstr};
    const auto start=counting_iterator<int>(0);
    const auto end=counting_iterator<int>(numOfBarStar*nAstroPSolved);

    std::for_each(POL, start, end,[=](const auto& j3){
        for(int iexc=0;iexc<nEqBarConstr;iexc++ ){
            double sum{ZERO};
            const long offBarConstrIx=nOfElBarObs*iexc;
            sum = sum + systemMatrix_dev[offBarConstrIx+j3]*vVect_dev[j3];
            atomicAdd(&knownTerms_dev[ktIx+iexc], sum);
        }
    });  


}



/// InstrConstr
void aprod1_Kernel_InstrConstr(double*   knownTerms_dev, 
                                const double*   systemMatrix_dev, 
                                const double*   vVect_dev, 
                                const int*   instrConstrIlung_dev, 
                                const int*   instrCol_dev, 
                                const long VrIdAstroPDimMax, 
                                const long nobs, 
                                const long nDegFreedomAtt, 
                                const int nOfElextObs, 
                                const int nEqExtConstr, 
                                const int nOfElBarObs, 
                                const int nEqBarConstr, 
                                const int myid, 
                                const int nOfInstrConstr, 
                                const int nproc, 
                                const short nAstroPSolved, 
                                const short nAttAxes, 
                                const short nInstrPSolved){

    const long offSetInstrConstr{nOfElextObs*nEqExtConstr+nOfElBarObs*nEqBarConstr};
    const long offSetInstrConstr1{VrIdAstroPDimMax*nAstroPSolved+nDegFreedomAtt*nAttAxes};
    const long ktIx{nobs+nEqExtConstr+nEqBarConstr};
    
    if(myid<nOfInstrConstr){
    const int nel{(nOfInstrConstr-myid-1)/nproc};
    const auto start=counting_iterator<int>(0);
    const auto end=counting_iterator<int>(nel+1);

    std::vector<int> iter(nel+1);
    std::fill(POL,iter.begin(),iter.end(),myid);
    std::transform(POL,start,end,iter.begin(),iter.begin(),[nproc](auto x, auto y){return y+nproc*x; });

    std::for_each(POL, iter.cbegin(), iter.cend(),[=](const auto& i1){ // problem with range-v3 and nvhcp
        double sum{ZERO};
        long offSetInstrInc{offSetInstrConstr};
        int offSetInstr{0};
        for(int m=0;m<i1;m++) // could be an inclusive scan
        {
            offSetInstrInc+=instrConstrIlung_dev[m];
            offSetInstr+=instrConstrIlung_dev[m];
        }
        const long offvV{nobs*nInstrPSolved+offSetInstr};
        for(int j3 = 0; j3 < instrConstrIlung_dev[i1]; j3++)
            sum+=systemMatrix_dev[offSetInstrInc+j3]*vVect_dev[offSetInstrConstr1+instrCol_dev[offvV+j3]];
        atomicAdd(&knownTerms_dev[ktIx+i1], sum);
    });
    }
}



//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
//                                                                                    APROD2
//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


void aprod2_Kernel_astro(double*  __restrict__ vVect_dev, 
                        const double*  __restrict__ systemMatrix_dev, 
                        const double*  __restrict__ knownTerms_dev, 
                        const long*  __restrict__ matrixIndexAstro, 
                        const long*  __restrict__ startend, 
                        const long offLocalAstro, 
                        const long nobs, 
                        const short nAstroPSolved){

    const auto start=counting_iterator<long>(0);
    const auto end=counting_iterator<long>(nobs);
    
    std::for_each(POL, start, end,[=](const long ix){
        long stdix_start=startend[ix];
        long stdix_end=startend[ix+1];
        long tid=matrixIndexAstro[stdix_start]-offLocalAstro;
        for(long i=stdix_start; i<stdix_end; ++i){
            double tmp=knownTerms_dev[i];
            
            for (short jx = 0; jx < nAstroPSolved; jx++)
                vVect_dev[tid + jx]+= systemMatrix_dev[i*nAstroPSolved + jx] * tmp;
        }
    });
}






void aprod2_Kernel_att_AttAxis(double * __restrict__ vVect_dev, 
                                const double * __restrict__ systemMatrix_dev, 
                                const double * __restrict__ knownTerms_dev, 
                                const long*  __restrict__ matrixIndexAtt, 
                                const short nAttP, 
                                const long nDegFreedomAtt, 
                                const long offLocalAtt, 
                                const long nobs, 
                                const short nAstroPSolved, 
                                const short nAttParAxis)
{
    const auto start=counting_iterator<long>(0);
    const auto end=counting_iterator<long>(nobs);

    std::for_each(POL, start, end,[=](const auto& ix){
        const long jstartAtt = matrixIndexAtt[ix] + offLocalAtt;
        const long tmp2=static_cast<long>(ix*nAttP);
        
        for (short inpax = 0; inpax < nAttParAxis; inpax++){
            atomicAdd(&vVect_dev[jstartAtt+inpax],systemMatrix_dev[tmp2+inpax]*knownTerms_dev[ix]);
        }
        
        for (short inpax = 0; inpax < nAttParAxis; inpax++){
            atomicAdd(&vVect_dev[jstartAtt+nDegFreedomAtt+inpax],systemMatrix_dev[tmp2+nAttParAxis+inpax]*knownTerms_dev[ix]);
        }
        
        for (short inpax = 0; inpax < nAttParAxis; inpax++){
            atomicAdd(&vVect_dev[jstartAtt+nDegFreedomAtt+nDegFreedomAtt+inpax],systemMatrix_dev[tmp2+nAttParAxis+nAttParAxis+inpax]*knownTerms_dev[ix]);
        }
    });
}

void aprod2_Kernel_instr(double * __restrict__ vVect_dev,
                        const double * __restrict__ systemMatrix_dev, 
                        const double * __restrict__ knownTerms_dev, 
                        const int * __restrict__ instrCol_dev, 
                        const long offLocalInstr, 
                        const long nobs,  
                        const short nInstrPSolved){

    const auto start=counting_iterator<long>(0);
    const auto end=counting_iterator<long>(nobs);

    std::for_each(POL, start, end,[=](const auto& ix){
        
        for (short inInstr = 0; inInstr < nInstrPSolved; inInstr++){
            atomicAdd(&vVect_dev[offLocalInstr + instrCol_dev[ix*nInstrPSolved+inInstr]],systemMatrix_dev[ix*nInstrPSolved+inInstr]*knownTerms_dev[ix]);
        }
    });
}




void aprod2_Kernel_ExtConstr(double* vVect_dev, 
                            const double* systemMatrix_dev, 
                            const double* knownTerms_dev, 
                            const long nobs, 
                            const long nDegFreedomAtt, 
                            const long VrIdAstroPDimMax, 
                            const int nEqExtConstr, 
                            const int nOfElextObs, 
                            const int numOfExtStar, 
                            const int startingAttColExtConstr, 
                            const int numOfExtAttCol, 
                            const short nAstroPSolved, 
                            const short nAttAxes){

    const long off1{VrIdAstroPDimMax*nAstroPSolved+startingAttColExtConstr};

    const auto start=counting_iterator<int>(0);
    const auto end=counting_iterator<int>(numOfExtStar);


    std::for_each(POL,start, end ,[=](const auto& i){
        const long off3{i*nAstroPSolved};        
        for(int ix = 0; ix < nEqExtConstr; ++ix){  
            const double yi{knownTerms_dev[nobs + ix]};
            const long offExtStarConstrEq{ix*nOfElextObs};
            const long off2{offExtStarConstrEq + off3};
                for(int j2 = 0; j2 < nAstroPSolved; ++j2){
                    vVect_dev[j2+off3] += systemMatrix_dev[off2+j2]*yi;
                }
            
        } 
    });



    const auto start2=counting_iterator<int>(0);
    const auto end2=counting_iterator<int>(numOfExtAttCol);

    std::for_each(POL, start2, end2,[=](const auto& i){
        for(int ix=0;ix<nEqExtConstr;ix++ ){  
            const double yi = knownTerms_dev[nobs + ix];
            const long offExtAttConstrEq{ix*nOfElextObs + numOfExtStar*nAstroPSolved}; 
            for(int nax = 0; nax < nAttAxes; nax++){
                const long off2{offExtAttConstrEq+nax*numOfExtAttCol};
                const long offExtUnk{off1 + nax*nDegFreedomAtt};
                vVect_dev[offExtUnk+i] = vVect_dev[offExtUnk+i] + systemMatrix_dev[off2+i]*yi;
            }
        }
    });


}



void aprod2_Kernel_BarConstr(double* vVect_dev, 
                            const double* systemMatrix_dev, 
                            const double* knownTerms_dev, 
                            const long nobs, 
                            const int nEqBarConstr, 
                            const int nEqExtConstr, 
                            const int nOfElextObs, 
                            const int nOfElBarObs, 
                            const int& numOfBarStar, 
                            const short nAstroPSolved){

    const auto start=counting_iterator<int>(0);
    const auto end=counting_iterator<int>(numOfBarStar);

    std::for_each(POL,start,end,[=](const auto& yx){
        for(int ix=0;ix<nEqBarConstr;++ix){  
            const double yi{knownTerms_dev[nobs+nEqExtConstr+ix]};
            const long offBarStarConstrEq{nEqExtConstr*nOfElextObs+ix*nOfElBarObs};//Star offset on the row i2 (all the row)
            for(auto j2=0;j2<nAstroPSolved;j2++){
                vVect_dev[yx*nAstroPSolved + j2] += systemMatrix_dev[offBarStarConstrEq+yx*nAstroPSolved+j2]*yi;
            }
        }
    });

}


void aprod2_Kernel_InstrConstr(double* vVect, 
                                const double* systemMatrix, 
                                const double* knownTerms, 
                                const int* instrConstrIlung, 
                                const int* instrCol, 
                                const long VrIdAstroPDimMax, 
                                const long nDegFreedomAtt, 
                                const long mapNoss, 
                                const int nEqExtConstr, 
                                const int nEqBarConstr, 
                                const int nOfElextObs, 
                                const int nOfElBarObs, 
                                const int myid, 
                                const int nOfInstrConstr, 
                                const int nproc, 
                                const short nInstrPSolved, 
                                const short nAstroPSolved, 
                                const short nAttAxes){

    if(myid<nOfInstrConstr){

    const int nel{(nOfInstrConstr-myid-1)/nproc};
    const auto start=counting_iterator<int>(0);
    const auto end=counting_iterator<int>(nel+1);

    std::vector<int> iter(nel+1);
    std::fill(POL,iter.begin(),iter.end(),myid);
    std::transform(POL,start,end,iter.begin(),iter.begin(),[nproc](auto x, auto y){return y+nproc*x; });

    const long off3{nOfElextObs*nEqExtConstr+nOfElBarObs*nEqBarConstr};
    const long offInstrUnk{VrIdAstroPDimMax*nAstroPSolved+nDegFreedomAtt*nAttAxes};
    const long off2{mapNoss+nEqExtConstr+nEqBarConstr};
    const long off4{mapNoss*nInstrPSolved};


    std::for_each(POL, iter.cbegin(), iter.cend(),[=](const auto& k1_Aux){ // problem with range-v3 and nvhcp
        const double yi{knownTerms[off2+k1_Aux]};
        int offSetInstr=0;
        for(int m=0;m<k1_Aux;++m){
            offSetInstr=offSetInstr+instrConstrIlung[m];
        }
        const long off1{off3+offSetInstr};
        const long off5{off4+offSetInstr};
        for(int j=0;j<instrConstrIlung[k1_Aux];j++){
            double tmp=systemMatrix[off1+j]*yi;
            atomicAdd(&vVect[offInstrUnk+instrCol[off5+j]], tmp);
        }
    });
    }



}

//-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
//-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
//-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
//-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
//-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
// ---------------------------------------------------------------------
// LSQR
// ---------------------------------------------------------------------
void lsqr(
          long int m,
          long int n,
          double damp,
          double *knownTerms,     
          double *vVect,     
          double *wVect,     
          double *xSolution,     
          double *standardError,    
          double atol,
          double btol,
          double conlim,
          int    itnlim,
          // The remaining variables are output only.
          int    *istop_out,
          int    *itn_out,
          double *anorm_out,
          double *acond_out,
          double *rnorm_out,
          double *arnorm_out,
          double *xnorm_out,
          double *sysmatAstro,
          double *sysmatAtt,
          double *sysmatInstr,
          double *sysmatGloB,
          double *sysmatConstr,
          long *matrixIndexAstro, 
          long *matrixIndexAtt, 
          int *instrCol,
          int *instrConstrIlung,
	  struct comData comlsqr){
    //     ------------------------------------------------------------------
    //
    //     LSQR  finds a solution x to the following problems:
    //
    //     1. Unsymmetric equations --    solve  A*x = b
    //
    //     2. Linear least squares  --    solve  A*x = b
    //                                    in the least-squares sense
    //
    //     3. Damped least squares  --    solve  (   A    )*x = ( b )
    //                                           ( damp*I )     ( 0 )
    //                                    in the least-squares sense
    //
    //     where A is a matrix with m rows and n columns, b is an
    //     m-vector, and damp is a scalar.  (All quantities are real.)
    //     The matrix A is intended to be large and sparse.  It is accessed
    //     by means of subroutine calls of the form
    //
    //     References
    //     ----------
    //
    //     C.C. Paige and M.A. Saunders,  LSQR: An algorithm for sparse
    //          linear equations and sparse least squares,
    //          ACM Transactions on Mathematical Software 8, 1 (March 1982),
    //          pp. 43-71.
    //
    //     C.C. Paige and M.A. Saunders,  Algorithm 583, LSQR: Sparse
    //          linear equations and least-squares problems,
    //          ACM Transactions on Mathematical Software 8, 2 (June 1982),
    //          pp. 195-209.
    //
    //     C.L. Lawson, R.J. Hanson, D.R. Kincaid and F.T. Krogh,
    //          Basic linear algebra subprograms for Fortran usage,
    //          ACM Transactions on Mathematical Software 5, 3 (Sept 1979),
    //          pp. 308-323 and 324-325.
    //     ------------------------------------------------------------------
    //
    //
    //     LSQR development:
    //     22 Feb 1982: LSQR sent to ACM TOMS to become Algorithm 583.
    //     15 Sep 1985: Final F66 version.  LSQR sent to "misc" in netlib.
    //     13 Oct 1987: Bug (Robert Davies, DSIR).  Have to delete
    //                     if ( (one + dabs(t)) .le. one ) GO TO 200
    //                  from loop 200.  The test was an attempt to reduce
    //                  underflows, but caused w(i) not to be updated.
    //     17 Mar 1989: First F77 version.
    //     04 May 1989: Bug (David Gay, AT&T).  When the second beta is zero,
    //                  rnorm = 0 and
    //                  test2 = arnorm / (anorm * rnorm) overflows.
    //                  Fixed by testing for rnorm = 0.
    //     05 May 1989: Sent to "misc" in netlib.
    //     14 Mar 1990: Bug (John Tomlin via IBM OSL testing).
    //                  Setting rhbar2 = rhobar**2 + dampsq can give zero
    //                  if rhobar underflows and damp = 0.
    //                  Fixed by testing for damp = 0 specially.
    //     15 Mar 1990: Converted to lower case.
    //     21 Mar 1990: d2norm introduced to avoid overflow in numerous
    //                  items like  c = sqrt( a**2 + b**2 ).
    //     04 Sep 1991: wantse added as an argument to LSQR, to make
    //                  standard errors optional.  This saves storage and
    //                  time when se(*) is not wanted.
    //     13 Feb 1992: istop now returns a value in [1,5], not [1,7].
    //                  1, 2 or 3 means that x solves one of the problems
    //                  Ax = b,  min norm(Ax - b)  or  damped least squares.
    //                  4 means the limit on cond(A) was reached.
    //                  5 means the limit on iterations was reached.
    //     07 Dec 1994: Keep track of dxmax = max_k norm( phi_k * d_k ).
    //                  So far, this is just printed at the end.
    //                  A large value (relative to norm(x)) indicates
    //                  significant cancellation in forming
    //                  x  =  D*f  =  sum( phi_k * d_k ).
    //                  A large column of D need NOT be serious if the
    //                  corresponding phi_k is small.
    //     27 Dec 1994: Include estimate of alfa_opt in iteration log.
    //                  alfa_opt is the optimal scale factor for the
    //                  residual in the "augmented system", as described by
    //                  A. Bjorck (1992),
    //                  Pivoting and stability in the augmented system method,
    //                  in D. F. Griffiths and G. A. Watson (eds.),
    //                  "Numerical Analysis 1991",
    //                  Proceedings of the 14th Dundee Conference,
    //                  Pitman Research Notes in Mathematics 260,
    //                  Longman Scientific and Technical, Harlow, Essex, 1992.
    //     14 Apr 2006: "Line-by-line" conversion to ISO C by
    //                  Michael P. Friedlander.
    //
    //
    //     Michael A. Saunders                  mike@sol-michael.stanford.edu
    //     Dept of Operations Research          na.Msaunders@na-net.ornl.gov
    //     Stanford University
    //     Stanford, CA 94305-4022              (415) 723-1875
    //-----------------------------------------------------------------------

    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::: TIME 1
    time_point startCycleTime;
    time_point endCycleTime;
    #ifdef USE_MPI
        time_point starttime;
        time_point endtime;
        time_point communicationtime;
        time_point totTimeIteration=ZERO;
    #else
        std::chrono::duration<double> totTimeIteration(ZERO);
    #endif


    const int myid=comlsqr.myid;
    const long mapNoss=static_cast<long>(comlsqr.mapNoss[myid]);

    //-----------------------------------------------------------------------
    ///////////// Specific definitions

    ////////////////////////////////	
    //  Initialize.
    const long VrIdAstroPDimMax=comlsqr.VrIdAstroPDimMax; 
    const long VrIdAstroPDim=comlsqr.VrIdAstroPDim;  
    const long nDegFreedomAtt=comlsqr.nDegFreedomAtt;
    const long localAstro=VrIdAstroPDim*comlsqr.nAstroPSolved;
    const long offsetAttParam = comlsqr.offsetAttParam;
    const long offsetInstrParam = comlsqr.offsetInstrParam;
    const long offsetGlobParam = comlsqr.offsetGlobParam;  
    const long nunkSplit=comlsqr.nunkSplit;
    const long offLocalAstro = comlsqr.mapStar[myid][0] * comlsqr.nAstroPSolved;
    const long localAstroMax = VrIdAstroPDimMax * comlsqr.nAstroPSolved; 
    const long offLocalInstr = offsetInstrParam + (localAstroMax - offsetAttParam); 
    const long offLocalGlob = offsetGlobParam + (localAstroMax - offsetAttParam); 
    const long offLocalAtt = localAstroMax - offsetAttParam; 
    
    const int nEqExtConstr=comlsqr.nEqExtConstr;
    const int nEqBarConstr=comlsqr.nEqBarConstr;
    const int nOfInstrConstr=comlsqr.nOfInstrConstr;
    const int nproc=comlsqr.nproc;
    const int nAttParam=comlsqr.nAttParam; 
    const int nInstrParam=comlsqr.nInstrParam; 
    const int nGlobalParam=comlsqr.nGlobalParam; 
    const int numOfExtStar=comlsqr.numOfExtStar;
    const int numOfBarStar=comlsqr.numOfBarStar;
    const int numOfExtAttCol=comlsqr.numOfExtAttCol;
    const int nElemIC=comlsqr.nElemIC;
    const int nOfElextObs = comlsqr.nOfElextObs;
    const int nOfElBarObs = comlsqr.nOfElBarObs;
    const int startingAttColExtConstr = comlsqr.startingAttColExtConstr;


    const short nAttP=comlsqr.nAttP;
    const short nAttAxes=comlsqr.nAttAxes;
    const short nInstrPSolved=comlsqr.nInstrPSolved;
    const short nAttParAxis = comlsqr.nAttParAxis;
    const short nAstroPSolved=comlsqr.nAstroPSolved;
    const short nGlobP = comlsqr.nGlobP;



    long nElemKnownTerms = mapNoss+nEqExtConstr+nEqBarConstr+nOfInstrConstr;



    double max_knownTerms{ZERO};
    double ssq_knownTerms{ZERO};
    double max_vVect{ZERO};
    double ssq_vVect{ZERO};

    double alpha{ZERO};
    double alphaLoc{ZERO};
    double alphaLoc2{ZERO};
    double temp{ZERO};


    double beta{ZERO};
    double betaLoc{ZERO};
    double betaLoc2{ZERO};

    double anorm{ZERO};
    double acond{ZERO};
    double rnorm{ZERO};
    double arnorm{ZERO};
    double xnorm{ZERO};
    double rhobar{ZERO};
    double phibar{ZERO};
    double bnorm{ZERO};

    
    double rhbar1,cs1,sn1,psi,rho,cs,sn,theta,phi,tau,t1,t2,t3,dknorm;
    double dnorm, dxk, dxmax, delta, gambar, rhs, zbar, gamma, cs2, sn2,z,xnorm1;
    double res2, test1,test3,rtol,ctol;
    double test2{ZERO};

    int itn=0,istop=0;
    int nAstroElements;

    long  other=nAttParam + nInstrParam + nGlobalParam; 

    const bool damped = damp > ZERO;
    const bool wantse = standardError != NULL;

    double t{ZERO};

    #if defined(__HIPSTDPAR__) || defined(__HIPCC__)
        
        int deviceCount = 0;
        checkHip( hipGetDeviceCount(&deviceCount) );
        checkHip( hipSetDevice(myid % deviceCount) );
        
    #endif

    #ifdef USE_MPI
        double* knownTerms_loc = new double[nEqExtConstr+nEqBarConstr+nOfInstrConstr]();
    #endif

    double* kAuxcopy = new double[nEqExtConstr+nEqBarConstr+nOfInstrConstr]();
    double* vAuxVect = new double[localAstroMax](); 

    //--------------------------------------------------------------------------------------------------------
    long nnz=1;
    for(long i=0; i<mapNoss-1; i++){
        if(matrixIndexAstro[i]!=matrixIndexAstro[i+1]){
            nnz++;
        }
    }


    long *startend=(long*)malloc(sizeof(long)*(nnz+1));

    long count=0; nnz=0;
    startend[nnz]=count;nnz++;
    for(long i=0; i<mapNoss-1; ++i){
        if(matrixIndexAstro[i]!=matrixIndexAstro[i+1]){
            count++;
            startend[nnz]=count;
            nnz++;
        }else{
            count++;
        }
    }
    startend[nnz]=count+1;

    //--------------------------------------------------------------------------------------------------------

    comlsqr.itn=itn;


    itn    =   0;
    istop  =   0;

    ctol   =   ZERO;
    if (conlim > ZERO) ctol = ONE / conlim;
    anorm  =   ZERO;
    acond  =   ZERO;
    dnorm  =   ZERO;
    dxmax  =   ZERO;
    res2   =   ZERO;
    psi    =   ZERO;
    xnorm  =   ZERO;
    xnorm1 =   ZERO;
    cs2    = - ONE;
    sn2    =   ZERO;
    z      =   ZERO;

    //  ------------------------------------------------------------------
    //  Set up the first vectors u and v for the bidiagonalization.
    //  These satisfy  beta*u = b,  alpha*v = A(transpose)*u.
    //  ------------------------------------------------------------------


    dload(vVect, nunkSplit, ZERO);

    dload(xSolution,nunkSplit, ZERO);

    if ( wantse )   dload(standardError,nunkSplit, ZERO );

    max_knownTerms = maxCommMultiBlock_double(knownTerms,nElemKnownTerms);

    if(!myid){
        ssq_knownTerms = sumCommMultiBlock_double(knownTerms,nElemKnownTerms, max_knownTerms);
        betaLoc = max_knownTerms*std::sqrt(ssq_knownTerms);
    }else{
        ssq_knownTerms = sumCommMultiBlock_double(knownTerms, mapNoss, max_knownTerms);
        betaLoc = max_knownTerms*std::sqrt(ssq_knownTerms);
    }

    betaLoc2=betaLoc*betaLoc;

    #ifdef USE_MPI
    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    //------------------------------------------------------------------------------------------------  TIME 2
        starttime=get_time();
        MPI_Allreduce(MPI_IN_PLACE,&betaLoc2,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
        endtime=get_time();
        communicationtime=compute_time(endtime,starttime);
    //------------------------------------------------------------------------------------------------
    #endif

    beta=std::sqrt(betaLoc2);


    if(beta > ZERO){

        dscal(knownTerms,1.0/beta,nElemKnownTerms,ONE);
        if(myid) vVect_Put_To_Zero_Kernel(vVect,localAstroMax,nunkSplit);

        //APROD2 CALL BEFORE LSQR
        if(nAstroPSolved) aprod2_Kernel_astro(vVect, sysmatAstro, knownTerms, matrixIndexAstro, startend, offLocalAstro, nnz, nAstroPSolved);
        if(nAttP) aprod2_Kernel_att_AttAxis(vVect,sysmatAtt,knownTerms,matrixIndexAtt,nAttP,nDegFreedomAtt,offLocalAtt,mapNoss,nAstroPSolved,nAttParAxis);
        if(nInstrPSolved) aprod2_Kernel_instr(vVect,sysmatInstr, knownTerms, instrCol, offLocalInstr, mapNoss, nInstrPSolved);
        //APROD2 CONSTRAINTS
        if(nEqExtConstr) aprod2_Kernel_ExtConstr(vVect,sysmatConstr,knownTerms,mapNoss,nDegFreedomAtt,VrIdAstroPDimMax,nEqExtConstr,nOfElextObs,numOfExtStar,startingAttColExtConstr,numOfExtAttCol,nAstroPSolved,nAttAxes);
        if(nEqBarConstr) aprod2_Kernel_BarConstr(vVect,sysmatConstr,knownTerms,mapNoss,nEqBarConstr,nEqExtConstr,nOfElextObs,nOfElBarObs,numOfBarStar,nAstroPSolved);
        if(nOfInstrConstr) aprod2_Kernel_InstrConstr(vVect, sysmatConstr, knownTerms, instrConstrIlung, instrCol,  VrIdAstroPDimMax, nDegFreedomAtt, mapNoss, nEqExtConstr, nEqBarConstr, nOfElextObs, nOfElBarObs, myid, nOfInstrConstr, nproc, nInstrPSolved, nAstroPSolved, nAttAxes);

        #ifdef USE_MPI
            //------------------------------------------------------------------------------------------------  TIME4
            starttime=get_time();
                MPI_Allreduce(MPI_IN_PLACE,&vVect[localAstroMax],static_cast<long>(nAttParam+nInstrParam+nGlobalParam),MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
                if(nAstroPSolved) SumCirc(vVect,comlsqr);
            endtime=get_time();
            communicationtime+=compute_time(endtime,starttime);
            //------------------------------------------------------------------------------------------------
        #endif

        nAstroElements=comlsqr.mapStar[myid][1]-comlsqr.mapStar[myid][0] + 1;
        if(myid<nproc-1)
        {
            nAstroElements=comlsqr.mapStar[myid][1]-comlsqr.mapStar[myid][0] +1;
            if(comlsqr.mapStar[myid][1]==comlsqr.mapStar[myid+1][0]) nAstroElements--;
        }

        max_vVect = maxCommMultiBlock_double(vVect, nunkSplit);
        ssq_vVect = sumCommMultiBlock_double(vVect, nAstroElements*nAstroPSolved, max_vVect);


        alphaLoc = max_vVect*std::sqrt(ssq_vVect);
        alphaLoc2=alphaLoc*alphaLoc;


        if(!myid){
            double alphaOther2 = alphaLoc2;
            ssq_vVect = sumCommMultiBlock_double(&vVect[localAstroMax], nunkSplit - localAstroMax, max_vVect);
            alphaLoc = max_vVect*std::sqrt(ssq_vVect);
            alphaLoc2 = alphaLoc*alphaLoc;
            alphaLoc2 = alphaOther2 + alphaLoc2;
        }

        #ifdef USE_MPI
            //------------------------------------------------------------------------------------------------  TIME6
            starttime=get_time();
                MPI_Allreduce(MPI_IN_PLACE,&alphaLoc2,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
            endtime=get_time();
            communicationtime+=compute_time(endtime,starttime);
            //------------------------------------------------------------------------------------------------
        #endif

        alpha=std::sqrt(alphaLoc2);
    }





    if(alpha > ZERO){
        dscal(vVect, 1/alpha, nunkSplit, ONE);
        cblas_dcopy_kernel(wVect,vVect,nunkSplit);
    }

    arnorm  = alpha * beta;

    if (arnorm == ZERO){
        *istop_out  = istop;
        *itn_out    = itn;
        *anorm_out  = anorm;
        *acond_out  = acond;
        *rnorm_out  = rnorm;
        *arnorm_out = test2;
        *xnorm_out  = xnorm;

        delete [] kAuxcopy;
        delete [] vAuxVect;

        return ;

    } 
    rhobar =   alpha;
    phibar =   beta;
    bnorm  =   beta;
    rnorm  =   beta;


    if(!myid){
        test1  = ONE;
        test2  = alpha / beta;
    }


    //  ==================================================================
    //  Main iteration loop.
    //  ==================================================================

    if (!myid) std::cout<<"LSQR: START ITERATIONS"<<std::endl; 
    ////////////////////////  START ITERATIONS
    #ifdef USE_MPI
        MPI_Bcast( &itnlim, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast( &comlsqr.itnLimit, 1, MPI_INT, 0, MPI_COMM_WORLD);
    #elif defined(KERNELTIME)
        event startAprod1Astro,   stopAprod1Astro;
        event startAprod1Att,     stopAprod1Att;
        event startAprod1Instr,   stopAprod1Instr;
        event startAprod2Astro,   stopAprod2Astro;
        event startAprod2Att,     stopAprod2Att;
        event startAprod2Instr,   stopAprod2Instr;
        float timekernel[6]={0.0,0.0,0.0,0.0,0.0,0.0};
        float milliseconds;
        eventCreate(&startAprod1Astro);
        eventCreate(&stopAprod1Astro);
        eventCreate(&startAprod1Att);
        eventCreate(&stopAprod1Att);
        eventCreate(&startAprod1Instr);
        eventCreate(&stopAprod1Instr);
        eventCreate(&startAprod2Astro);
        eventCreate(&stopAprod2Astro);
        eventCreate(&startAprod2Att);
        eventCreate(&stopAprod2Att);
        eventCreate(&startAprod2Instr);
        eventCreate(&stopAprod2Instr);
    #endif
    
    
    while (1) {
        startCycleTime=get_time();
        #ifdef USE_MPI
            MPI_Barrier(MPI_COMM_WORLD);
        #endif
        if (itn >= itnlim){
            istop = 5;
            break;
        }

        ++itn;
        comlsqr.itn=itn;

        //      ------------------------------------------------------------------
        //      Perform the next step of the bidiagonalization to obtain the
        //      next  beta, u, alpha, v.  These satisfy the relations
        //                 beta*u  =  A*v  -  alpha*u,
        //                alpha*v  =  A(transpose)*u  -  beta*v.
        //      ------------------------------------------------------------------


        dscal(knownTerms, alpha, nElemKnownTerms, MONE);
        kAuxcopy_Kernel(knownTerms, kAuxcopy, mapNoss, nEqExtConstr+nEqBarConstr+nOfInstrConstr);
        // //////////////////////////////////// APROD MODE 1
        #ifdef KERNELTIME

            milliseconds = 0.0;
            eventRecord(startAprod1Astro,0);
                if(nAstroPSolved) aprod1_Kernel_astro(knownTerms, sysmatAstro, vVect, matrixIndexAstro, offLocalAstro, mapNoss, nAstroPSolved);
            eventRecord(stopAprod1Astro,0);
            eventSynchronize(stopAprod1Astro);
            eventElapsedTime(&milliseconds, startAprod1Astro, stopAprod1Astro);
            timekernel[0]+=milliseconds;

            milliseconds = 0.0;
            eventRecord(startAprod1Att,0);
                if(nAttP) aprod1_Kernel_att_AttAxis(knownTerms, sysmatAtt, vVect, matrixIndexAtt, nAttP, mapNoss, nDegFreedomAtt, offLocalAtt, nAttParAxis);
            eventRecord(stopAprod1Att,0);
            eventSynchronize(stopAprod1Att);
            eventElapsedTime(&milliseconds, startAprod1Att, stopAprod1Att);
            timekernel[1]+=milliseconds;

            milliseconds = 0.0;
            eventRecord(startAprod1Instr,0);
                if(nInstrPSolved) aprod1_Kernel_instr(knownTerms, sysmatInstr, vVect, instrCol, mapNoss, offLocalInstr, nInstrPSolved);
            eventRecord(stopAprod1Instr,0);
            eventSynchronize(stopAprod1Instr);
            eventElapsedTime(&milliseconds, startAprod1Instr, stopAprod1Instr);
            timekernel[2]+=milliseconds;

            if(nGlobP) aprod1_Kernel_glob(knownTerms, sysmatGloB, vVect, offLocalGlob, mapNoss, nGlobP);
            // //////////////////////////////////// CONSTRAi_intS APROD MODE 1        
            if(nEqExtConstr) aprod1_Kernel_ExtConstr(knownTerms,sysmatConstr,vVect,VrIdAstroPDimMax,mapNoss,nDegFreedomAtt,startingAttColExtConstr,nEqExtConstr,nOfElextObs,numOfExtStar,numOfExtAttCol,nAstroPSolved,nAttAxes);
            if(nEqBarConstr) aprod1_Kernel_BarConstr(knownTerms, sysmatConstr, vVect, nOfElextObs, nOfElBarObs, nEqExtConstr, mapNoss, nEqBarConstr, numOfBarStar, nAstroPSolved );
            if(nOfInstrConstr) aprod1_Kernel_InstrConstr(knownTerms,sysmatConstr,vVect,instrConstrIlung,instrCol,VrIdAstroPDimMax,mapNoss,nDegFreedomAtt,nOfElextObs,nEqExtConstr,nOfElBarObs,nEqBarConstr,myid,nOfInstrConstr,nproc,nAstroPSolved,nAttAxes,nInstrPSolved);
        
        #else
            if(nAstroPSolved) aprod1_Kernel_astro(knownTerms, sysmatAstro, vVect, matrixIndexAstro, offLocalAstro, mapNoss, nAstroPSolved);
            if(nAttP) aprod1_Kernel_att_AttAxis(knownTerms, sysmatAtt, vVect, matrixIndexAtt, nAttP, mapNoss, nDegFreedomAtt, offLocalAtt, nAttParAxis);
            if(nInstrPSolved) aprod1_Kernel_instr(knownTerms, sysmatInstr, vVect, instrCol, mapNoss, offLocalInstr, nInstrPSolved);
            if(nGlobP) aprod1_Kernel_glob(knownTerms, sysmatGloB, vVect, offLocalGlob, mapNoss, nGlobP);
            // //////////////////////////////////// CONSTRAi_intS APROD MODE 1        
            if(nEqExtConstr) aprod1_Kernel_ExtConstr(knownTerms,sysmatConstr,vVect,VrIdAstroPDimMax,mapNoss,nDegFreedomAtt,startingAttColExtConstr,nEqExtConstr,nOfElextObs,numOfExtStar,numOfExtAttCol,nAstroPSolved,nAttAxes);
            if(nEqBarConstr) aprod1_Kernel_BarConstr(knownTerms, sysmatConstr, vVect, nOfElextObs, nOfElBarObs, nEqExtConstr, mapNoss, nEqBarConstr, numOfBarStar, nAstroPSolved );
            if(nOfInstrConstr) aprod1_Kernel_InstrConstr(knownTerms,sysmatConstr,vVect,instrConstrIlung,instrCol,VrIdAstroPDimMax,mapNoss,nDegFreedomAtt,nOfElextObs,nEqExtConstr,nOfElBarObs,nEqBarConstr,myid,nOfInstrConstr,nproc,nAstroPSolved,nAttAxes,nInstrPSolved);
        #endif


        #ifdef USE_MPI
            std::copy_n(POL,&knownTerms[mapNoss],nEqExtConstr+nEqBarConstr+nOfInstrConstr,knownTerms_loc);
            //------------------------------------------------------------------------------------------------
            starttime=get_time();
                MPI_Allreduce(MPI_IN_PLACE,knownTerms_loc,nEqExtConstr+nEqBarConstr+nOfInstrConstr,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
            endtime=get_time();
            communicationtime+=compute_time(endtime,starttime);
            //------------------------------------------------------------------------------------------------
            std::copy_n(POL,knownTerms_loc,nEqExtConstr+nEqBarConstr+nOfInstrConstr,&knownTerms[mapNoss]);

        #endif

            std::transform(POL,
                            kAuxcopy,
                            kAuxcopy+nEqExtConstr+nEqBarConstr+nOfInstrConstr,
                            knownTerms+mapNoss,
                            knownTerms+mapNoss,
                            std::plus<>());


        max_knownTerms = maxCommMultiBlock_double(knownTerms,nElemKnownTerms);


        if(!myid){
            ssq_knownTerms = sumCommMultiBlock_double(knownTerms,nElemKnownTerms,max_knownTerms);
            betaLoc = max_knownTerms*std::sqrt(ssq_knownTerms);
            betaLoc2 = betaLoc*betaLoc;
        }else{
            ssq_knownTerms = sumCommMultiBlock_double(knownTerms, mapNoss, max_knownTerms);
            betaLoc = max_knownTerms*std::sqrt(ssq_knownTerms);
            betaLoc2 = betaLoc*betaLoc;
        }

        #ifdef USE_MPI
            //------------------------------------------------------------------------------------------------
            starttime=get_time();
                MPI_Allreduce(MPI_IN_PLACE,&betaLoc2,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
            endtime=get_time();
            communicationtime+=compute_time(endtime,starttime);
            //------------------------------------------------------------------------------------------------
        #endif

        beta=std::sqrt(betaLoc2);


        temp   =   d2norm( alpha, beta );
        temp   =   d2norm( temp , damp );
        anorm  =   d2norm( anorm, temp );

        if(beta>ZERO){

            dscal(knownTerms,1/beta, nElemKnownTerms,ONE);
            dscal(vVect,beta,nunkSplit,MONE);
            vAuxVect_Kernel(vVect, vAuxVect, localAstroMax);


            if (myid) {
                vVect_Put_To_Zero_Kernel(vVect,localAstroMax,nunkSplit);
            }


            //////////////////////////////////// APROD MODE 2
        #ifdef KERNELTIME
            milliseconds = 0.0;
            eventRecord(startAprod2Astro,0);
                if(nAstroPSolved) aprod2_Kernel_astro(vVect, sysmatAstro, knownTerms, matrixIndexAstro, startend, offLocalAstro, nnz, nAstroPSolved);
            eventRecord(stopAprod2Astro,0);
            eventSynchronize(stopAprod2Astro);
            eventElapsedTime(&milliseconds, startAprod2Astro, stopAprod2Astro);
            timekernel[3]+=milliseconds;

            milliseconds = 0.0;
            eventRecord(startAprod2Att,0);
                if(nAttP) aprod2_Kernel_att_AttAxis(vVect,sysmatAtt,knownTerms,matrixIndexAtt,nAttP,nDegFreedomAtt,offLocalAtt,mapNoss,nAstroPSolved,nAttParAxis);
            eventRecord(stopAprod2Att,0);
            eventSynchronize(stopAprod2Att);
            eventElapsedTime(&milliseconds, startAprod2Att, stopAprod2Att);
            timekernel[4]+=milliseconds;

            milliseconds = 0.0;
            eventRecord(startAprod2Instr,0);
                if(nInstrPSolved) aprod2_Kernel_instr(vVect,sysmatInstr, knownTerms, instrCol, offLocalInstr, mapNoss, nInstrPSolved);
            eventRecord(stopAprod2Instr,0);
            eventSynchronize(stopAprod2Instr);
            eventElapsedTime(&milliseconds, startAprod2Instr, stopAprod2Instr);
            timekernel[5]+=milliseconds;

            // //////////////////////////////////// CONSTRAi_intS APROD MODE 1        
            if(nEqExtConstr) aprod2_Kernel_ExtConstr(vVect,sysmatConstr,knownTerms,mapNoss,nDegFreedomAtt,VrIdAstroPDimMax,nEqExtConstr,nOfElextObs,numOfExtStar,startingAttColExtConstr,numOfExtAttCol,nAstroPSolved,nAttAxes);
            if(nEqBarConstr) aprod2_Kernel_BarConstr(vVect,sysmatConstr,knownTerms,mapNoss,nEqBarConstr,nEqExtConstr,nOfElextObs,nOfElBarObs,numOfBarStar,nAstroPSolved);
            if(nOfInstrConstr) aprod2_Kernel_InstrConstr(vVect, sysmatConstr, knownTerms, instrConstrIlung, instrCol,  VrIdAstroPDimMax, nDegFreedomAtt, mapNoss, nEqExtConstr, nEqBarConstr, nOfElextObs, nOfElBarObs, myid, nOfInstrConstr, nproc, nInstrPSolved, nAstroPSolved, nAttAxes);

        #else
            if(nAstroPSolved) aprod2_Kernel_astro(vVect, sysmatAstro, knownTerms, matrixIndexAstro, startend, offLocalAstro, nnz, nAstroPSolved);
            if(nAttP) aprod2_Kernel_att_AttAxis(vVect,sysmatAtt,knownTerms,matrixIndexAtt,nAttP,nDegFreedomAtt,offLocalAtt,mapNoss,nAstroPSolved,nAttParAxis);
            if(nInstrPSolved) aprod2_Kernel_instr(vVect,sysmatInstr, knownTerms, instrCol, offLocalInstr, mapNoss, nInstrPSolved);
            //APROD2 CONSTRAINTS
            if(nEqExtConstr) aprod2_Kernel_ExtConstr(vVect,sysmatConstr,knownTerms,mapNoss,nDegFreedomAtt,VrIdAstroPDimMax,nEqExtConstr,nOfElextObs,numOfExtStar,startingAttColExtConstr,numOfExtAttCol,nAstroPSolved,nAttAxes);
            if(nEqBarConstr) aprod2_Kernel_BarConstr(vVect,sysmatConstr,knownTerms,mapNoss,nEqBarConstr,nEqExtConstr,nOfElextObs,nOfElBarObs,numOfBarStar,nAstroPSolved);
            if(nOfInstrConstr) aprod2_Kernel_InstrConstr(vVect, sysmatConstr, knownTerms, instrConstrIlung, instrCol,  VrIdAstroPDimMax, nDegFreedomAtt, mapNoss, nEqExtConstr, nEqBarConstr, nOfElextObs, nOfElBarObs, myid, nOfInstrConstr, nproc, nInstrPSolved, nAstroPSolved, nAttAxes);
        #endif


        #ifdef USE_MPI
            //------------------------------------------------------------------------------------------------
            starttime=get_time();
                MPI_Allreduce(MPI_IN_PLACE,&vVect[localAstroMax],static_cast<long>(nAttParam+nInstrParam+nGlobalParam),MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);  
            endtime=get_time();
            communicationtime+=compute_time(endtime,starttime);
            //------------------------------------------------------------------------------------------------
            //------------------------------------------------------------------------------------------------
            if(nAstroPSolved) SumCirc(vVect,comlsqr);
            //------------------------------------------------------------------------------------------------
        #endif

            std::transform(POL, 
                                vAuxVect, 
                                vAuxVect+localAstroMax,
                                vVect,
                                vVect, std::plus<>());

            max_vVect = maxCommMultiBlock_double(vVect, nunkSplit);
            ssq_vVect = sumCommMultiBlock_double(vVect, nAstroElements*nAstroPSolved, max_vVect);

            double alphaLoc{ZERO};
            alphaLoc = max_vVect*std::sqrt(ssq_vVect);
            alphaLoc2=alphaLoc*alphaLoc;



            if(!myid){
                double alphaOther2 = alphaLoc2;
                ssq_vVect = sumCommMultiBlock_double(&vVect[localAstroMax], nunkSplit - localAstroMax, max_vVect);
                alphaLoc = max_vVect*std::sqrt(ssq_vVect);
                alphaLoc2 = alphaLoc*alphaLoc;
                alphaLoc2 = alphaOther2 + alphaLoc2;
            }

        #ifdef USE_MPI
            //------------------------------------------------------------------------------------------------
            starttime=get_time();
                MPI_Allreduce(MPI_IN_PLACE,&alphaLoc2,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
            endtime=get_time();
            communicationtime+=compute_time(endtime,starttime);
            //------------------------------------------------------------------------------------------------
        #endif

            alpha=std::sqrt(alphaLoc2);
                    
            if (alpha > ZERO) {
                dscal(vVect,1/alpha,nunkSplit,1);
            }

        }



        //------------------------------------------------------------------
        //      Use a plane rotation to eliminate the damping parameter.
        //      This alters the diagonal (rhobar) of the lower-bidiagonal matrix.
        //------------------------------------------------------------------
        rhbar1 = rhobar;
        if ( damped ) {
            rhbar1 = d2norm( rhobar, damp );
            cs1    = rhobar / rhbar1;
            sn1    = damp   / rhbar1;
            psi    = sn1 * phibar;
            phibar = cs1 * phibar;
        }

        //------------------------------------------------------------------
        //      Use a plane rotation to eliminate the subdiagonal element (beta)
        //      of the lower-bidiagonal matrix, giving an upper-bidiagonal matrix.
        //------------------------------------------------------------------
        rho    =   d2norm( rhbar1, beta );
        cs     =   rhbar1 / rho;
        sn     =   beta   / rho;
        theta  =   sn * alpha;
        rhobar = - cs * alpha;
        phi    =   cs * phibar;
        phibar =   sn * phibar;
        tau    =   sn * phi;


        //------------------------------------------------------------------
        //      Update  x, w  and (perhaps) the standard error estimates.
        //------------------------------------------------------------------
        t1     =   phi   / rho;
        t2     = - theta / rho;
        t3     =   ONE   / rho;
        dknorm =   ZERO;


        dknorm=std::transform_reduce(POL,wVect,wVect+nAstroElements*nAstroPSolved,ZERO,std::plus<>(),[=](const auto& x){return (t3*x)*(t3*x);});
        
        #ifdef USE_MPI
            //------------------------------------------------------------------------------------------------
            starttime=get_time();
                MPI_Allreduce(MPI_IN_PLACE,&dknorm,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
            endtime=get_time();
            communicationtime+=compute_time(endtime,starttime);
            //------------------------------------------------------------------------------------------------ 		
        #endif

        std::transform(POL, wVect, wVect + localAstro, xSolution, xSolution,
                 [=](auto xi, auto yi){ return t1 * xi + yi; });
        if(wantse){
            std::transform(POL, wVect, wVect + localAstro, standardError, standardError,
                    [=](auto xi, auto yi){ return yi +(t3*xi)*(t3*xi); });
        }
        std::transform(POL, vVect, vVect + localAstro, wVect, wVect,
                    [=](auto xi, auto yi){ return xi + t2*yi; });


        std::transform(POL,wVect+localAstroMax, wVect + localAstroMax+other, xSolution+localAstroMax, xSolution+localAstroMax,
                 [=](auto xi, auto yi){ return t1 * xi + yi; });
        if(wantse){
            std::transform(POL,wVect+localAstroMax, wVect + localAstroMax+other, standardError+localAstroMax, standardError+localAstroMax,
                    [=](auto xi, auto yi){ return yi +(t3*xi)*(t3*xi); });
        }
        std::transform(POL, vVect+localAstroMax, vVect + localAstroMax+other, wVect+localAstroMax, wVect+localAstroMax,
                    [=](auto xi, auto yi){ return xi + t2*yi; });

        dknorm=std::transform_reduce(POL,wVect+localAstroMax,wVect+localAstroMax+other,dknorm,std::plus<>(),[=](const auto& x){return (t3*x)*(t3*x);});


        //------------------------------------------------------------------
        //      Monitor the norm of d_k, the update to x.
        //      dknorm = norm( d_k )
        //      dnorm  = norm( D_k ),        where   D_k = (d_1, d_2, ..., d_k )
        //      dxk    = norm( phi_k d_k ),  where new x = x_k + phi_k d_k.
        //------------------------------------------------------------------
        dknorm = std::sqrt( dknorm );
        dnorm  = d2norm( dnorm, dknorm );
        dxk    = std::fabs( phi * dknorm );
        if (dxmax < dxk ) {
            dxmax   =  dxk;
            #ifdef VERBOSE
                maxdx   =  itn;
            #endif
        }


        //      ------------------------------------------------------------------
        //      Use a plane rotation on the right to eliminate the
        //      super-diagonal element (theta) of the upper-bidiagonal matrix.
        //      Then use the result to estimate  norm(x).
        //      ------------------------------------------------------------------
        delta  =   sn2 * rho;
        gambar = - cs2 * rho;
        rhs    =   phi    - delta * z;
        zbar   =   rhs    / gambar;
        xnorm  =   d2norm( xnorm1, zbar  );
        gamma  =   d2norm( gambar, theta );
        cs2    =   gambar / gamma;
        sn2    =   theta  / gamma;
        z      =   rhs    / gamma;
        xnorm1 =   d2norm( xnorm1, z     );


        //      ------------------------------------------------------------------
        //      Test for convergence.
        //      First, estimate the norm and condition of the matrix  Abar,
        //      and the norms of  rbar  and  Abar(transpose)*rbar.
        //      ------------------------------------------------------------------
        acond  =   anorm * dnorm;
        res2   =   d2norm( res2 , psi    );
        rnorm  =   d2norm( res2 , phibar );
        arnorm =   alpha * std::fabs( tau );

        //      Now use these norms to estimate certain other quantities,
        //      some of which will be small near a solution.

        #ifdef VERBOSE
            alfopt =   std::sqrt( rnorm / (dnorm * xnorm) );
        #endif

        test1  =   rnorm /  bnorm;
        test2  =   ZERO;
        if (rnorm   > ZERO) test2 = arnorm / (anorm * rnorm);
        test3  =   ONE   /  acond;
        t1     =   test1 / (ONE  +  anorm * xnorm / bnorm);
        rtol   =   btol  +  atol *  anorm * xnorm / bnorm;

        //      The following tests guard against extremely small values of
        //      atol, btol  or  ctol.  (The user may have set any or all of
        //      the parameters  atol, btol, conlim  to zero.)
        //      The effect is equivalent to the normal tests using
        //      atol = relpr,  btol = relpr,  conlim = 1/relpr.

        t3     =   ONE + test3;
        t2     =   ONE + test2;
        t1     =   ONE + t1;

        if (itn >= itnlim) istop = 5;
        if (t3  <= ONE   ) istop = 4;
        if (t2  <= ONE   ) istop = 2;
        if (t1  <= ONE   ) istop = 1;

        //      Allow for tolerances set by the user.

        if (test3 <= ctol) istop = 4;
        if (test2 <= atol) istop = 2;
        if (test1 <= rtol) istop = 1;   //(Michael Friedlander had this commented out)

        //      ------------------------------------------------------------------
        //      See if it is time to pri_int something.
        //      ------------------------------------------------------------------

        //------------------------------------------------------------------------------------------------
        #ifdef USE_MPI
            endCycleTime=get_time(startCycleTime);
            totTimeIteration+=endCycleTime;
            if(myid==0) printf("lsqr: Iteration number %d. Iteration seconds %lf. Global Seconds %lf \n",itn,endCycleTime,totTimeIteration);
        #else
            endCycleTime=get_time();
            std::chrono::duration<double>  elapsedTime = endCycleTime - startCycleTime;
            totTimeIteration += elapsedTime;  
            printf("lsqr: Iteration number %d. Iteration seconds %lf. Global Seconds %lf \n", itn, elapsedTime.count(), totTimeIteration.count());
        #endif
        //------------------------------------------------------------------------------------------------


            
        if (istop) break;

    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

    }
    //  ==================================================================
    //  End of iteration loop.
    //  ==================================================================
    //  Finish off the standard error estimates.
    #ifdef USE_MPI
        MPI_Allreduce(MPI_IN_PLACE, &communicationtime, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        if(!myid) printf("Max Communication time: %lf \n",communicationtime);
        double maxavtime=totTimeIteration/itn;
        if(!myid) printf("Average iteration time: %lf \n", totTimeIteration/itn);
        MPI_Allreduce(MPI_IN_PLACE, &maxavtime, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        if(!myid) printf("Max Average iteration time: %lf \n",maxavtime);
    #elif defined(KERNELTIME)
        printf("Average iteration time: %lf \n", totTimeIteration.count()/itn);
        printf("Average kernel Aprod1Astro time: %lf \n", 1e-3*timekernel[0]/itn);
        printf("Average kernel Aprod1Att time: %lf \n", 1e-3*timekernel[1]/itn);
        printf("Average kernel Aprod1Instr time: %lf \n", 1e-3*timekernel[2]/itn);
        printf("Average kernel Aprod2Astro time: %lf \n", 1e-3*timekernel[3]/itn);
        printf("Average kernel Aprod2Att time: %lf \n", 1e-3*timekernel[4]/itn);
        printf("Average kernel Aprod2Instr time: %lf \n", 1e-3*timekernel[5]/itn);
    #else
        printf("Average iteration time: %lf \n", totTimeIteration.count()/itn);
    #endif
    //------------------------------------------------------------------------------------------------ 		


    if ( wantse ) {
        t    =   ONE;
        if (m > n)     t = m - n;
        if ( damped )  t = m;
        t    =   rnorm / sqrt( t );
      
        for (long i = 0; i < nunkSplit; i++)
            standardError[i]  = t * std::sqrt( standardError[i] );
        
    }



    *istop_out  = istop;
    *itn_out    = itn;
    *anorm_out  = anorm;
    *acond_out  = acond;
    *rnorm_out  = rnorm;
    *arnorm_out = test2;
    *xnorm_out  = xnorm;




    delete [] kAuxcopy;
    delete [] vAuxVect;
    
    #ifdef USE_MPI
        delete [] knownTerms_loc;
    #endif

    return;

}





