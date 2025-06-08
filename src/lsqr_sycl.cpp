/* 

SYCL Version

*/

#include <sycl/sycl.hpp>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stdbool.h>
#include <math.h>
#include <time.h>
#include "util.h"
#include <limits.h>
#include <sys/time.h>
#include "lsqr.h"


#ifdef USE_MPI
    #include <mpi.h>
#endif

#ifdef PROF // put also AMD
    #include <nvToolsExt.h>

    class Astro1Kernel; 
    class Att1Kernel; 
    class Instr1kernel; 
    class Astro2Kernel; 
    class Att2Kernel;
    class Instr2kernel; 


#endif

#define ZERO   0.0
#define ONE    1.0
#define MONE    -1.0

static const int blockSize = 128;
static const int gridSize = 1024;

static const int THREADS_PER_WRAPS=32;


//AMD 

#if defined(__MI100X__)
    #define ThreadsXBlock 128 
    #define	BlockXGrid 1024 	
    #define ThreadsXBlockAprod2Astro  256 	
    #define ThreadsXBlockAprod1Astro 128
    #define ThreadsXBlockAprod2AttInstr 16 
    #define ThreadsXBlockAprod1AttInstr 16
    #define TILE_WIDTH 32
    #define RowsPerBlock 1
    #define RowsPerBlock 1
#elif defined(__MI250X__)
    #define ThreadsXBlock 128 
    #define	BlockXGrid 1024 	
    #define ThreadsXBlockAprod2Astro  256 	
    #define ThreadsXBlockAprod1Astro 128
    #define ThreadsXBlockAprod2AttInstr 16 
    #define ThreadsXBlockAprod1AttInstr 16
    #define TILE_WIDTH 32
    #define RowsPerBlock 1
    #define RowsPerBlock 1
#elif defined(__NVIDIA90__)
    #define ThreadsXBlock 64
    #define BlockXGrid 1024
    #define ThreadsXBlockAprod2Astro 256
    #define ThreadsXBlockAprod1Astro 128
    #define ThreadsXBlockAprod2AttInstr 16
    #define ThreadsXBlockAprod1AttInstr 16
    #define RowsPerBlock 1
    #define TILE_WIDTH 32 
#elif defined (__NVIDIA80__)
    #define ThreadsXBlock 	64
    #define BlockXGrid 	1024
    #define ThreadsXBlockAprod2Astro 256  
    #define ThreadsXBlockAprod1Astro 128
    #define ThreadsXBlockAprod2AttInstr 16
    #define ThreadsXBlockAprod1AttInstr 16 
    #define RowsPerBlock 1
    #define TILE_WIDTH 32
#elif defined(__NVIDIA70__)
    #define ThreadsXBlock 	32
    #define BlockXGrid 	1024 //128
    #define ThreadsXBlockAprod2Astro 256
    #define ThreadsXBlockAprod1Astro 128
    #define ThreadsXBlockAprod2AttInstr 16
    #define ThreadsXBlockAprod1AttInstr 16 
    #define RowsPerBlock 1
    #define TILE_WIDTH  32
#else
    #error "Unknown platform"
#endif


// #define atomicAdd(x, y) (sycl::atomic_ref<double, sycl::memory_order::relaxed, sycl::memory_scope::work_group, sycl::access::address_space::generic_space>(*(x)) +=(y))

#define atomicAdd(x, y) (sycl::atomic_ref<double, sycl::memory_order::relaxed, sycl::memory_scope::device, sycl::access::address_space::global_space>(*(x)) +=(y))


__inline__ double warpReduceSum(double val, const sycl::nd_item<3> &item_ct1) {
    for (int i = item_ct1.get_sub_group().get_local_range().get(0) >> 1; i > 0;
         i >>= 1) {
        val += sycl::shift_group_left(item_ct1.get_sub_group(), val, i);
    }
    return val;
}

template<long THREAD_PER_BLOCK>
void dknorm_compute(double* __restrict__ dknorm_vec,
                                const double* __restrict__ wVect_dev,
                                const long begin, 
                                const long end,
                                const double t3,
                                const sycl::nd_item<3> &item_ct1,
                                double *val)
{
    long tid = item_ct1.get_group(2)*item_ct1.get_local_range(2)+item_ct1.get_local_id(2) + begin;
    long gridSize = item_ct1.get_local_range(2)*item_ct1.get_group_range(2);
    double sum=0.0;
    *dknorm_vec=0.0;
    for (auto i = tid; i < end; i += gridSize){
        sum += wVect_dev[i]*wVect_dev[i]*t3*t3;
    }

    int lane = item_ct1.get_local_id(2)%item_ct1.get_sub_group().get_local_range().get(0);
    int wid = item_ct1.get_local_id(2)/item_ct1.get_sub_group().get_local_range().get(0);

    sum = warpReduceSum(sum, item_ct1);
    if (lane==0) val[wid]=sum; // Write reduced value to shared memory
    item_ct1.barrier(sycl::access::fence_space::local_space);
    sum = (item_ct1.get_local_id(2)< item_ct1.get_local_range(2)/item_ct1.get_sub_group().get_local_range().get(0)) ? val[lane] : 0;
    if (wid == 0) sum = warpReduceSum(sum, item_ct1);

    if (item_ct1.get_local_id(2) == 0)
        atomicAdd(dknorm_vec, sum);
}




template<long THREAD_PER_BLOCK, typename T>
void maxCommMultiBlock_double(double *gArr, double *gOut, const T arraySize, const sycl::nd_item<3> &item_ct1, double *val) {
    
    const T gthIdx =  item_ct1.get_local_id(2) + item_ct1.get_group(2) * blockSize;
    const T gridSize = blockSize * item_ct1.get_group_range(2);

    double maxVal = 0.0;
    for (T i = gthIdx; i < arraySize; i += gridSize) {
        maxVal = fmax(maxVal, fabs(gArr[i]));
    }

    int lane = item_ct1.get_local_id(2)%item_ct1.get_sub_group().get_local_range().get(0);
    int wid = item_ct1.get_local_id(2)/item_ct1.get_sub_group().get_local_range().get(0);
    for (int i = item_ct1.get_sub_group().get_local_range().get(0) >> 1; i > 0;i >>= 1) {
        maxVal = sycl::fmax(maxVal, sycl::shift_group_left(item_ct1.get_sub_group(), maxVal, i));
    }
    if (lane==0) val[wid]=maxVal;
    item_ct1.barrier(sycl::access::fence_space::local_space);
    
    maxVal = (item_ct1.get_local_id(2)<item_ct1.get_local_range(2)/item_ct1.get_sub_group().get_local_range().get(0)) ? val[lane] : 0;

    if(wid==0){  //Final reduce within first warp
        for (int i = item_ct1.get_sub_group().get_local_range().get(0) >> 1;i > 0; i >>= 1) {
            maxVal=sycl::fmax(maxVal, sycl::shift_group_left(item_ct1.get_sub_group(), val[wid], i));
        }
    }

    if (item_ct1.get_local_id(2) == 0)
        gOut[item_ct1.get_group(2)] = maxVal;
 }



template<long THREAD_PER_BLOCK, typename T>
void sumCommMultiBlock_double(double *gArr, double *gOut, const double max, const T arraySize, const sycl::nd_item<3> &item_ct1, double *val) {
    const T gthIdx = item_ct1.get_local_id(2)+item_ct1.get_group(2) * item_ct1.get_local_range(2);
    const T gridSize =item_ct1.get_local_range(2) * item_ct1.get_group_range(2);

    double sum = 0.0;
    double divmax = 1.0 / max;
    for (T i = gthIdx; i < arraySize; i += gridSize) {
        sum += (gArr[i] * divmax) * (gArr[i] * divmax);
    }

    int lane = item_ct1.get_local_id(2)%item_ct1.get_sub_group().get_local_range().get(0);
    int wid = item_ct1.get_local_id(2)/item_ct1.get_sub_group().get_local_range().get(0);

    sum = warpReduceSum(sum, item_ct1);
    if (lane==0) val[wid]=sum; // Write reduced value to shared memory
    item_ct1.barrier(sycl::access::fence_space::local_space);

    sum = (item_ct1.get_local_id(2)<item_ct1.get_local_range(2)/item_ct1.get_sub_group().get_local_range().get(0)) ? val[lane] : 0;
    if (wid == 0) sum = warpReduceSum(sum, item_ct1);

    if (item_ct1.get_local_id(2) == 0)
        gOut[item_ct1.get_group(2)] = sum;

}




template<long THREAD_PER_BLOCK, typename T>
void realsumCommMultiBlock_double(double *gArr, double *gOut, const T arraySize,const sycl::nd_item<3> &item_ct1, double *val) {

    const T gthIdx = item_ct1.get_group(2)*item_ct1.get_local_range(2)+item_ct1.get_local_id(2);
    const T gridSize = item_ct1.get_local_range(2) * item_ct1.get_group_range(2);
    double sum = 0.0;

    for (T i = gthIdx; i < arraySize; i += gridSize)
        sum += gArr[i];

    int lane = item_ct1.get_local_id(2) % item_ct1.get_sub_group().get_local_range().get(0);
    int wid = item_ct1.get_local_id(2) / item_ct1.get_sub_group().get_local_range().get(0);
    sum = warpReduceSum(sum, item_ct1);
    if (lane==0) val[wid]=sum; // Write reduced value to shared memory
    item_ct1.barrier(sycl::access::fence_space::local_space);


    sum = (item_ct1.get_local_id(2) <item_ct1.get_local_range(2) /item_ct1.get_sub_group().get_local_range().get(0)) ? val[lane] : 0;
    if (wid == 0) sum = warpReduceSum(sum, item_ct1);

    if (item_ct1.get_local_id(2) == 0)
        gOut[item_ct1.get_group(2)] = sum;
}


template<typename L>
void dscal (double* knownTerms_dev, const double val, const L& N, const int sign,const sycl::nd_item<3> &item_ct1)
{
    L ix = item_ct1.get_group(2)*item_ct1.get_local_range(2)+item_ct1.get_local_id(2);

    while(ix < N){
        knownTerms_dev[ix]=sign*(knownTerms_dev[ix]*val);
        ix += item_ct1.get_group_range(2) * item_ct1.get_local_range(2);
    }
}

template<typename L, typename I>
void  kAuxcopy_Kernel(double* knownTerms_dev, double* kAuxcopy_dev, const L& nobs, const I& N, const sycl::nd_item<3> &item_ct1)
{
    I ix = item_ct1.get_group(2)*item_ct1.get_local_range(2)+item_ct1.get_local_id(2);

    if (ix < N) {
        kAuxcopy_dev[ix] = knownTerms_dev[nobs + ix];
        knownTerms_dev[nobs + ix] = 0.0;
    }
}

template<typename L>
void vAuxVect_Kernel (double*vVect_dev, double* vAuxVect_dev, const L& N, const sycl::nd_item<3> &item_ct1)
{
    L ix = item_ct1.get_group(2)*item_ct1.get_local_range(2)+item_ct1.get_local_id(2);

    if (ix < N) {
        vAuxVect_dev[ix] = vVect_dev[ix];
        vVect_dev[ix] = ZERO;
    }
}

template<typename L>
void vVect_Put_To_Zero_Kernel (double*vVect_dev, const L& localAstroMax, const L& nunkSplit,const sycl::nd_item<3> &item_ct1)
{
    L ix = item_ct1.get_group(2)*item_ct1.get_local_range(2)+item_ct1.get_local_id(2);

    if (ix >= localAstroMax && ix < nunkSplit) {
        vVect_dev[ix] = ZERO;
    }
}

template<typename L>
void kauxsum (double* knownTerms_dev,double* kAuxcopy_dev, const L& n,const sycl::nd_item<3> &item_ct1)
{
    L ix = item_ct1.get_group(2)*item_ct1.get_local_range(2)+item_ct1.get_local_id(2);

    if (ix < n) {
        knownTerms_dev[ix] = knownTerms_dev[ix]+kAuxcopy_dev[ix];
    }
}

template<typename L>
void vaux_sum(double* vV, double* vA, const L& lAM,const sycl::nd_item<3> &item_ct1)
{
    L ix = item_ct1.get_group(2)*item_ct1.get_local_range(2)+item_ct1.get_local_id(2);

    if(ix<lAM){
        vV[ix]+=vA[ix];
    }
} 

void transform1(double* xSolution, const double* wVect, const long begin, const long end, const double t1,
                const sycl::nd_item<3> &item_ct1){

    long ix = item_ct1.get_group(2)*item_ct1.get_local_range(2)+item_ct1.get_local_id(2) + begin;
    while(ix < end){
        xSolution[ix]   =  xSolution[ix] + t1*wVect[ix];
        ix += item_ct1.get_group_range(2) * item_ct1.get_local_range(2);
    }
}

void transform2(double* standardError, const double* wVect, const long begin, const long end, const double t3,
                const sycl::nd_item<3> &item_ct1){
    long ix = item_ct1.get_group(2)*item_ct1.get_local_range(2)+item_ct1.get_local_id(2) + begin;

    while(ix < end){
        standardError[ix]  =  standardError[ix] +(t3*wVect[ix])*(t3*wVect[ix]);
        ix += item_ct1.get_group_range(2) * item_ct1.get_local_range(2);
    }
}

void transform3(double* wVect, const double* vVect, const long begin, const long end, const double t2,
                const sycl::nd_item<3> &item_ct1){
    long ix = item_ct1.get_group(2)*item_ct1.get_local_range(2)+item_ct1.get_local_id(2) + begin;

    while(ix < end){
        wVect[ix]   =  vVect[ix]+t2*wVect[ix];
        ix += item_ct1.get_group_range(2) * item_ct1.get_local_range(2);
    }
}


void cblas_dcopy_kernel (const long nunkSplit, double* vVect_dev, double* wVect_dev,
                         const sycl::nd_item<3> &item_ct1)
{
    long ix = item_ct1.get_group(2)*item_ct1.get_local_range(2)+item_ct1.get_local_id(2);

    if(ix < nunkSplit)
    {
        wVect_dev[ix] = vVect_dev[ix];
    }
}


static inline double
d2norm( const double a, const double b )
{
    double scale;
    const double zero = ZERO;

    scale  = fabs( a ) + fabs( b );
    if (scale == zero)
        return zero;
    else
        return scale * sqrt( (a/scale)*(a/scale) + (b/scale)*(b/scale) );
}

static inline void
dload( const long n, const double alpha, double x[] )
{    
    #pragma omp for
    for (int i = 0; i < n; i++) x[i] = alpha;
    return;
}


//-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
//-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
//-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
//-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
//-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
// Astrometric part
void aprod1_Kernel_astro(double*  __restrict__ knownTerms_dev, 
                        const double*  __restrict__ systemMatrix_dev, 
                        const double*  __restrict__ vVect_dev, 
                        const long*  __restrict__ matrixIndexAstro_dev, 
                        const long nobs, 
                        const long offLocalAstro, 
                        const short  nAstroPSolved,
                         const sycl::nd_item<3> &item_ct1)
{
    long ix=item_ct1.get_group(2)*item_ct1.get_local_range(2)+item_ct1.get_local_id(2);

    while (ix < nobs) {
        double sum = 0.0;
        const long jstartAstro = matrixIndexAstro_dev[ix] - offLocalAstro;
        
        for(short jx = 0; jx <   nAstroPSolved; jx++) {
            double matrixVal = systemMatrix_dev[ix * nAstroPSolved + jx];  
            double vectVal = vVect_dev[jstartAstro + jx];                  
            sum += matrixVal * vectVal;  
        }

        knownTerms_dev[ix] += sum;
        ix += item_ct1.get_group_range(2)*item_ct1.get_local_range(2);
    }
}

// Attitude part
void aprod1_Kernel_att_AttAxis(double*  __restrict__ knownTerms_dev, 
                                const double*  __restrict__ systemMatrix_dev, 
                                const double*  __restrict__ vVect_dev, 
                                const long*  __restrict__ matrixIndexAtt_dev, 
                                const long nAttP, 
                                const long nobs, 
                                const long nDegFreedomAtt, 
                                const long offLocalAtt, 
                                const short nAttParAxis, 
                               const sycl::nd_item<3> &item_ct1)
{

    long ix=item_ct1.get_group(2)*item_ct1.get_local_range(2)+item_ct1.get_local_id(2);

    while (ix < nobs) {
        double sum = 0.0;
        long jstartAtt_0 = matrixIndexAtt_dev[ix] + offLocalAtt; 
        
        
        for(short inpax = 0;inpax<nAttParAxis;++inpax)
            sum += systemMatrix_dev[ix * nAttP + inpax] * vVect_dev[jstartAtt_0 + inpax];
        jstartAtt_0 += nDegFreedomAtt;
        
        
        for(short inpax = 0;inpax<nAttParAxis;++inpax)
            sum += systemMatrix_dev[ix * nAttP + nAttParAxis + inpax ] * vVect_dev[jstartAtt_0 + inpax];
        jstartAtt_0 += nDegFreedomAtt;
        
        
        for(short inpax = 0;inpax<nAttParAxis;++inpax)
            sum += systemMatrix_dev[ix * nAttP + nAttParAxis + nAttParAxis + inpax] * vVect_dev[jstartAtt_0 + inpax];

        knownTerms_dev[ix] += sum;

        ix += item_ct1.get_group_range(2) * item_ct1.get_local_range(2);
  }
}

// Instrumental part
void aprod1_Kernel_instr(double*  __restrict__ knownTerms_dev, 
                        const double*  __restrict__ systemMatrix_dev, 
                        const double*  __restrict__ vVect_dev, 
                        const int*  __restrict__ instrCol_dev, 
                        const long nobs, 
                        const long offLocalInstr, 
                        const short nInstrPSolved,
                        const sycl::nd_item<3> &item_ct1){

    long ix = item_ct1.get_group(2)*item_ct1.get_local_range(2)+item_ct1.get_local_id(2);

    while (ix < nobs) {
        double sum = 0.0;
        const long iiVal=ix*nInstrPSolved;
        long ixInstr = 0;
        
        for(short inInstr=0;inInstr<nInstrPSolved;inInstr++){
            ixInstr=offLocalInstr+instrCol_dev[iiVal+inInstr];
            sum += systemMatrix_dev[ix * nInstrPSolved + inInstr]*vVect_dev[ixInstr];
        }
        knownTerms_dev[ix] += sum;
        ix += item_ct1.get_group_range(2) * item_ct1.get_local_range(2);
    }
}


// Global part
void aprod1_Kernel_glob(double*  __restrict__ knownTerms_dev, 
                        const double*  __restrict__ systemMatrix_dev, 
                        const double*  __restrict__ vVect_dev, 
                        const long offLocalGlob, 
                        const long nobs, 
                        const short nGlobP,
                        const sycl::nd_item<3> &item_ct1)
{
    double sum = 0.0;
    long ix = item_ct1.get_group(2)*item_ct1.get_local_range(2)+item_ct1.get_local_id(2);
    while (ix < nobs) {
        sum = 0.0;
        for(short inGlob=0;inGlob<nGlobP;inGlob++){
            sum=sum+systemMatrix_dev[ix * nGlobP + inGlob]*vVect_dev[offLocalGlob+inGlob];
        }
        knownTerms_dev[ix] = knownTerms_dev[ix] + sum;
        ix += item_ct1.get_group_range(2) * item_ct1.get_local_range(2);
    }
}


// //  CONSTRAINTS OF APROD MODE 1
/// ExtConstr
/// Mode 1 ExtConstr
void aprod1_Kernel_ExtConstr (double*  __restrict__ knownTerms_dev,
                                const double*  __restrict__ systemMatrix_dev, 
                                const double*  __restrict__ vVect_dev, 
                                const long VrIdAstroPDimMax,
                                const long mapNoss, 
                                const long nDegFreedomAtt, 
                                const int startingAttColExtConstr, 
                                const int nEqExtConstr, 
                                const int nOfElextObs, 
                                const int numOfExtStar, 
                                const int numOfExtAttCol, 
                                const short nAstroPSolved, 
                                const short nAttAxes,
                                const sycl::nd_item<3> &item_ct1)
{
    long offExtAtt;
    long offExtAttConstr = VrIdAstroPDimMax*nAstroPSolved+startingAttColExtConstr;
    long vVIx;
    long ktIx = mapNoss;
    long offExtConstr;
    long j3 = item_ct1.get_group(2)*item_ct1.get_local_range(2)+item_ct1.get_local_id(2);

    double sum = 0.0;

    for (int iexc = 0; iexc < nEqExtConstr; iexc++) {
        sum=0.0;
        offExtConstr = iexc*nOfElextObs;
        if (j3 < numOfExtStar*nAstroPSolved) {
            sum = sum + systemMatrix_dev[offExtConstr+j3]*vVect_dev[j3];
        }
        for (short nax = 0; nax < nAttAxes; nax++) {
            offExtAtt = offExtConstr + numOfExtStar*nAstroPSolved + nax*numOfExtAttCol;
            vVIx=offExtAttConstr+nax*nDegFreedomAtt;

            if (j3 < numOfExtAttCol) {
                sum += systemMatrix_dev[offExtAtt+j3]*vVect_dev[vVIx+j3];
            }
        }
        atomicAdd(&knownTerms_dev[ktIx + iexc], sum);
    }
}


/// BarConstr
/// Mode 1 BarConstr
void aprod1_Kernel_BarConstr(double*  __restrict__ knownTerms_dev,
                            const double*  __restrict__ systemMatrix_dev, 
                            const double*  __restrict__ vVect_dev,
                            const int nOfElextObs, 
                            const int nOfElBarObs, 
                            const int nEqExtConstr, 
                            const long mapNoss, 
                            const int nEqBarConstr, 
                            const int numOfBarStar, 
                            const short nAstroPSolved,
                            const sycl::nd_item<3> &item_ct1)
{
    long offBarConstrIx;
    long ktIx = mapNoss + nEqExtConstr;
    long j3=item_ct1.get_group(2)*item_ct1.get_local_range(2)+item_ct1.get_local_id(2);

    for(int iexc=0;iexc<nEqBarConstr;iexc++ ){
        double sum = 0.0;
        offBarConstrIx=iexc*nOfElBarObs;
        if (j3 < numOfBarStar*nAstroPSolved)
            sum = sum + systemMatrix_dev[offBarConstrIx+j3]*vVect_dev[j3];

        atomicAdd(&knownTerms_dev[ktIx + iexc], sum);
    }//for iexc
}


/// InstrConstr
/// Mode 1 InstrConstr
void aprod1_Kernel_InstrConstr (double*  __restrict__ knownTerms_dev,
                                const double*  __restrict__ systemMatrix_dev, 
                                const double*  __restrict__ vVect_dev,
                                const int*  __restrict__ instrConstrIlung_dev, 
                                const int*  __restrict__ instrCol_dev, 
                                const long VrIdAstroPDimMax, 
                                const long mapNoss, 
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
                                const short nInstrPSolved, 
                                 const sycl::nd_item<3> &item_ct1){

    
    const long ktIx=mapNoss+nEqExtConstr+nEqBarConstr;    
    const int i1=item_ct1.get_group(2)*item_ct1.get_local_range(2)+item_ct1.get_local_id(2);
    const int i1_Aux = myid + i1*nproc;
    long offSetInstrConstr1=VrIdAstroPDimMax*nAstroPSolved+nDegFreedomAtt*nAttAxes;
    long offSetInstrInc=nOfElextObs*nEqExtConstr+nOfElBarObs*nEqBarConstr;
    long offvV=0;
    int offSetInstr=0;
    int vVix=0;
    double sum = 0.0;
    
    if(i1_Aux < nOfInstrConstr)
    {
        sum=0.0;
        offSetInstr=0;
        for(int m=0;m<i1_Aux;m++)
        {
            offSetInstrInc+=instrConstrIlung_dev[m];
            offSetInstr+=instrConstrIlung_dev[m];
        }
        offvV=mapNoss*nInstrPSolved+offSetInstr;
        for(int j3 = 0; j3 < instrConstrIlung_dev[i1_Aux]; j3++)
        {
            vVix=instrCol_dev[offvV+j3];
            sum=sum+systemMatrix_dev[offSetInstrInc+j3]*vVect_dev[offSetInstrConstr1+vVix];
        }
        atomicAdd(&knownTerms_dev[ktIx + i1_Aux], sum);
    }
}
//-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
//-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
//-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
//-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
//-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

template<long THREADS_PER_ROW, long THREADS_PER_COL>
void aprod2_Kernel_astro(double * __restrict__ vVect_dev, 
                                    const double * __restrict__ systemMatrix_dev, 
                                    const double * __restrict__ knownTerms_dev, 
                                    const long* __restrict__ matrixIndexAstro_dev, 
                                    const long* __restrict__ startend_dev,  
                                    const long offLocalAstro, 
                                    const long mapNoss, 
                                    const short nAstroPSolved,
                                    const sycl::nd_item<3> &item_ct1,
                                    sycl::local_accessor<double, 3> sharedSum){

    const long ix = item_ct1.get_group(2)*item_ct1.get_local_range(2)+item_ct1.get_local_id(2); // x-dimension

    if (ix < mapNoss) {

        long p = startend_dev[ix];
        long q = startend_dev[ix+1];
        long tid = matrixIndexAstro_dev[startend_dev[ix]];

        for (short jx = 0; jx < nAstroPSolved; ++jx) {
            sharedSum[item_ct1.get_local_id(2)][item_ct1.get_local_id(1)][jx] =0.0;
        }

        item_ct1.barrier(sycl::access::fence_space::local_space);

        for (long i = p + item_ct1.get_local_id(1); i < q;
             i += THREADS_PER_COL) {
            
            for (short jx = 0; jx < nAstroPSolved; ++jx) {
                sharedSum[item_ct1.get_local_id(2)][item_ct1.get_local_id(1)][jx] += systemMatrix_dev[i*nAstroPSolved+jx]*knownTerms_dev[i];
            }
        }
 
        item_ct1.barrier(sycl::access::fence_space::local_space);
 
        for (int offset = THREADS_PER_COL / 2; offset > 0; offset >>= 1) {
            if (item_ct1.get_local_id(1) < offset) {
                
                for (short jx = 0; jx < nAstroPSolved; ++jx) {
                    sharedSum[item_ct1.get_local_id(2)][item_ct1.get_local_id(1)][jx] += sharedSum[item_ct1.get_local_id(2)][item_ct1.get_local_id(1) + offset][jx];
                }
            }
            item_ct1.barrier(sycl::access::fence_space::local_space);
        }

        if (item_ct1.get_local_id(1) == 0) {
            
            for (short jx = 0; jx < nAstroPSolved; ++jx) {
                vVect_dev[tid - offLocalAstro + jx]+=sharedSum[item_ct1.get_local_id(2)][0][jx]; // Write reduced sum
            }
        }
    }
}




void aprod2_Kernel_att_AttAxis(double * __restrict__ vVect_dev,                         //0
                                const double * __restrict__ systemMatrix_dev,           //1
                                const double * __restrict__ knownTerms_dev,             //2
                                const long*  __restrict__ matrixIndexAtt_dev,           //3
                                const long nAttP,                                       //4     
                                const long nDegFreedomAtt,                              //5    
                                const long offLocalAtt,                                 //6
                                const long nobs,                                        //7
                                const short  nAstroPSolved,                             //8
                                const short  nAttParAxis,                               //9
                               const sycl::nd_item<3> &item_ct1)                        //10                
{
    long ix = item_ct1.get_group(2)*item_ct1.get_local_range(2)+item_ct1.get_local_id(2);
    while(ix < nobs)
    {
        long jstartAtt = matrixIndexAtt_dev[ix] + offLocalAtt;
        
        for (short inpax = 0; inpax < nAttParAxis; ++inpax)
            atomicAdd(&vVect_dev[jstartAtt + inpax],systemMatrix_dev[ix * nAttP + inpax] * knownTerms_dev[ix]);
        jstartAtt +=nDegFreedomAtt;
        
        
        for (short inpax = 0; inpax < nAttParAxis; ++inpax)
            atomicAdd(&vVect_dev[jstartAtt + inpax],systemMatrix_dev[ix * nAttP + nAttParAxis + inpax]*knownTerms_dev[ix]);
        jstartAtt +=nDegFreedomAtt;
        
        
        for (short inpax = 0; inpax < nAttParAxis; ++inpax)
            atomicAdd(&vVect_dev[jstartAtt + inpax],systemMatrix_dev[ix * nAttP + nAttParAxis + nAttParAxis +inpax] *knownTerms_dev[ix]);
        ix += item_ct1.get_group_range(2) * item_ct1.get_local_range(2);

   }
}


void aprod2_Kernel_instr(double * __restrict__ vVect_dev, 
                        const double * __restrict__ systemMatrix_dev, 
                        const double * __restrict__ knownTerms_dev, 
                        const int * __restrict__ instrCol_dev,  
                        const long offLocalInstr, 
                        const long nobs, 
                        const short nInstrPSolved,
                         const sycl::nd_item<3> &item_ct1)
{
    long ix=item_ct1.get_group(2)*item_ct1.get_local_range(2)+item_ct1.get_local_id(2);

    while(ix < nobs)
    {
        
        for (short inInstr = 0; inInstr < nInstrPSolved; inInstr++){
            double MatVal{systemMatrix_dev[ix*nInstrPSolved + inInstr]};
            double rhs{knownTerms_dev[ix]};
            double tmp=MatVal*rhs;
            atomicAdd(&vVect_dev[offLocalInstr+instrCol_dev[ix * nInstrPSolved + inInstr]],tmp);
        }
        ix += item_ct1.get_group_range(2) * item_ct1.get_local_range(2);
   }
}


void sumCommMultiBlock_double_aprod2_Kernel_glob(double * __restrict__ dev_vVect_glob_sum, 
                                                const double * __restrict__ systemMatrix_dev, 
                                                const double * __restrict__ knownTerms_dev, 
                                                const double * __restrict__ vVect_dev, 
                                                const long nGlobP, 
                                                const long nobs,  
                                                const long offLocalGlob, 
                                                const int inGlob,
                                                 const sycl::nd_item<3> &item_ct1,
                                                 double *shArr)
{

    long gthIdx = item_ct1.get_local_id(2) + item_ct1.get_group(2) * blockSize;
    const int gridSize = blockSize * item_ct1.get_group_range(2);

    shArr[item_ct1.get_local_id(2)] = 0.0;

    for (long ix = gthIdx; ix < nobs; ix += gridSize)
        shArr[item_ct1.get_local_id(2)] +=
            systemMatrix_dev[ix * nGlobP + inGlob] * knownTerms_dev[ix];

    item_ct1.barrier(sycl::access::fence_space::local_space);
    for (int size = blockSize/2; size>0; size/=2) {
        if (item_ct1.get_local_id(2) < size)
            shArr[item_ct1.get_local_id(2)] +=
                shArr[item_ct1.get_local_id(2) + size];
        item_ct1.barrier(sycl::access::fence_space::local_space);
    }
    if (item_ct1.get_local_id(2) == 0)
        dev_vVect_glob_sum[item_ct1.get_group(2)] = shArr[0];
}



void realsumCommMultiBlock_double_aprod2_Kernel_glob(double * __restrict__ vVect_dev, 
                                                    const double * __restrict__ gArr, 
                                                    const long arraySize, 
                                                    const long offLocalGlob, 
                                                    const short inGlob,
                                                     const sycl::nd_item<3> &item_ct1,
                                                     double *shArr)
{

    int thIdx = item_ct1.get_local_id(2);
    long gthIdx = thIdx + item_ct1.get_group(2) * blockSize;

    const int gridSize = blockSize * item_ct1.get_group_range(2);
    double sum = 0.0;
    for (long i = gthIdx; i < arraySize; i += gridSize)
        sum += gArr[i];

    shArr[thIdx] = sum;
    item_ct1.barrier();
    for (int size = blockSize/2; size>0; size/=2) {
        if (thIdx<size)
            shArr[thIdx] += shArr[thIdx+size];
        item_ct1.barrier();
    }
    if (thIdx == 0)
    {
        vVect_dev[offLocalGlob + inGlob] = vVect_dev[offLocalGlob + inGlob] + shArr[0];
    }
}


//  CONSTRAINTS OF APROD MODE 2
void aprod2_Kernel_ExtConstr (double*  __restrict__ vVect_dev, 
                            const double*  __restrict__ systemMatrix_dev, 
                            const double*  __restrict__ knownTerms_dev, 
                            const long mapNoss, 
                            const long nDegFreedomAtt, 
                            const long VrIdAstroPDimMax, 
                            const int nEqExtConstr, 
                            const int nOfElextObs, 
                            const int numOfExtStar, 
                            const int startingAttColExtConstr, 
                            const int numOfExtAttCol,
                            const short  nAttAxes, 
                            const short nAstroPSolved,
                            const sycl::nd_item<3> &item_ct1)
{
    const long off1 = VrIdAstroPDimMax*nAstroPSolved+startingAttColExtConstr;
    const int i =item_ct1.get_group(2)*item_ct1.get_local_range(2)+item_ct1.get_local_id(2);
    long off2;
    long off3;
    long offExtAttConstrEq;
    long offExtUnk;
    
    double yi;
    for(int ix = 0; ix < nEqExtConstr; ix++ ){ 
        yi = knownTerms_dev[mapNoss + ix];

        if (i < numOfExtStar) {
            off3 = i*nAstroPSolved;
            off2 = ix*nOfElextObs + off3;
            for(short j2 = 0; j2 < nAstroPSolved; j2++){
                vVect_dev[j2+off3] += systemMatrix_dev[off2+j2]*yi;
            }
        }
    } 


    for(int ix=0;ix<nEqExtConstr;ix++ ){  
        yi = knownTerms_dev[mapNoss + ix];
        offExtAttConstrEq =  ix*nOfElextObs;

        offExtAttConstrEq += numOfExtStar*nAstroPSolved; 
        for(short nax = 0; nax < nAttAxes; nax++){
            offExtUnk = off1 + nax*nDegFreedomAtt;
            off2=offExtAttConstrEq+nax*numOfExtAttCol;

            if (i < numOfExtAttCol) {
                vVect_dev[offExtUnk+i] = vVect_dev[offExtUnk+i] + systemMatrix_dev[off2+i]*yi;
            }
        }
    }
}



void aprod2_Kernel_BarConstr(double*  __restrict__ vVect_dev,
                            const double*  __restrict__ systemMatrix_dev, 
                            const double*  __restrict__ knownTerms_dev, 
                            const long mapNoss, 
                            const int nEqBarConstr, 
                            const int nEqExtConstr, 
                            const int nOfElextObs, 
                            const int nOfElBarObs, 
                            const int numOfBarStar, 
                            const short nAstroPSolved,
                            const sycl::nd_item<3> &item_ct1)
{
    const int yx = item_ct1.get_group(2)*item_ct1.get_local_range(2)+item_ct1.get_local_id(2);
    double yi;
    long offBarStarConstrEq;

    for(int ix=0;ix<nEqBarConstr;ix++ ){  
        yi = knownTerms_dev[mapNoss+nEqExtConstr+ix];
        offBarStarConstrEq = nEqExtConstr*nOfElextObs+ix*nOfElBarObs;
        if (yx < numOfBarStar) {
            for(short j2=0;j2<nAstroPSolved;j2++){
                vVect_dev[j2+yx*nAstroPSolved] = vVect_dev[j2+yx*nAstroPSolved] + systemMatrix_dev[offBarStarConstrEq+yx*nAstroPSolved+j2]*yi;
            }
        }
    } 
}




void aprod2_Kernel_InstrConstr (double*  __restrict__ vVect_dev,
                                const double*  __restrict__ systemMatrix_dev, 
                                const double*  __restrict__ knownTerms_dev, 
                                const int*  __restrict__ instrConstrIlung_dev, 
                                const int*  __restrict__ instrCol_dev, 
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
                                const short nAstroPSolved, 
                                const short  nAttAxes, 
                                const short  nInstrPSolved,
                                const sycl::nd_item<3> &item_ct1){

    const long k1 = item_ct1.get_group(2)*item_ct1.get_local_range(2)+item_ct1.get_local_id(2);
    long k1_Aux = myid + k1*nproc;
    const long off3=nOfElextObs*nEqExtConstr+nOfElBarObs*nEqBarConstr;
    const long offInstrUnk=VrIdAstroPDimMax*nAstroPSolved+nAttAxes*nDegFreedomAtt;
    const long off2=mapNoss+nEqExtConstr+nEqBarConstr;
    const long off4=mapNoss*nInstrPSolved;
    double yi;
    
    if(k1_Aux < nOfInstrConstr) {
        yi=knownTerms_dev[off2+k1_Aux];
        int offSetInstr=0;
        for(long m=0;m<k1_Aux;m++)
            offSetInstr+=instrConstrIlung_dev[m];

        const long off1=off3+offSetInstr;
        const long off5=off4+offSetInstr;
        for(int j = 0; j < instrConstrIlung_dev[k1_Aux]; j++) {
            atomicAdd(&vVect_dev[offInstrUnk + instrCol_dev[off5 + j]],systemMatrix_dev[off1 + j] * yi);
            }
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


    sycl::queue myQueue;  

    auto platform = myQueue.get_context().get_platform();
    auto devices = platform.get_devices();
    int deviceCount = devices.size();
    int deviceNum = myid % deviceCount;
    auto selectedDevice = devices[deviceNum];

    #ifdef USE_MPI
        sycl::queue queue(sycl::default_selector{},sycl::property_list{sycl::property::queue::in_order{}});
        sycl::queue queue_Aprod2_0(sycl::default_selector{},sycl::property_list{sycl::property::queue::in_order{}});
        sycl::queue queue_Aprod2_1(sycl::default_selector{},sycl::property_list{sycl::property::queue::in_order{}});
        sycl::queue queue_Aprod2_2(sycl::default_selector{},sycl::property_list{sycl::property::queue::in_order{}});
        sycl::queue queue_Aprod2_3(sycl::default_selector{},sycl::property_list{sycl::property::queue::in_order{}});
        sycl::queue queue_Aprod2_4(sycl::default_selector{},sycl::property_list{sycl::property::queue::in_order{}});
        sycl::queue queue_Aprod2_5(sycl::default_selector{},sycl::property_list{sycl::property::queue::in_order{}});
        sycl::queue queue_Aprod2_6(sycl::default_selector{},sycl::property_list{sycl::property::queue::in_order{}});
        sycl::queue queue_Aprod2_7(sycl::default_selector{},sycl::property_list{sycl::property::queue::in_order{}});
    #else
        sycl::queue queue(sycl::default_selector{},sycl::property_list{sycl::property::queue::in_order{},sycl::property::queue::enable_profiling()});
        sycl::queue queue_Aprod2_0(sycl::default_selector{},sycl::property_list{sycl::property::queue::in_order{},sycl::property::queue::enable_profiling()});
        sycl::queue queue_Aprod2_1(sycl::default_selector{},sycl::property_list{sycl::property::queue::in_order{},sycl::property::queue::enable_profiling()});
        sycl::queue queue_Aprod2_2(sycl::default_selector{},sycl::property_list{sycl::property::queue::in_order{},sycl::property::queue::enable_profiling()});
        sycl::queue queue_Aprod2_3(sycl::default_selector{},sycl::property_list{sycl::property::queue::in_order{},sycl::property::queue::enable_profiling()});
        sycl::queue queue_Aprod2_4(sycl::default_selector{},sycl::property_list{sycl::property::queue::in_order{},sycl::property::queue::enable_profiling()});
        sycl::queue queue_Aprod2_5(sycl::default_selector{},sycl::property_list{sycl::property::queue::in_order{},sycl::property::queue::enable_profiling()});
        sycl::queue queue_Aprod2_6(sycl::default_selector{},sycl::property_list{sycl::property::queue::in_order{},sycl::property::queue::enable_profiling()});
        sycl::queue queue_Aprod2_7(sycl::default_selector{},sycl::property_list{sycl::property::queue::in_order{},sycl::property::queue::enable_profiling()});
    #endif

    int
        istop  = 0,
        itn    = 0;
    double
        anorm  = ZERO,
        acond  = ZERO,
        rnorm  = ZERO,
        arnorm = ZERO,
        xnorm  = ZERO;
       
    //  Local variables
    const bool
        damped = damp > ZERO,
        wantse = standardError != NULL;

    double
        alpha, beta, bnorm,
        cs, cs1, cs2, ctol,
        delta, dknorm, dnorm, dxk, dxmax,
        gamma, gambar, phi, phibar, psi,
        res2, rho, rhobar, rhbar1,
        rhs, rtol, sn, sn1, sn2,
        t, tau, temp, test1, test3,
        theta, t1, t2, t3, xnorm1, z, zbar;
    double test2=0;
    
    //-----------------------------------------------------------------------
    ///////////// Specific definitions
    long  other; 
    int nAstroElements;
    
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


    double alphaLoc2;

    long nElemKnownTerms = mapNoss+nEqExtConstr+nEqBarConstr+nOfInstrConstr;
    long nTotConstraints  =   nOfElextObs*nEqExtConstr+nOfElBarObs*nEqBarConstr+nElemIC;

    
    double max_knownTerms;
    double ssq_knownTerms;
    double max_vVect;  
    double ssq_vVect;

    double  *sysmatAstro_dev        =nullptr;
    double  *sysmatAtt_dev          =nullptr;
    double  *sysmatInstr_dev        =nullptr;
    double  *sysmatGloB_dev         =nullptr;
    double  *sysmatConstr_dev       =nullptr;
    double  *vVect_dev              =nullptr;
    double  *knownTerms_dev         =nullptr; 
    double  *wVect_dev              =nullptr;
    double  *kAuxcopy_dev           =nullptr;
    double  *vAuxVect_dev           =nullptr;

    double *dev_vVect_glob_sum      =nullptr;
    double *dev_max_knownTerms      =nullptr;
    double *dev_ssq_knownTerms      =nullptr;
    double *dev_max_vVect           =nullptr;
    double *dev_ssq_vVect           =nullptr;
    double *xSolution_dev           =nullptr;
    double *standardError_dev       =nullptr;
    double *dknorm_vec              =nullptr;

            
    long    *matrixIndexAstro_dev   =nullptr;
    long    *matrixIndexAtt_dev     =nullptr;
    long    *startend_dev           =nullptr;

    int     *instrCol_dev           =nullptr; 
    int     *instrConstrIlung_dev   =nullptr;


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

    matrixIndexAstro_dev = sycl::malloc_device<long>(mapNoss, queue);
    matrixIndexAtt_dev = sycl::malloc_device<long>(mapNoss, queue);
    startend_dev = sycl::malloc_device<long>((nnz + 1), queue);
    
    queue.memcpy(matrixIndexAstro_dev, matrixIndexAstro,mapNoss*sizeof(long)).wait();
    queue.memcpy(startend_dev,startend,sizeof(long)*(nnz+1)).wait();
    queue.memcpy(matrixIndexAtt_dev, matrixIndexAtt,mapNoss*sizeof(long)).wait();

    //--------------------------------------------------------------------------------------------------------

    if(nAstroPSolved){
        sysmatAstro_dev = sycl::malloc_device<double>(mapNoss*nAstroPSolved, queue);
        queue.memcpy(sysmatAstro_dev, sysmatAstro,mapNoss*nAstroPSolved*sizeof(double)).wait();
    }
    if(nAttP){
        sysmatAtt_dev = sycl::malloc_device<double>(mapNoss * nAttP, queue);
        queue.memcpy(sysmatAtt_dev,sysmatAtt,mapNoss*nAttP*sizeof(double)).wait();
    }
    if(nInstrPSolved){
        sysmatInstr_dev = sycl::malloc_device<double>(mapNoss*nInstrPSolved, queue);
        queue.memcpy(sysmatInstr_dev, sysmatInstr,mapNoss*nInstrPSolved*sizeof(double)).wait();
    }
    if(nGlobP){
        sysmatGloB_dev = sycl::malloc_device<double>(mapNoss * nGlobP, queue);
        queue.memcpy(sysmatGloB_dev,sysmatGloB,mapNoss*nGlobP*sizeof(double)).wait();
    }
    if(nTotConstraints){
        sysmatConstr_dev = sycl::malloc_device<double>((nTotConstraints), queue);
        queue.memcpy(sysmatConstr_dev, sysmatConstr,(nTotConstraints)*sizeof(double)).wait();
    }


    vVect_dev            = sycl::malloc_device<double>(nunkSplit, queue);
    knownTerms_dev       = sycl::malloc_device<double>(nElemKnownTerms, queue);
    wVect_dev            = sycl::malloc_device<double>(nunkSplit, queue);
    kAuxcopy_dev         = sycl::malloc_device<double>((nEqExtConstr + nEqBarConstr + nOfInstrConstr),queue);
    vAuxVect_dev         = sycl::malloc_device<double>(localAstroMax, queue);
    instrCol_dev         = sycl::malloc_device<int>((nInstrPSolved * mapNoss + nElemIC),queue); 
    instrConstrIlung_dev = sycl::malloc_device<int>(nOfInstrConstr, queue);
    xSolution_dev        = sycl::malloc_device<double>(nunkSplit, queue);
    standardError_dev    = sycl::malloc_device<double>(nunkSplit, queue);
    dknorm_vec           = sycl::malloc_device<double>(1, queue);


    //  Copies H2D:
    queue.memcpy(instrCol_dev,instrCol,(nInstrPSolved * mapNoss + nElemIC) * sizeof(int)).wait(); 
    queue.memcpy(instrConstrIlung_dev, instrConstrIlung,nOfInstrConstr * sizeof(int)).wait();
    queue.memcpy(knownTerms_dev, knownTerms,nElemKnownTerms * sizeof(double)).wait();
    dev_vVect_glob_sum =sycl::malloc_device<double>(gridSize, queue);
    dev_max_knownTerms =sycl::malloc_device<double>(gridSize, queue);
    dev_ssq_knownTerms =sycl::malloc_device<double>(gridSize, queue);
    dev_max_vVect =sycl::malloc_device<double>(gridSize, queue);
    dev_ssq_vVect=sycl::malloc_device<double>(gridSize, queue);

    //  Grid topologies:
    sycl::range<3> gridDim_aprod1(1, 1, (mapNoss - 1) / TILE_WIDTH + 1);
    sycl::range<3> gridDim_aprod1astro(1, 1, (mapNoss - 1) / ThreadsXBlockAprod1Astro + 1);
    sycl::range<3> gridDim_aprod1attinstr(1, 1, (mapNoss - 1) / ThreadsXBlockAprod1AttInstr + 1);


    sycl::range<3> gridDim_aprod1_Plus_Constr(1, 1,(nElemKnownTerms - 1)/TILE_WIDTH +1);
    sycl::range<3> gridDim_vAuxVect_Kernel(1, 1, (localAstroMax - 1) / TILE_WIDTH + 1);
    sycl::range<3> gridDim_vVect_Put_To_Zero_Kernel(1, 1, (nunkSplit - 1) / TILE_WIDTH + 1);
    sycl::range<3> gridDim_nunk(1, 1, (nunkSplit - 1) / TILE_WIDTH + 1);
    sycl::range<3> gridDim_aprod2(1, 1, (mapNoss - 1) / TILE_WIDTH + 1);
    sycl::range<3> gridDim_aprod2attinstr(1, 1, (mapNoss - 1) / ThreadsXBlockAprod2AttInstr + 1);

    sycl::range<3> gridDim_kAuxcopy_Kernel(1, 1,((nEqExtConstr + nEqBarConstr + nOfInstrConstr) - 1) / TILE_WIDTH + 1);

    //  Grid topologies for the constraints sections:
    const int numOfExtStarTimesnAstroPSolved = numOfExtStar*nAstroPSolved;
    int max_numOfExtStarTimesnAstroPSolved_numOfExtAttCol;
    max_numOfExtStarTimesnAstroPSolved_numOfExtAttCol = numOfExtStarTimesnAstroPSolved;
    if (numOfExtStarTimesnAstroPSolved < numOfExtAttCol)
        max_numOfExtStarTimesnAstroPSolved_numOfExtAttCol = numOfExtAttCol;

    sycl::range<3> gridDim_aprod1_ExtConstr(1, 1,(max_numOfExtStarTimesnAstroPSolved_numOfExtAttCol - 1) / TILE_WIDTH +1);
    sycl::range<3> gridDim_aprod1_BarConstr(1, 1, (numOfBarStar * nAstroPSolved - 1) / TILE_WIDTH + 1);
    sycl::range<3> gridDim_aprod1_InstrConstr(1, 1, (nOfInstrConstr - 1) / TILE_WIDTH + 1);

    int max_numOfExtStar_numOfExtAttCol;
    max_numOfExtStar_numOfExtAttCol = numOfExtStar;
    if (numOfExtStar < numOfExtAttCol)  max_numOfExtStar_numOfExtAttCol = numOfExtAttCol;

    sycl::range<3> gridDim_aprod2_ExtConstr(1, 1, (max_numOfExtStar_numOfExtAttCol - 1) / TILE_WIDTH + 1);
    sycl::range<3> gridDim_aprod2_BarConstr(1, 1, (numOfBarStar - 1) / TILE_WIDTH + 1);
    sycl::range<3> gridDim_aprod2_InstrConstr(1, 1, (nOfInstrConstr - 1) / TILE_WIDTH + 1);

////////////////////////////////////////////// 
other=(long)nAttParam + nInstrParam + nGlobalParam; 
// comlsqr.itn=itn;


if(ThreadsXBlockAprod2Astro*RowsPerBlock>1024){
    std::cerr<<"ThreadsXBlockAprod2Astro*RowsPerBlock should not bigger than 1024 max threads "<<std::endl;
    abort();
}

sycl::range<3> gridAprod2astro(1, 1, (nnz + RowsPerBlock - 1) / RowsPerBlock);
sycl::range<3> blockAprod2astro(1, ThreadsXBlockAprod2Astro, RowsPerBlock);

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

dload( nunkSplit, ZERO, vVect );

queue.memcpy(vVect_dev, vVect, nunkSplit * sizeof(double)).wait();

dload( nunkSplit, ZERO, xSolution );

if ( wantse )   dload( nunkSplit, ZERO, standardError );

queue.memcpy(xSolution_dev, xSolution, nunkSplit * sizeof(double)).wait();
queue.memcpy(standardError_dev, standardError, nunkSplit * sizeof(double)).wait();


alpha  =   ZERO;
    
{
    queue.submit([&](sycl::handler &cgh) {
        sycl::accessor<double, 1, sycl::access::mode::read_write, sycl::access::target::local> shArr_acc_ct1(sycl::range<1>(blockSize), cgh);
        cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, gridSize)*sycl::range<3>(1, 1, blockSize),sycl::range<3>(1, 1, blockSize)),
                        [=](sycl::nd_item<3> item_ct1) {
                            maxCommMultiBlock_double<blockSize>(knownTerms_dev, dev_max_knownTerms,nElemKnownTerms, item_ct1,shArr_acc_ct1.get_pointer());
                        });
    });
}
{
    queue.submit([&](sycl::handler &cgh) {        
        sycl::accessor<double, 1, sycl::access::mode::read_write, sycl::access::target::local> shArr_acc_ct1(sycl::range<1>(blockSize), cgh);
        const int gridSize_ct2 = gridSize;
        cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, blockSize),sycl::range<3>(1, 1, blockSize)),
                            [=](sycl::nd_item<3> item_ct1) {
                                maxCommMultiBlock_double<blockSize>(dev_max_knownTerms, dev_max_knownTerms,gridSize_ct2, item_ct1,shArr_acc_ct1.get_pointer());
                            });
    });
}
queue.memcpy(&max_knownTerms, dev_max_knownTerms, sizeof(double)).wait();

double betaLoc, betaLoc2;
if (myid == 0) {
        {
            queue.submit([&](sycl::handler &cgh) {
                sycl::accessor<double, 1, sycl::access::mode::read_write, sycl::access::target::local> shArr_acc_ct1(sycl::range<1>(blockSize), cgh);
                cgh.parallel_for(
                    sycl::nd_range<3>(sycl::range<3>(1, 1, gridSize) *sycl::range<3>(1, 1, blockSize),sycl::range<3>(1, 1, blockSize)),
                    [=](sycl::nd_item<3> item_ct1) {
                        sumCommMultiBlock_double<blockSize>(knownTerms_dev, dev_ssq_knownTerms, max_knownTerms,mapNoss + nEqExtConstr +nEqBarConstr +nOfInstrConstr,item_ct1, shArr_acc_ct1.get_pointer());
                    });
            });
        }
        {
            queue.submit([&](sycl::handler &cgh) {
                sycl::accessor<double, 1, sycl::access::mode::read_write, sycl::access::target::local> shArr_acc_ct1(sycl::range<1>(blockSize), cgh);
                const int gridSize_ct2 = gridSize;
                cgh.parallel_for(
                    sycl::nd_range<3>(sycl::range<3>(1, 1, blockSize),sycl::range<3>(1, 1, blockSize)),
                    [=](sycl::nd_item<3> item_ct1) {
                        realsumCommMultiBlock_double<blockSize>(dev_ssq_knownTerms, dev_ssq_knownTerms,gridSize_ct2, item_ct1,shArr_acc_ct1.get_pointer());
                    });
            });
        }
    queue.memcpy(&ssq_knownTerms, dev_ssq_knownTerms, sizeof(double)).wait();
    betaLoc = max_knownTerms*sqrt(ssq_knownTerms);
} else {
    {
        queue.submit([&](sycl::handler &cgh) {
            sycl::accessor<double, 1, sycl::access::mode::read_write, sycl::access::target::local> shArr_acc_ct1(sycl::range<1>(blockSize), cgh);
            cgh.parallel_for(
                sycl::nd_range<3>(sycl::range<3>(1, 1, gridSize) *sycl::range<3>(1, 1, blockSize),sycl::range<3>(1, 1, blockSize)),
                [=](sycl::nd_item<3> item_ct1) {
                    sumCommMultiBlock_double<blockSize>(knownTerms_dev, dev_ssq_knownTerms, max_knownTerms,mapNoss, item_ct1, shArr_acc_ct1.get_pointer());
                });
        });
    }
    {
        queue.submit([&](sycl::handler &cgh) {
            sycl::accessor<double, 1, sycl::access::mode::read_write, sycl::access::target::local> shArr_acc_ct1(sycl::range<1>(blockSize), cgh);
            const int gridSize_ct2 = gridSize;
            cgh.parallel_for(
                sycl::nd_range<3>(sycl::range<3>(1, 1, blockSize),sycl::range<3>(1, 1, blockSize)),
                [=](sycl::nd_item<3> item_ct1) {
                    realsumCommMultiBlock_double<blockSize>(dev_ssq_knownTerms, dev_ssq_knownTerms,gridSize_ct2, item_ct1,shArr_acc_ct1.get_pointer());
                });
        });
    }
    queue.memcpy(&ssq_knownTerms, dev_ssq_knownTerms, sizeof(double)).wait();
    betaLoc = max_knownTerms*sqrt(ssq_knownTerms);
}
    

betaLoc2=betaLoc*betaLoc;

#ifdef USE_MPI
    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    //------------------------------------------------------------------------------------------------  TIME 2
    starttime=MPI_Wtime();
    MPI_Allreduce(MPI_IN_PLACE,&betaLoc2,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
    communicationtime+=MPI_Wtime()-starttime;
    //------------------------------------------------------------------------------------------------
#endif

beta=sqrt(betaLoc2);

    
if (beta > ZERO) 
{
    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::: TIME3
    {
        queue.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, BlockXGrid)*sycl::range<3>(1, 1, ThreadsXBlock),sycl::range<3>(1, 1, ThreadsXBlock)),
            [=](sycl::nd_item<3> item_ct1) {
                dscal(knownTerms_dev, 1.0 / beta,nElemKnownTerms,ONE, item_ct1);
            });
    }


	if(myid!=0)
	{

        queue.parallel_for(sycl::nd_range<3>(gridDim_vVect_Put_To_Zero_Kernel*sycl::range<3>(1, 1, TILE_WIDTH),sycl::range<3>(1, 1, TILE_WIDTH)),
                [=](sycl::nd_item<3> item_ct1) {
                    vVect_Put_To_Zero_Kernel(vVect_dev, localAstroMax,nunkSplit, item_ct1);
                });
        }
        queue.wait();

        //APROD2 CALL BEFORE LSQR
        {

            if (nAstroPSolved) {
                queue_Aprod2_0.submit([&](sycl::handler &cgh) {
                    sycl::local_accessor<double, 3> sharedSum_acc_ct1(sycl::range<3>(RowsPerBlock, ThreadsXBlockAprod2Astro, 5),cgh);

                    cgh.parallel_for(
                        sycl::nd_range<3>(gridAprod2astro*blockAprod2astro,blockAprod2astro),
                        [=](sycl::nd_item<3> item_ct1) {
                            aprod2_Kernel_astro<RowsPerBlock,ThreadsXBlockAprod2Astro>(
                                vVect_dev, sysmatAstro_dev, knownTerms_dev,matrixIndexAstro_dev, startend_dev, offLocalAstro,nnz, nAstroPSolved, item_ct1, sharedSum_acc_ct1);
                        });
                });
            }

        }

        {
            if(nAttP){
                queue_Aprod2_1.parallel_for(sycl::nd_range<3>(gridDim_aprod2attinstr*sycl::range<3>(1, 1, ThreadsXBlockAprod2AttInstr), sycl::range<3>(1, 1, ThreadsXBlockAprod2AttInstr)),
                    [=](sycl::nd_item<3> item_ct1) {
                        aprod2_Kernel_att_AttAxis(vVect_dev, sysmatAtt_dev, knownTerms_dev,matrixIndexAtt_dev, nAttP, nDegFreedomAtt, offLocalAtt,mapNoss, nAstroPSolved, nAttParAxis, item_ct1);
                    });
            }
        }

        {
            if(nInstrPSolved){
                queue_Aprod2_2.parallel_for(sycl::nd_range<3>(gridDim_aprod2attinstr*sycl::range<3>(1, 1, ThreadsXBlockAprod2AttInstr),sycl::range<3>(1, 1, ThreadsXBlockAprod2AttInstr)),
                    [=](sycl::nd_item<3> item_ct1) {
                        aprod2_Kernel_instr(vVect_dev, sysmatInstr_dev,knownTerms_dev, instrCol_dev,offLocalInstr, mapNoss, nInstrPSolved,item_ct1);
                    });
            }
        }


        for (short inGlob = 0; inGlob < nGlobP; inGlob++)
        {
                {
                    queue_Aprod2_3.submit([&](sycl::handler &cgh) {
                        sycl::accessor<double, 1, sycl::access::mode::read_write, sycl::access::target::local> shArr_acc_ct1(sycl::range<1>(blockSize), cgh);
                        cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, gridSize)*sycl::range<3>(1, 1, blockSize),sycl::range<3>(1, 1, blockSize)),
                            [=](sycl::nd_item<3> item_ct1) {
                                sumCommMultiBlock_double_aprod2_Kernel_glob(dev_vVect_glob_sum, sysmatGloB_dev,knownTerms_dev, vVect_dev, nGlobP, mapNoss,offLocalGlob, inGlob, item_ct1,shArr_acc_ct1.get_pointer());
                            });
                    });
                }
                {
                    queue_Aprod2_4.submit([&](sycl::handler &cgh) {
                        sycl::accessor<double, 1, sycl::access::mode::read_write, sycl::access::target::local> shArr_acc_ct1(sycl::range<1>(blockSize), cgh);
                        const long gridSize_ct2 = gridSize;
                        cgh.parallel_for(
                            sycl::nd_range<3>(sycl::range<3>(1, 1, blockSize),sycl::range<3>(1, 1, blockSize)),
                            [=](sycl::nd_item<3> item_ct1) {
                                realsumCommMultiBlock_double_aprod2_Kernel_glob(vVect_dev, dev_vVect_glob_sum, gridSize_ct2,offLocalGlob, inGlob, item_ct1,shArr_acc_ct1.get_pointer());
                            });
                    });
                }
        }

    queue_Aprod2_0.wait_and_throw();
    queue_Aprod2_1.wait_and_throw();
    queue_Aprod2_2.wait_and_throw();
    queue_Aprod2_3.wait_and_throw();
    queue_Aprod2_4.wait_and_throw();


    //  CONSTRAINTS OF APROD MODE 2:
    /* ExtConstr */
        {
            if(nEqExtConstr){
                queue_Aprod2_5.parallel_for(
                    sycl::nd_range<3>(gridDim_aprod2_ExtConstr*sycl::range<3>(1, 1, TILE_WIDTH),sycl::range<3>(1, 1, TILE_WIDTH)),
                    [=](sycl::nd_item<3> item_ct1) {
                            aprod2_Kernel_ExtConstr(vVect_dev,sysmatConstr_dev,knownTerms_dev,mapNoss,nDegFreedomAtt,VrIdAstroPDimMax,nEqExtConstr,nOfElextObs,numOfExtStar, 
                            startingAttColExtConstr,numOfExtAttCol,nAttAxes, nAstroPSolved,item_ct1);
                    });
            }
        }
    /* BarConstr */
        {
            if(nEqBarConstr){
                queue_Aprod2_6.parallel_for(
                    sycl::nd_range<3>(gridDim_aprod2_BarConstr*sycl::range<3>(1, 1, TILE_WIDTH),sycl::range<3>(1, 1, TILE_WIDTH)),
                    [=](sycl::nd_item<3> item_ct1) {
                            aprod2_Kernel_BarConstr(vVect_dev,sysmatConstr_dev, knownTerms_dev,mapNoss,nEqBarConstr,nEqExtConstr, 
                            nOfElextObs,nOfElBarObs,numOfBarStar, nAstroPSolved,item_ct1);
                    });
            }
        }
    // /* InstrConstr */
        {
            if(nOfInstrConstr){
                queue_Aprod2_7.parallel_for(
                    sycl::nd_range<3>(gridDim_aprod2_InstrConstr *sycl::range<3>(1, 1, TILE_WIDTH),sycl::range<3>(1, 1, TILE_WIDTH)),
                    [=](sycl::nd_item<3> item_ct1) {
                            aprod2_Kernel_InstrConstr(vVect_dev,sysmatConstr_dev,knownTerms_dev,instrConstrIlung_dev,instrCol_dev,VrIdAstroPDimMax,nDegFreedomAtt,mapNoss,nEqExtConstr, 
                            nEqBarConstr,nOfElextObs,nOfElBarObs,myid,nOfInstrConstr,nproc, nAstroPSolved,nAttAxes,nInstrPSolved, item_ct1);   
                    });
            }
        }

    queue_Aprod2_5.wait_and_throw();
    queue_Aprod2_6.wait_and_throw();
    queue_Aprod2_7.wait_and_throw();

    #ifdef USE_MPI
        /* ~~~~~~~~~~~~~~ */
        queue.memcpy(vVect, vVect_dev, nunkSplit * sizeof(double)).wait();
        //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        //------------------------------------------------------------------------------------------------  TIME4
        starttime=MPI_Wtime();
        MPI_Allreduce(MPI_IN_PLACE,&vVect[localAstroMax], nAttParam+nInstrParam+nGlobalParam,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
        communicationtime+=MPI_Wtime()-starttime;
        //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        //------------------------------------------------------------------------------------------------
        if(nAstroPSolved) SumCirc(vVect,comlsqr);
        //------------------------------------------------------------------------------------------------
        //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        queue.memcpy(vVect_dev, vVect, nunkSplit * sizeof(double)).wait();
    #endif

    nAstroElements=comlsqr.mapStar[myid][1]-comlsqr.mapStar[myid][0] + 1;
 	if(myid<nproc-1)
 	{
 		nAstroElements=comlsqr.mapStar[myid][1]-comlsqr.mapStar[myid][0] +1;
 		if(comlsqr.mapStar[myid][1]==comlsqr.mapStar[myid+1][0]) nAstroElements--;
 	}

    // reset internal state

        {
            queue.submit([&](sycl::handler &cgh) {
                sycl::accessor<double, 1, sycl::access::mode::read_write, sycl::access::target::local> shArr_acc_ct1(sycl::range<1>(blockSize), cgh);
                cgh.parallel_for(
                    sycl::nd_range<3>(sycl::range<3>(1, 1, gridSize) *sycl::range<3>(1, 1, blockSize),sycl::range<3>(1, 1, blockSize)),
                    [=](sycl::nd_item<3> item_ct1) {
                        maxCommMultiBlock_double<blockSize>(vVect_dev, dev_max_vVect,nunkSplit, item_ct1,shArr_acc_ct1.get_pointer());
                    });
            });
        }
        {
            queue.submit([&](sycl::handler &cgh) {
                sycl::accessor<double, 1, sycl::access::mode::read_write, sycl::access::target::local> shArr_acc_ct1(sycl::range<1>(blockSize), cgh);
                const int gridSize_ct2 = gridSize;
                cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, blockSize),sycl::range<3>(1, 1, blockSize)),
                    [=](sycl::nd_item<3> item_ct1) {
                        maxCommMultiBlock_double<blockSize>(dev_max_vVect, dev_max_vVect,gridSize_ct2, item_ct1,shArr_acc_ct1.get_pointer());
                    });
            });
        }

    queue.memcpy(&max_vVect, dev_max_vVect, sizeof(double)).wait();

    double alphaLoc=ZERO;
        {
            queue.submit([&](sycl::handler &cgh) {
                sycl::accessor<double, 1, sycl::access::mode::read_write, sycl::access::target::local> shArr_acc_ct1(sycl::range<1>(blockSize), cgh);

                cgh.parallel_for(
                    sycl::nd_range<3>(sycl::range<3>(1, 1, gridSize)*sycl::range<3>(1, 1, blockSize),sycl::range<3>(1, 1, blockSize)),
                    [=](sycl::nd_item<3> item_ct1) {
                        sumCommMultiBlock_double<blockSize>(vVect_dev, dev_ssq_vVect, max_vVect,nAstroElements * nAstroPSolved, item_ct1,shArr_acc_ct1.get_pointer());
                    });
            });
        }

        {
            queue.submit([&](sycl::handler &cgh) {
                sycl::accessor<double, 1, sycl::access::mode::read_write, sycl::access::target::local> shArr_acc_ct1(sycl::range<1>(blockSize), cgh);
                const int gridSize_ct2 = gridSize;
                cgh.parallel_for(
                    sycl::nd_range<3>(sycl::range<3>(1, 1, blockSize),sycl::range<3>(1, 1, blockSize)),
                    [=](sycl::nd_item<3> item_ct1) {
                        realsumCommMultiBlock_double<blockSize>(dev_ssq_vVect,dev_ssq_vVect,gridSize_ct2,item_ct1,shArr_acc_ct1.get_pointer());
                    });
            });
        }
    queue.memcpy(&ssq_vVect, dev_ssq_vVect, sizeof(double)).wait();
    alphaLoc = max_vVect*sqrt(ssq_vVect);
     
    alphaLoc2=alphaLoc*alphaLoc;
	if(myid==0) {
        double alphaOther2 = alphaLoc2;
            {
                queue.submit([&](sycl::handler &cgh) {
                    sycl::accessor<double, 1, sycl::access::mode::read_write, sycl::access::target::local> shArr_acc_ct1(sycl::range<1>(blockSize), cgh);
                    double *vVect_dev_localAstroMax_ct0=&vVect_dev[localAstroMax];
                    cgh.parallel_for(
                        sycl::nd_range<3>(sycl::range<3>(1, 1, gridSize)*sycl::range<3>(1, 1, blockSize),sycl::range<3>(1, 1, blockSize)),
                        [=](sycl::nd_item<3> item_ct1) {
                            sumCommMultiBlock_double<blockSize>(vVect_dev_localAstroMax_ct0, dev_ssq_vVect,max_vVect, nunkSplit - localAstroMax, item_ct1,shArr_acc_ct1.get_pointer());
                        });
                });
            }
            {
                queue.submit([&](sycl::handler &cgh) {
                    sycl::accessor<double, 1, sycl::access::mode::read_write, sycl::access::target::local> shArr_acc_ct1(sycl::range<1>(blockSize), cgh);
                    const int gridSize_ct2 = gridSize;
                    cgh.parallel_for(
                        sycl::nd_range<3>(sycl::range<3>(1, 1, blockSize),sycl::range<3>(1, 1, blockSize)),
                        [=](sycl::nd_item<3> item_ct1) {
                            realsumCommMultiBlock_double<blockSize>(dev_ssq_vVect,dev_ssq_vVect,gridSize_ct2,item_ct1, shArr_acc_ct1.get_pointer());
                        });
                });
            }
        
        queue.memcpy(&ssq_vVect, dev_ssq_vVect, sizeof(double)).wait();

        alphaLoc = max_vVect*sqrt(ssq_vVect);
        alphaLoc2 = alphaLoc*alphaLoc;
        alphaLoc2 = alphaOther2 + alphaLoc2;
	}

    #ifdef USE_MPI
        //------------------------------------------------------------------------------------------------  TIME6
        starttime=MPI_Wtime();
        MPI_Allreduce(MPI_IN_PLACE,&alphaLoc2,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
        communicationtime+=MPI_Wtime()-starttime;
        //------------------------------------------------------------------------------------------------
    #endif

    alpha=sqrt(alphaLoc2);
   }


    //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::: TIME7


    if (alpha > ZERO) 
    {
        {
            queue.parallel_for(
                sycl::nd_range<3>(sycl::range<3>(1, 1, BlockXGrid)*sycl::range<3>(1, 1, ThreadsXBlock),sycl::range<3>(1, 1, ThreadsXBlock)),
                [=](sycl::nd_item<3> item_ct1) {
                    dscal(vVect_dev, 1/alpha, nunkSplit, ONE, item_ct1);
                });
        }
        {
            queue.parallel_for(
                sycl::nd_range<3>(gridDim_nunk*sycl::range<3>(1, 1, TILE_WIDTH),sycl::range<3>(1, 1, TILE_WIDTH)),
                [=](sycl::nd_item<3> item_ct1) {
                    cblas_dcopy_kernel(nunkSplit, vVect_dev, wVect_dev,item_ct1);
                });
        }
    }



    queue.memcpy(vVect,vVect_dev,nunkSplit*sizeof(double)).wait();
    queue.memcpy(wVect,wVect_dev, nunkSplit * sizeof(double)).wait();
    queue.memcpy(knownTerms,knownTerms_dev,nElemKnownTerms*sizeof(double)).wait();

    arnorm  = alpha * beta;



    if (arnorm == ZERO){
        if (damped  &&  istop == 2) istop = 3;


        sycl::free(vVect_dev, queue);
        sycl::free(wVect_dev, queue);
        sycl::free(knownTerms_dev, queue);
        sycl::free(kAuxcopy_dev, queue);
        sycl::free(vAuxVect_dev, queue);
        sycl::free(instrCol_dev, queue);
        sycl::free(instrConstrIlung_dev, queue);
        sycl::free(dev_vVect_glob_sum, queue);
        sycl::free(dev_max_knownTerms, queue);
        sycl::free(dev_ssq_knownTerms, queue);
        sycl::free(xSolution_dev, queue);
        sycl::free(standardError_dev, queue);


        sycl::free(dev_max_vVect, queue);
        sycl::free(dev_ssq_vVect, queue);
        sycl::free(matrixIndexAstro_dev, queue);
        sycl::free(matrixIndexAtt_dev, queue);
        sycl::free(startend_dev, queue);

        if(nAstroPSolved)   sycl::free(sysmatAstro_dev, queue);
        if(nAttP)           sycl::free(sysmatAtt_dev, queue);
        if(nInstrPSolved)   sycl::free(sysmatInstr_dev, queue);    
        if(nGlobP)          sycl::free(sysmatGloB_dev, queue);
        if(nTotConstraints) sycl::free(sysmatConstr_dev, queue);

        free(startend);
        *istop_out  = istop;
        *itn_out    = itn;
        *anorm_out  = anorm;
        *acond_out  = acond;
        *rnorm_out  = rnorm;
        *arnorm_out = test2;
        *xnorm_out  = xnorm;
        
        return;
    }


    rhobar =   alpha;
    phibar =   beta;
    bnorm  =   beta;
    rnorm  =   beta;


    if(!myid){
        test1  = ONE;
        test2  = alpha / beta;
    }

    queue.memcpy(knownTerms_dev, knownTerms,nElemKnownTerms * sizeof(double)).wait();

    queue.memcpy(vVect_dev, vVect, nunkSplit * sizeof(double)).wait();

    queue.memcpy(wVect_dev, wVect, nunkSplit * sizeof(double)).wait();


    //  ==================================================================
    //  Main iteration loop.
    //  ==================================================================
    
    if (myid == 0) printf("LSQR: START ITERATIONS\n");
    ////////////////////////  START ITERATIONS

    
    #ifdef USE_MPI
        MPI_Bcast( &itnlim, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast( &comlsqr.itnLimit, 1, MPI_INT, 0, MPI_COMM_WORLD);
    #elif defined(KERNELTIME)
        float timekernel[6]={0.0,0.0,0.0,0.0,0.0,0.0};
    #else
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

        itn    = itn + 1;
        comlsqr.itn=itn;

        
        //      ------------------------------------------------------------------
        //      Perform the next step of the bidiagonalization to obtain the
        //      next  beta, u, alpha, v.  These satisfy the relations
        //                 beta*u  =  A*v  -  alpha*u,
        //                alpha*v  =  A(transpose)*u  -  beta*v.
        //      ------------------------------------------------------------------
        {
            queue.parallel_for(
                sycl::nd_range<3>(sycl::range<3>(1, 1, BlockXGrid)*sycl::range<3>(1, 1, ThreadsXBlock),sycl::range<3>(1, 1, ThreadsXBlock)),
                [=](sycl::nd_item<3> item_ct1) {
                    dscal(knownTerms_dev,alpha,nElemKnownTerms,MONE, item_ct1);
                });
        }

        {
            queue.parallel_for(
                sycl::nd_range<3>(gridDim_kAuxcopy_Kernel*sycl::range<3>(1, 1, TILE_WIDTH),sycl::range<3>(1, 1, TILE_WIDTH)),
                [=](sycl::nd_item<3> item_ct1) {
                    kAuxcopy_Kernel(knownTerms_dev, kAuxcopy_dev, mapNoss,nEqExtConstr+nEqBarConstr+nOfInstrConstr,item_ct1);
                });
        }
        //{ // CONTEXT MODE 1//////////////////////////////////// APROD MODE 1
        #ifdef KERNELTIME

                if(nAstroPSolved){

                    #ifdef PROF

                        auto event=queue.parallel_for<Astro1Kernel>(
                            sycl::nd_range<3>(gridDim_aprod1astro*sycl::range<3>(1, 1, ThreadsXBlockAprod1Astro),sycl::range<3>(1, 1, ThreadsXBlockAprod1Astro)),
                            [=](sycl::nd_item<3> item_ct1) {
                                aprod1_Kernel_astro(knownTerms_dev, sysmatAstro_dev,vVect_dev, matrixIndexAstro_dev,mapNoss, offLocalAstro, nAstroPSolved,item_ct1);
                            });
                
                        event.wait();


                    #else

                        auto event=queue.parallel_for(
                            sycl::nd_range<3>(gridDim_aprod1astro*sycl::range<3>(1, 1, ThreadsXBlockAprod1Astro),sycl::range<3>(1, 1, ThreadsXBlockAprod1Astro)),
                            [=](sycl::nd_item<3> item_ct1) {
                                aprod1_Kernel_astro(knownTerms_dev, sysmatAstro_dev,vVect_dev, matrixIndexAstro_dev,mapNoss, offLocalAstro, nAstroPSolved,item_ct1);
                            });
                
                        event.wait();

                    #endif

                    auto start = event.get_profiling_info<sycl::info::event_profiling::command_start>();
                    auto end = event.get_profiling_info<sycl::info::event_profiling::command_end>();
                    timekernel[0]+=end-start;

                }

                if(nAttP){ 

                    #ifdef PROF

                        auto event=queue.parallel_for<Att1Kernel>(
                            sycl::nd_range<3>(gridDim_aprod1attinstr*sycl::range<3>(1, 1, ThreadsXBlockAprod1AttInstr),sycl::range<3>(1, 1, ThreadsXBlockAprod1AttInstr)),
                            [=](sycl::nd_item<3> item_ct1) {
                                aprod1_Kernel_att_AttAxis(knownTerms_dev, sysmatAtt_dev, vVect_dev,matrixIndexAtt_dev, nAttP, mapNoss, nDegFreedomAtt,offLocalAtt, nAttParAxis, item_ct1);
                            });

                        event.wait();


                    #else

                        auto event=queue.parallel_for(
                            sycl::nd_range<3>(gridDim_aprod1attinstr*sycl::range<3>(1, 1, ThreadsXBlockAprod1AttInstr),sycl::range<3>(1, 1, ThreadsXBlockAprod1AttInstr)),
                            [=](sycl::nd_item<3> item_ct1) {
                                aprod1_Kernel_att_AttAxis(knownTerms_dev, sysmatAtt_dev, vVect_dev,matrixIndexAtt_dev, nAttP, mapNoss, nDegFreedomAtt,offLocalAtt, nAttParAxis, item_ct1);
                            });

                        event.wait();

                    #endif

                    auto start = event.get_profiling_info<sycl::info::event_profiling::command_start>();
                    auto end = event.get_profiling_info<sycl::info::event_profiling::command_end>();
                    timekernel[1]+=end-start;

                }

            //    APROD1 INSTR CALL
            {
                if(nInstrPSolved){

                    #ifdef PROF

                        auto event=queue.parallel_for<Instr1kernel>(
                            sycl::nd_range<3>(gridDim_aprod1attinstr *sycl::range<3>(1, 1, ThreadsXBlockAprod1AttInstr),sycl::range<3>(1, 1, ThreadsXBlockAprod1AttInstr)),
                                [=](sycl::nd_item<3> item_ct1) {
                                aprod1_Kernel_instr(knownTerms_dev, sysmatInstr_dev,vVect_dev, instrCol_dev, mapNoss,offLocalInstr, nInstrPSolved, item_ct1);
                            });

                        event.wait();

                    #else

                        auto event=queue.parallel_for(
                            sycl::nd_range<3>(gridDim_aprod1attinstr *sycl::range<3>(1, 1, ThreadsXBlockAprod1AttInstr),sycl::range<3>(1, 1, ThreadsXBlockAprod1AttInstr)),
                                [=](sycl::nd_item<3> item_ct1) {
                                aprod1_Kernel_instr(knownTerms_dev, sysmatInstr_dev,vVect_dev, instrCol_dev, mapNoss,offLocalInstr, nInstrPSolved, item_ct1);
                            });

                        event.wait();

                    #endif

                    auto start = event.get_profiling_info<sycl::info::event_profiling::command_start>();
                    auto end = event.get_profiling_info<sycl::info::event_profiling::command_end>();
                    timekernel[2]+=end-start;
                }
            }
            //    APROD1 GLOB CALL
            {
                if(nGlobP){
                    queue.parallel_for(
                        sycl::nd_range<3>(gridDim_aprod1*sycl::range<3>(1, 1, TILE_WIDTH),sycl::range<3>(1, 1, TILE_WIDTH)),
                            [=](sycl::nd_item<3> item_ct1) {
                            aprod1_Kernel_glob(knownTerms_dev, sysmatGloB_dev,vVect_dev, offLocalGlob, mapNoss, nGlobP,item_ct1);
                        });
                }
            }
            //        CONSTRAINTS APROD MODE 1        
            /* ExtConstr */
            {
                if(nEqExtConstr){
                    queue.parallel_for(
                        sycl::nd_range<3>(gridDim_aprod1_ExtConstr*sycl::range<3>(1, 1, TILE_WIDTH),sycl::range<3>(1, 1, TILE_WIDTH)),
                        [=](sycl::nd_item<3> item_ct1) {
                            aprod1_Kernel_ExtConstr(knownTerms_dev,sysmatConstr_dev,vVect_dev,
                            VrIdAstroPDimMax,mapNoss,nDegFreedomAtt,startingAttColExtConstr,nEqExtConstr,
                            nOfElextObs,numOfExtStar,numOfExtAttCol,nAstroPSolved,nAttAxes, item_ct1);
                        });
                }
            }
            /* BarConstr */
            {
                if(nEqBarConstr){
                    queue.parallel_for(
                        sycl::nd_range<3>(gridDim_aprod1_BarConstr*sycl::range<3>(1, 1, TILE_WIDTH),sycl::range<3>(1, 1, TILE_WIDTH)),
                        [=](sycl::nd_item<3> item_ct1) {
                            aprod1_Kernel_BarConstr(knownTerms_dev,sysmatConstr_dev,vVect_dev,nOfElextObs,
                            nOfElBarObs,nEqExtConstr,mapNoss,nEqBarConstr,numOfBarStar,
                            nAstroPSolved, item_ct1);
                        });
                }
            }
            // /* InstrConstr */
            {
                if(nOfInstrConstr){
                    queue.parallel_for(
                        sycl::nd_range<3>(gridDim_aprod1_InstrConstr*sycl::range<3>(1, 1, TILE_WIDTH),sycl::range<3>(1, 1, TILE_WIDTH)),
                        [=](sycl::nd_item<3> item_ct1) {
                                aprod1_Kernel_InstrConstr(knownTerms_dev,sysmatConstr_dev,vVect_dev,
                                instrConstrIlung_dev,instrCol_dev,VrIdAstroPDimMax,mapNoss, 
                                nDegFreedomAtt,nOfElextObs,nEqExtConstr,nOfElBarObs,nEqBarConstr, 
                                myid,nOfInstrConstr,nproc,nAstroPSolved,nAttAxes,nInstrPSolved, item_ct1);
                        });
                }
            }


        #else


            //    APROD1 ASTRO CALL
            {
                if(nAstroPSolved){
                    queue.parallel_for(
                        sycl::nd_range<3>(gridDim_aprod1astro*sycl::range<3>(1, 1, ThreadsXBlockAprod1Astro),sycl::range<3>(1, 1, ThreadsXBlockAprod1Astro)),
                        [=](sycl::nd_item<3> item_ct1) {
                            aprod1_Kernel_astro(knownTerms_dev, sysmatAstro_dev,vVect_dev, matrixIndexAstro_dev,mapNoss, offLocalAstro, nAstroPSolved,item_ct1);
                        });
                }
            }
            //    APROD1 ATT CALL
            {
                if(nAttP){ 
                    queue.parallel_for(     
                        sycl::nd_range<3>(gridDim_aprod1attinstr*sycl::range<3>(1, 1, ThreadsXBlockAprod1AttInstr),sycl::range<3>(1, 1, ThreadsXBlockAprod1AttInstr)),
                        [=](sycl::nd_item<3> item_ct1) {
                            aprod1_Kernel_att_AttAxis(knownTerms_dev, sysmatAtt_dev, vVect_dev,matrixIndexAtt_dev, nAttP, mapNoss, nDegFreedomAtt,offLocalAtt, nAttParAxis, item_ct1);
                        });
                }
            }

            //    APROD1 INSTR CALL
            {
                if(nInstrPSolved){
                    queue.parallel_for(
                        sycl::nd_range<3>(gridDim_aprod1attinstr *sycl::range<3>(1, 1, ThreadsXBlockAprod1AttInstr),sycl::range<3>(1, 1, ThreadsXBlockAprod1AttInstr)),
                            [=](sycl::nd_item<3> item_ct1) {
                            aprod1_Kernel_instr(knownTerms_dev, sysmatInstr_dev,vVect_dev, instrCol_dev, mapNoss,offLocalInstr, nInstrPSolved, item_ct1);
                        });
                }
            }
            //    APROD1 GLOB CALL
            {
                if(nGlobP){
                    queue.parallel_for(
                        sycl::nd_range<3>(gridDim_aprod1*sycl::range<3>(1, 1, TILE_WIDTH),sycl::range<3>(1, 1, TILE_WIDTH)),
                            [=](sycl::nd_item<3> item_ct1) {
                            aprod1_Kernel_glob(knownTerms_dev, sysmatGloB_dev,vVect_dev, offLocalGlob, mapNoss, nGlobP,item_ct1);
                        });
                }
            }

            //        CONSTRAINTS APROD MODE 1        
            /* ExtConstr */

            {
                if(nEqExtConstr){
                    queue.parallel_for(
                        sycl::nd_range<3>(gridDim_aprod1_ExtConstr*sycl::range<3>(1, 1, TILE_WIDTH),sycl::range<3>(1, 1, TILE_WIDTH)),
                        [=](sycl::nd_item<3> item_ct1) {
                            aprod1_Kernel_ExtConstr(knownTerms_dev,sysmatConstr_dev,vVect_dev,
                            VrIdAstroPDimMax,mapNoss,nDegFreedomAtt,startingAttColExtConstr,nEqExtConstr,
                            nOfElextObs,numOfExtStar,numOfExtAttCol,nAstroPSolved,nAttAxes, item_ct1);
                        });
                }
            }
            /* BarConstr */

            {
                if(nEqBarConstr){
                    queue.parallel_for(
                        sycl::nd_range<3>(gridDim_aprod1_BarConstr*sycl::range<3>(1, 1, TILE_WIDTH),sycl::range<3>(1, 1, TILE_WIDTH)),
                        [=](sycl::nd_item<3> item_ct1) {
                            aprod1_Kernel_BarConstr(knownTerms_dev,sysmatConstr_dev,vVect_dev,nOfElextObs,
                            nOfElBarObs,nEqExtConstr,mapNoss,nEqBarConstr,numOfBarStar,
                            nAstroPSolved, item_ct1);
                        });
                }
            }
            // /* InstrConstr */
            {
                if(nOfInstrConstr){
                    queue.parallel_for(
                        sycl::nd_range<3>(gridDim_aprod1_InstrConstr*sycl::range<3>(1, 1, TILE_WIDTH),sycl::range<3>(1, 1, TILE_WIDTH)),
                        [=](sycl::nd_item<3> item_ct1) {
                                aprod1_Kernel_InstrConstr(knownTerms_dev,sysmatConstr_dev,vVect_dev,
                                instrConstrIlung_dev,instrCol_dev,VrIdAstroPDimMax,mapNoss, 
                                nDegFreedomAtt,nOfElextObs,nEqExtConstr,nOfElBarObs,nEqBarConstr, 
                                myid,nOfInstrConstr,nproc,nAstroPSolved,nAttAxes,nInstrPSolved, item_ct1);
                        });
                }
            }

        #endif




        #ifdef USE_MPI
            //}// END CONTEXT MODE 1
            queue.memcpy(&knownTerms[mapNoss], &knownTerms_dev[mapNoss],(nEqExtConstr+nEqBarConstr+nOfInstrConstr)*sizeof(double)).wait();
            //------------------------------------------------------------------------------------------------
            starttime=MPI_Wtime();
            MPI_Allreduce(MPI_IN_PLACE,&knownTerms[mapNoss],nEqExtConstr+nEqBarConstr+nOfInstrConstr,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
            communicationtime+=MPI_Wtime()-starttime;
            //------------------------------------------------------------------------------------------------
            queue.memcpy(&knownTerms_dev[mapNoss], &knownTerms[mapNoss],(nEqExtConstr+nEqBarConstr+nOfInstrConstr)*sizeof(double)).wait();
        #endif

        //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        {
            queue.submit([&](sycl::handler &cgh){double *knownTerms_dev_mapNoss_ct0 = &knownTerms_dev[mapNoss];

                cgh.parallel_for(
                    sycl::nd_range<3>(gridDim_kAuxcopy_Kernel*sycl::range<3>(1, 1, TILE_WIDTH),sycl::range<3>(1, 1, TILE_WIDTH)),
                    [=](sycl::nd_item<3> item_ct1) {
                        kauxsum(knownTerms_dev_mapNoss_ct0, kAuxcopy_dev,nEqExtConstr + nEqBarConstr + nOfInstrConstr,item_ct1);
                    });
            });
        }
        {

            queue.submit([&](sycl::handler &cgh) {

                sycl::accessor<double, 1, sycl::access::mode::read_write, sycl::access::target::local> shArr_acc_ct1(sycl::range<1>(blockSize), cgh);

                cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, gridSize)*sycl::range<3>(1, 1, blockSize),sycl::range<3>(1, 1, blockSize)),
                    [=](sycl::nd_item<3> item_ct1) {
                        maxCommMultiBlock_double<blockSize>(knownTerms_dev, dev_max_knownTerms, nElemKnownTerms,item_ct1, shArr_acc_ct1.get_pointer());
                    });
            });
        }
        {

            queue.submit([&](sycl::handler &cgh) {
                sycl::accessor<double, 1, sycl::access::mode::read_write, sycl::access::target::local> shArr_acc_ct1(sycl::range<1>(blockSize), cgh);
                const int gridSize_ct2 = gridSize;
                cgh.parallel_for(
                    sycl::nd_range<3>(sycl::range<3>(1, 1, blockSize),sycl::range<3>(1, 1, blockSize)),
                    [=](sycl::nd_item<3> item_ct1) {
                        maxCommMultiBlock_double<blockSize>(dev_max_knownTerms,dev_max_knownTerms,gridSize_ct2, item_ct1,shArr_acc_ct1.get_pointer());
                    });
            });
        }
        queue.memcpy(&max_knownTerms, dev_max_knownTerms, sizeof(double)).wait();



        if(myid==0)
        {
            {

                queue.submit([&](sycl::handler &cgh) {
                    
                    sycl::accessor<double, 1, sycl::access::mode::read_write, sycl::access::target::local> shArr_acc_ct1(sycl::range<1>(blockSize), cgh);

                    cgh.parallel_for(
                        sycl::nd_range<3>(sycl::range<3>(1, 1, gridSize)*sycl::range<3>(1, 1, blockSize),sycl::range<3>(1, 1, blockSize)),
                        [=](sycl::nd_item<3> item_ct1) {
                            sumCommMultiBlock_double<blockSize>(knownTerms_dev, dev_ssq_knownTerms,max_knownTerms,nElemKnownTerms,item_ct1, shArr_acc_ct1.get_pointer());
                        });
                });
            }
            {

                queue.submit([&](sycl::handler &cgh) {
                    sycl::accessor<double, 1, sycl::access::mode::read_write, sycl::access::target::local> shArr_acc_ct1(sycl::range<1>(blockSize), cgh);
                    const int gridSize_ct2 = gridSize;
                    cgh.parallel_for(
                        sycl::nd_range<3>(sycl::range<3>(1, 1, blockSize),sycl::range<3>(1, 1, blockSize)),
                        [=](sycl::nd_item<3> item_ct1) {
                            realsumCommMultiBlock_double<blockSize>(dev_ssq_knownTerms, dev_ssq_knownTerms,gridSize_ct2, item_ct1,shArr_acc_ct1.get_pointer());
                        });
                });
            }
            queue.memcpy(&ssq_knownTerms, dev_ssq_knownTerms, sizeof(double)).wait();
            betaLoc = max_knownTerms*sqrt(ssq_knownTerms);
            betaLoc2 = betaLoc*betaLoc;
        }
        else
        {
            {

                queue.submit([&](sycl::handler &cgh) {
                    

                    sycl::accessor<double, 1, sycl::access::mode::read_write, sycl::access::target::local> shArr_acc_ct1(sycl::range<1>(blockSize), cgh);

                    cgh.parallel_for(
                        sycl::nd_range<3>(sycl::range<3>(1, 1, gridSize)*sycl::range<3>(1, 1, blockSize),sycl::range<3>(1, 1, blockSize)),
                        [=](sycl::nd_item<3> item_ct1) {
                            sumCommMultiBlock_double<blockSize>(knownTerms_dev, dev_ssq_knownTerms,max_knownTerms, mapNoss, item_ct1,shArr_acc_ct1.get_pointer());
                        });
                });
            }
            {

                queue.submit([&](sycl::handler &cgh) {
                    
                    sycl::accessor<double, 1, sycl::access::mode::read_write, sycl::access::target::local> shArr_acc_ct1(sycl::range<1>(blockSize), cgh);

                    const int gridSize_ct2 = gridSize;

                    cgh.parallel_for(
                        sycl::nd_range<3>(sycl::range<3>(1, 1, blockSize),sycl::range<3>(1, 1, blockSize)),
                        [=](sycl::nd_item<3> item_ct1) {
                            realsumCommMultiBlock_double<blockSize>(dev_ssq_knownTerms, dev_ssq_knownTerms,gridSize_ct2, item_ct1,shArr_acc_ct1.get_pointer());
                        });
                });
            }
            queue.memcpy(&ssq_knownTerms, dev_ssq_knownTerms, sizeof(double)).wait();
            betaLoc = max_knownTerms*sqrt(ssq_knownTerms);
            betaLoc2 = betaLoc*betaLoc;
        }
            

        #ifdef USE_MPI
            //------------------------------------------------------------------------------------------------
            starttime=MPI_Wtime();
                MPI_Allreduce(MPI_IN_PLACE,&betaLoc2,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
            communicationtime+=MPI_Wtime()-starttime;
            //------------------------------------------------------------------------------------------------
            //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        #endif

        beta=sqrt(betaLoc2);
            
        //  Accumulate  anorm = || Bk || =  sqrt( sum of  alpha**2 + beta**2 + damp**2 ).

        temp   =   d2norm( alpha, beta );
        temp   =   d2norm( temp , damp );
        anorm  =   d2norm( anorm, temp );

        if (beta > ZERO) {

            {

                queue.parallel_for(
                    sycl::nd_range<3>(sycl::range<3>(1, 1, BlockXGrid)*sycl::range<3>(1, 1, ThreadsXBlock),sycl::range<3>(1, 1, ThreadsXBlock)),
                    [=](sycl::nd_item<3> item_ct1) {
                        dscal(knownTerms_dev, 1/beta,nElemKnownTerms,ONE,item_ct1);
                    });
            }
            {

                queue.parallel_for(
                    sycl::nd_range<3>(sycl::range<3>(1, 1, BlockXGrid)*sycl::range<3>(1, 1, ThreadsXBlock),sycl::range<3>(1, 1, ThreadsXBlock)),
                    [=](sycl::nd_item<3> item_ct1) {
                        dscal(vVect_dev,beta,nunkSplit,MONE,item_ct1);
                    });
            }

            {

                queue.parallel_for(
                    sycl::nd_range<3>(gridDim_vAuxVect_Kernel*sycl::range<3>(1, 1, TILE_WIDTH),sycl::range<3>(1, 1, TILE_WIDTH)),
                    [=](sycl::nd_item<3> item_ct1) {
                        vAuxVect_Kernel(vVect_dev, vAuxVect_dev, localAstroMax,item_ct1);
                    });
            }

            if (myid != 0) {


                queue.parallel_for(
                    sycl::nd_range<3>(gridDim_vVect_Put_To_Zero_Kernel*sycl::range<3>(1, 1, TILE_WIDTH),sycl::range<3>(1, 1, TILE_WIDTH)),
                    [=](sycl::nd_item<3> item_ct1) {
                        vVect_Put_To_Zero_Kernel(vVect_dev, localAstroMax,  nunkSplit, item_ct1);
                    });
            }
            
            queue.wait_and_throw();


	    
            //{ // CONTEXT MODE 2 //////////////////////////////////// APROD MODE 2
            #ifdef KERNELTIME

                    if (nAstroPSolved) {

                        #ifdef PROF

                            auto event=queue_Aprod2_0.submit([&](sycl::handler &cgh) {
                                sycl::local_accessor<double, 3> sharedSum_acc_ct1(sycl::range<3>(RowsPerBlock, ThreadsXBlockAprod2Astro, 5),cgh);

                                cgh.parallel_for<Astro2Kernel>(
                                    sycl::nd_range<3>(gridAprod2astro*blockAprod2astro,blockAprod2astro),
                                    [=](sycl::nd_item<3> item_ct1) {
                                        aprod2_Kernel_astro<RowsPerBlock,ThreadsXBlockAprod2Astro>(
                                            vVect_dev, sysmatAstro_dev, knownTerms_dev,matrixIndexAstro_dev, startend_dev, offLocalAstro,nnz, nAstroPSolved, item_ct1, sharedSum_acc_ct1);
                                    });
                            });

                            event.wait();

                        #else


                            auto event=queue_Aprod2_0.submit([&](sycl::handler &cgh) {
                                sycl::local_accessor<double, 3> sharedSum_acc_ct1(sycl::range<3>(RowsPerBlock, ThreadsXBlockAprod2Astro, 5),cgh);

                                cgh.parallel_for(
                                    sycl::nd_range<3>(gridAprod2astro*blockAprod2astro,blockAprod2astro),
                                    [=](sycl::nd_item<3> item_ct1) {
                                        aprod2_Kernel_astro<RowsPerBlock,ThreadsXBlockAprod2Astro>(
                                            vVect_dev, sysmatAstro_dev, knownTerms_dev,matrixIndexAstro_dev, startend_dev, offLocalAstro,nnz, nAstroPSolved, item_ct1, sharedSum_acc_ct1);
                                    });
                            });

                            event.wait();



                        #endif

                        auto start = event.get_profiling_info<sycl::info::event_profiling::command_start>();
                        auto end = event.get_profiling_info<sycl::info::event_profiling::command_end>();
                        timekernel[3]+=end-start;

                    }

                    if(nAttP){

                        #ifdef PROF

                            auto event=queue_Aprod2_1.parallel_for<Att2Kernel>(sycl::nd_range<3>(gridDim_aprod2attinstr*sycl::range<3>(1, 1, ThreadsXBlockAprod2AttInstr),sycl::range<3>(1, 1, ThreadsXBlockAprod2AttInstr)),
                                [=](sycl::nd_item<3> item_ct1) {
                                    aprod2_Kernel_att_AttAxis(vVect_dev, sysmatAtt_dev, knownTerms_dev,matrixIndexAtt_dev, nAttP, nDegFreedomAtt, offLocalAtt,mapNoss, nAstroPSolved, nAttParAxis, item_ct1);
                                });

                            event.wait();


                        #else

                            auto event=queue_Aprod2_1.parallel_for(sycl::nd_range<3>(gridDim_aprod2attinstr*sycl::range<3>(1, 1, ThreadsXBlockAprod2AttInstr),sycl::range<3>(1, 1, ThreadsXBlockAprod2AttInstr)),
                                [=](sycl::nd_item<3> item_ct1) {
                                    aprod2_Kernel_att_AttAxis(vVect_dev, sysmatAtt_dev, knownTerms_dev,matrixIndexAtt_dev, nAttP, nDegFreedomAtt, offLocalAtt,mapNoss, nAstroPSolved, nAttParAxis, item_ct1);
                                });

                            event.wait();

                        #endif

                        auto start = event.get_profiling_info<sycl::info::event_profiling::command_start>();
                        auto end = event.get_profiling_info<sycl::info::event_profiling::command_end>();
                        timekernel[4]+=end-start;

                    }

                    if(nInstrPSolved){

                        #ifdef PROF

                            auto event=queue_Aprod2_2.parallel_for<Instr2kernel>(sycl::nd_range<3>(gridDim_aprod2attinstr*sycl::range<3>(1, 1, ThreadsXBlockAprod2AttInstr), sycl::range<3>(1, 1, ThreadsXBlockAprod2AttInstr)),
                                [=](sycl::nd_item<3> item_ct1) {
                                    aprod2_Kernel_instr(vVect_dev, sysmatInstr_dev,knownTerms_dev, instrCol_dev,offLocalInstr, mapNoss, nInstrPSolved,item_ct1);
                                });

                            event.wait();



                        #else

                            auto event=queue_Aprod2_2.parallel_for(sycl::nd_range<3>(gridDim_aprod2attinstr*sycl::range<3>(1, 1, ThreadsXBlockAprod2AttInstr), sycl::range<3>(1, 1, ThreadsXBlockAprod2AttInstr)),
                                [=](sycl::nd_item<3> item_ct1) {
                                    aprod2_Kernel_instr(vVect_dev, sysmatInstr_dev,knownTerms_dev, instrCol_dev,offLocalInstr, mapNoss, nInstrPSolved,item_ct1);
                                });

                            event.wait();

                        #endif

                        auto start = event.get_profiling_info<sycl::info::event_profiling::command_start>();
                        auto end = event.get_profiling_info<sycl::info::event_profiling::command_end>();
                        timekernel[5]+=end-start;

                    }

                for (short inGlob = 0; inGlob < nGlobP; inGlob++)
                {
                        {
                            queue_Aprod2_3.submit([&](sycl::handler &cgh) {
                                sycl::accessor<double, 1, sycl::access::mode::read_write, sycl::access::target::local> shArr_acc_ct1(sycl::range<1>(blockSize), cgh);
                                cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, gridSize)*sycl::range<3>(1, 1, blockSize),sycl::range<3>(1, 1, blockSize)),
                                    [=](sycl::nd_item<3> item_ct1) {
                                        sumCommMultiBlock_double_aprod2_Kernel_glob(dev_vVect_glob_sum, sysmatGloB_dev,knownTerms_dev, vVect_dev, nGlobP, mapNoss,offLocalGlob, inGlob, item_ct1,shArr_acc_ct1.get_pointer());
                                    });
                            });
                        }
                        {
                            queue_Aprod2_4.submit([&](sycl::handler &cgh) {
                                sycl::accessor<double, 1, sycl::access::mode::read_write, sycl::access::target::local> shArr_acc_ct1(sycl::range<1>(blockSize), cgh);
                                const long gridSize_ct2 = gridSize;
                                cgh.parallel_for(
                                    sycl::nd_range<3>(sycl::range<3>(1, 1, blockSize),sycl::range<3>(1, 1, blockSize)),
                                    [=](sycl::nd_item<3> item_ct1) {
                                        realsumCommMultiBlock_double_aprod2_Kernel_glob(vVect_dev, dev_vVect_glob_sum, gridSize_ct2,offLocalGlob, inGlob, item_ct1,shArr_acc_ct1.get_pointer());
                                    });
                            });
                        }
                }


            #else

                //APROD2 CALL BEFORE LSQR
                {

                    if (nAstroPSolved) {
                        queue_Aprod2_0.submit([&](sycl::handler &cgh) {
                            sycl::local_accessor<double, 3> sharedSum_acc_ct1(sycl::range<3>(RowsPerBlock, ThreadsXBlockAprod2Astro, 5),cgh);

                            cgh.parallel_for(
                                sycl::nd_range<3>(gridAprod2astro*blockAprod2astro,blockAprod2astro),
                                [=](sycl::nd_item<3> item_ct1) {
                                    aprod2_Kernel_astro<RowsPerBlock,ThreadsXBlockAprod2Astro>(
                                        vVect_dev, sysmatAstro_dev, knownTerms_dev,matrixIndexAstro_dev, startend_dev, offLocalAstro,nnz, nAstroPSolved, item_ct1, sharedSum_acc_ct1);
                                });
                        });
                    }

                }

                {
                    if(nAttP){
                        queue_Aprod2_1.parallel_for(sycl::nd_range<3>(gridDim_aprod2attinstr*sycl::range<3>(1, 1, ThreadsXBlockAprod2AttInstr), sycl::range<3>(1, 1, ThreadsXBlockAprod2AttInstr)),
                            [=](sycl::nd_item<3> item_ct1) {
                                aprod2_Kernel_att_AttAxis(vVect_dev, sysmatAtt_dev, knownTerms_dev,matrixIndexAtt_dev, nAttP, nDegFreedomAtt, offLocalAtt,mapNoss, nAstroPSolved, nAttParAxis, item_ct1);
                            });
                    }
                }

                {
                    if(nInstrPSolved){
                        queue_Aprod2_2.parallel_for(sycl::nd_range<3>(gridDim_aprod2attinstr*sycl::range<3>(1, 1, ThreadsXBlockAprod2AttInstr),sycl::range<3>(1, 1, ThreadsXBlockAprod2AttInstr)),
                            [=](sycl::nd_item<3> item_ct1) {
                                aprod2_Kernel_instr(vVect_dev, sysmatInstr_dev,knownTerms_dev, instrCol_dev,offLocalInstr, mapNoss, nInstrPSolved,item_ct1);
                            });
                    }
                }

                for (short inGlob = 0; inGlob < nGlobP; inGlob++)
                {
                        {
                            queue_Aprod2_3.submit([&](sycl::handler &cgh) {
                                sycl::accessor<double, 1, sycl::access::mode::read_write, sycl::access::target::local> shArr_acc_ct1(sycl::range<1>(blockSize), cgh);
                                cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, gridSize)*sycl::range<3>(1, 1, blockSize),sycl::range<3>(1, 1, blockSize)),
                                    [=](sycl::nd_item<3> item_ct1) {
                                        sumCommMultiBlock_double_aprod2_Kernel_glob(dev_vVect_glob_sum, sysmatGloB_dev,knownTerms_dev, vVect_dev, nGlobP, mapNoss,offLocalGlob, inGlob, item_ct1,shArr_acc_ct1.get_pointer());
                                    });
                            });
                        }
                        {
                            queue_Aprod2_4.submit([&](sycl::handler &cgh) {
                                sycl::accessor<double, 1, sycl::access::mode::read_write, sycl::access::target::local> shArr_acc_ct1(sycl::range<1>(blockSize), cgh);
                                const long gridSize_ct2 = gridSize;
                                cgh.parallel_for(
                                    sycl::nd_range<3>(sycl::range<3>(1, 1, blockSize),sycl::range<3>(1, 1, blockSize)),
                                    [=](sycl::nd_item<3> item_ct1) {
                                        realsumCommMultiBlock_double_aprod2_Kernel_glob(vVect_dev, dev_vVect_glob_sum, gridSize_ct2,offLocalGlob, inGlob, item_ct1,shArr_acc_ct1.get_pointer());
                                    });
                            });
                        }
                }

            #endif




            queue_Aprod2_0.wait_and_throw();
            queue_Aprod2_1.wait_and_throw();
            queue_Aprod2_2.wait_and_throw();
            queue_Aprod2_3.wait_and_throw();
            queue_Aprod2_4.wait_and_throw();


            //  CONSTRAINTS OF APROD MODE 2:
            {
                if(nEqExtConstr){
                    queue_Aprod2_5.parallel_for(
                        sycl::nd_range<3>(gridDim_aprod2_ExtConstr*sycl::range<3>(1, 1, TILE_WIDTH),sycl::range<3>(1, 1, TILE_WIDTH)),
                        [=](sycl::nd_item<3> item_ct1) {
                                aprod2_Kernel_ExtConstr(vVect_dev,sysmatConstr_dev,knownTerms_dev,mapNoss,nDegFreedomAtt,VrIdAstroPDimMax,nEqExtConstr,nOfElextObs,numOfExtStar, 
                                startingAttColExtConstr,numOfExtAttCol,nAttAxes, nAstroPSolved,item_ct1);
                        });
                }
            }
            {
                if(nEqBarConstr){
                    queue_Aprod2_6.parallel_for(
                        sycl::nd_range<3>(gridDim_aprod2_BarConstr*sycl::range<3>(1, 1, TILE_WIDTH),sycl::range<3>(1, 1, TILE_WIDTH)),
                        [=](sycl::nd_item<3> item_ct1) {
                                aprod2_Kernel_BarConstr(vVect_dev,sysmatConstr_dev, knownTerms_dev,mapNoss,nEqBarConstr,nEqExtConstr, 
                                nOfElextObs,nOfElBarObs,numOfBarStar, nAstroPSolved,item_ct1);
                        });
                }
            }
            {
                if(nOfInstrConstr){
                    queue_Aprod2_7.parallel_for(
                        sycl::nd_range<3>(gridDim_aprod2_InstrConstr *sycl::range<3>(1, 1, TILE_WIDTH),sycl::range<3>(1, 1, TILE_WIDTH)),
                        [=](sycl::nd_item<3> item_ct1) {
                                aprod2_Kernel_InstrConstr(vVect_dev,sysmatConstr_dev,knownTerms_dev,instrConstrIlung_dev,instrCol_dev,VrIdAstroPDimMax,nDegFreedomAtt,mapNoss,nEqExtConstr, 
                                nEqBarConstr,nOfElextObs,nOfElBarObs,myid,nOfInstrConstr,nproc, nAstroPSolved,nAttAxes,nInstrPSolved, item_ct1);   
                        });
                }
            }

            queue_Aprod2_5.wait_and_throw();
            queue_Aprod2_6.wait_and_throw();
            queue_Aprod2_7.wait_and_throw();


            #ifdef USE_MPI
                queue.memcpy(vVect, vVect_dev, nunkSplit * sizeof(double)).wait();
                //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
                //------------------------------------------------------------------------------------------------
                starttime=MPI_Wtime();
                MPI_Allreduce(MPI_IN_PLACE,&vVect[localAstroMax], nAttParam+nInstrParam+nGlobalParam,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);  
                communicationtime+=MPI_Wtime()-starttime;
                //------------------------------------------------------------------------------------------------
                if(nAstroPSolved) SumCirc(vVect,comlsqr);
                //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
                queue.memcpy(vVect_dev, vVect, nunkSplit * sizeof(double)).wait();
            #endif

            {
                queue.parallel_for(sycl::nd_range<3>(gridDim_vAuxVect_Kernel *sycl::range<3>(1, 1, TILE_WIDTH),sycl::range<3>(1, 1, TILE_WIDTH)),
                    [=](sycl::nd_item<3> item_ct1) {
                        vaux_sum(vVect_dev, vAuxVect_dev, localAstroMax,item_ct1);
                    });
            }

            {
                queue.submit([&](sycl::handler &cgh) {
                    sycl::accessor<double, 1, sycl::access::mode::read_write, sycl::access::target::local> shArr_acc_ct1(sycl::range<1>(blockSize), cgh);
                    cgh.parallel_for(
                        sycl::nd_range<3>(sycl::range<3>(1, 1, gridSize)*sycl::range<3>(1, 1, blockSize),sycl::range<3>(1, 1, blockSize)),
                        [=](sycl::nd_item<3> item_ct1) {
                            maxCommMultiBlock_double<blockSize>(vVect_dev, dev_max_vVect, nunkSplit, item_ct1,shArr_acc_ct1.get_pointer());
                        });
                });
            }
            {
                queue.submit([&](sycl::handler &cgh) {
                    sycl::accessor<double, 1, sycl::access::mode::read_write, sycl::access::target::local> shArr_acc_ct1(sycl::range<1>(blockSize), cgh);
                    const int gridSize_ct2 = gridSize;
                    cgh.parallel_for(
                        sycl::nd_range<3>(sycl::range<3>(1, 1, blockSize),sycl::range<3>(1, 1, blockSize)),
                        [=](sycl::nd_item<3> item_ct1) {
                            maxCommMultiBlock_double<blockSize>(dev_max_vVect, dev_max_vVect, gridSize_ct2,item_ct1, shArr_acc_ct1.get_pointer());
                        });
                });
            }
            queue.memcpy(&max_vVect, dev_max_vVect, sizeof(double)).wait();
            {
                queue.submit([&](sycl::handler &cgh) {
                    sycl::accessor<double, 1, sycl::access::mode::read_write, sycl::access::target::local> shArr_acc_ct1(sycl::range<1>(blockSize), cgh);
                    cgh.parallel_for(
                        sycl::nd_range<3>(sycl::range<3>(1, 1, gridSize)*sycl::range<3>(1, 1, blockSize),sycl::range<3>(1, 1, blockSize)),
                        [=](sycl::nd_item<3> item_ct1) {
                            sumCommMultiBlock_double<blockSize>(vVect_dev, dev_ssq_vVect, max_vVect,nAstroElements * nAstroPSolved, item_ct1,shArr_acc_ct1.get_pointer());
                        });
                });
            }
            {
                queue.submit([&](sycl::handler &cgh) {
                    sycl::accessor<double, 1, sycl::access::mode::read_write, sycl::access::target::local> shArr_acc_ct1(sycl::range<1>(blockSize), cgh);
                    const int gridSize_ct2 = gridSize;
                    cgh.parallel_for(
                        sycl::nd_range<3>(sycl::range<3>(1, 1, blockSize),sycl::range<3>(1, 1, blockSize)),
                        [=](sycl::nd_item<3> item_ct1) {
                            realsumCommMultiBlock_double<blockSize>(dev_ssq_vVect, dev_ssq_vVect, gridSize_ct2,item_ct1, shArr_acc_ct1.get_pointer());
                        });
                });
            }
            queue.memcpy(&ssq_vVect, dev_ssq_vVect, sizeof(double)).wait();

            double alphaLoc = ZERO;
            alphaLoc = max_vVect*sqrt(ssq_vVect);
            alphaLoc2=alphaLoc*alphaLoc;
    
            if(myid==0) {
                double alphaOther2 = alphaLoc2;
                {
                    queue.submit([&](sycl::handler &cgh) {
                        sycl::accessor<double, 1, sycl::access::mode::read_write, sycl::access::target::local> shArr_acc_ct1(sycl::range<1>(blockSize), cgh);
                        double *vVect_dev_localAstroMax_ct0=&vVect_dev[localAstroMax];
                        cgh.parallel_for(
                            sycl::nd_range<3>(sycl::range<3>(1, 1, gridSize)*sycl::range<3>(1, 1, blockSize),sycl::range<3>(1, 1, blockSize)),
                            [=](sycl::nd_item<3> item_ct1) {
                                sumCommMultiBlock_double<blockSize>(vVect_dev_localAstroMax_ct0, dev_ssq_vVect,max_vVect, nunkSplit - localAstroMax,item_ct1, shArr_acc_ct1.get_pointer());
                            });
                    });
                }
                {
                    queue.submit([&](sycl::handler &cgh) {
                        sycl::accessor<double, 1, sycl::access::mode::read_write, sycl::access::target::local> shArr_acc_ct1(sycl::range<1>(blockSize), cgh);
                        const int gridSize_ct2 = gridSize;
                        cgh.parallel_for(
                            sycl::nd_range<3>(sycl::range<3>(1, 1, blockSize),sycl::range<3>(1, 1, blockSize)),
                            [=](sycl::nd_item<3> item_ct1) {
                                realsumCommMultiBlock_double<blockSize>(dev_ssq_vVect, dev_ssq_vVect, gridSize_ct2,item_ct1, shArr_acc_ct1.get_pointer());
                            });
                    });
                }
                // queue.wait_and_throw();
                queue.memcpy(&ssq_vVect, dev_ssq_vVect, sizeof(double)).wait();
                alphaLoc = max_vVect*sqrt(ssq_vVect);
                alphaLoc2 = alphaLoc*alphaLoc;
                alphaLoc2 = alphaOther2 + alphaLoc2;
            }

            #ifdef USE_MPI
                //------------------------------------------------------------------------------------------------
                starttime=MPI_Wtime();
                MPI_Allreduce(MPI_IN_PLACE,&alphaLoc2,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
                communicationtime+=MPI_Wtime()-starttime;
                //------------------------------------------------------------------------------------------------
            #endif


            alpha=sqrt(alphaLoc2);
                    
                    
            if (alpha > ZERO) {
                {
                    queue.parallel_for(
                        sycl::nd_range<3>(sycl::range<3>(1, 1, BlockXGrid)*sycl::range<3>(1, 1, ThreadsXBlock),sycl::range<3>(1, 1, ThreadsXBlock)),
                        [=](sycl::nd_item<3> item_ct1) {
                            dscal(vVect_dev,1/alpha,nunkSplit,ONE,item_ct1);
                        });
                }
            }



        }


            

        //      ------------------------------------------------------------------
        //      Use a plane rotation to eliminate the damping parameter.
        //      This alters the diagonal (rhobar) of the lower-bidiagonal matrix.
        //      ------------------------------------------------------------------
        rhbar1 = rhobar;
        if ( damped ) {
            rhbar1 = d2norm( rhobar, damp );
            cs1    = rhobar / rhbar1;
            sn1    = damp   / rhbar1;
            psi    = sn1 * phibar;
            phibar = cs1 * phibar;
        }

        //      ------------------------------------------------------------------
        //      Use a plane rotation to eliminate the subdiagonal element (beta)
        //      of the lower-bidiagonal matrix, giving an upper-bidiagonal matrix.
        //      ------------------------------------------------------------------
        rho    =   d2norm( rhbar1, beta );
        cs     =   rhbar1 / rho;
        sn     =   beta   / rho;
        theta  =   sn * alpha;
        rhobar = - cs * alpha;
        phi    =   cs * phibar;
        phibar =   sn * phibar;
        tau    =   sn * phi;

        //      ------------------------------------------------------------------
        //      Update  x, w  and (perhaps) the standard error estimates.
        //      ------------------------------------------------------------------
        t1     =   phi   / rho;
        t2     = - theta / rho;
        t3     =   ONE   / rho;
        dknorm =   ZERO;


        {
            queue.submit([&](sycl::handler &cgh) {
            sycl::accessor<double, 1, sycl::access::mode::read_write, sycl::access::target::local> shArr_acc_ct1(sycl::range<1>(blockSize), cgh);
            cgh.parallel_for(
                sycl::nd_range<3>(sycl::range<3>(1, 1, gridSize)*sycl::range<3>(1, 1, blockSize),sycl::range<3>(1, 1, blockSize)),
                [=](sycl::nd_item<3> item_ct1) {
                    dknorm_compute<blockSize>(dknorm_vec, wVect_dev,0,nAstroElements * nAstroPSolved,t3,item_ct1, shArr_acc_ct1.get_pointer());
                });

            });
        }

        #ifdef USE_MPI
            queue.memcpy(&dknorm, dknorm_vec, sizeof(double)).wait();
            //------------------------------------------------------------------------------------------------
            starttime=MPI_Wtime();
            MPI_Allreduce(MPI_IN_PLACE,&dknorm,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
            communicationtime+=MPI_Wtime()-starttime;
            //------------------------------------------------------------------------------------------------ 		
  	        queue.memcpy(vVect, vVect_dev, nunkSplit * sizeof(double)).wait();
        #endif

        {
            queue.parallel_for(
                sycl::nd_range<3>(sycl::range<3>(1, 1, BlockXGrid)*sycl::range<3>(1, 1, ThreadsXBlock),sycl::range<3>(1, 1, ThreadsXBlock)),
                [=](sycl::nd_item<3> item_ct1) {
                    transform1(xSolution_dev, wVect_dev, 0, localAstro, t1,item_ct1);
                });
        }

        if(wantse){
            queue.parallel_for(
                sycl::nd_range<3>(sycl::range<3>(1, 1, BlockXGrid)*sycl::range<3>(1, 1, ThreadsXBlock),sycl::range<3>(1, 1, ThreadsXBlock)),
                [=](sycl::nd_item<3> item_ct1) {
                    transform2(standardError_dev, wVect_dev, 0, localAstro, t3,item_ct1);
                });
        }

        {
            queue.parallel_for(
                sycl::nd_range<3>(sycl::range<3>(1, 1, BlockXGrid)*sycl::range<3>(1, 1, ThreadsXBlock),sycl::range<3>(1, 1, ThreadsXBlock)),
                [=](sycl::nd_item<3> item_ct1) {
                    transform3(wVect_dev, vVect_dev, 0, localAstro, t2,item_ct1);
                });
        }


        {
            queue.parallel_for(
                sycl::nd_range<3>(sycl::range<3>(1, 1, BlockXGrid)*sycl::range<3>(1, 1, ThreadsXBlock),sycl::range<3>(1, 1, ThreadsXBlock)),
                [=](sycl::nd_item<3> item_ct1) {
                    transform1(xSolution_dev, wVect_dev, localAstroMax,localAstroMax + other, t1, item_ct1);
                });
        }

        if(wantse){
            queue.parallel_for(
                sycl::nd_range<3>(sycl::range<3>(1, 1, BlockXGrid)*sycl::range<3>(1, 1, ThreadsXBlock),sycl::range<3>(1, 1, ThreadsXBlock)),
                [=](sycl::nd_item<3> item_ct1) {
                    transform2(standardError_dev, wVect_dev, localAstroMax,localAstroMax + other, t3, item_ct1);
                });
        }

        {
            queue.parallel_for(
                sycl::nd_range<3>(sycl::range<3>(1, 1, BlockXGrid)*sycl::range<3>(1, 1, ThreadsXBlock),sycl::range<3>(1, 1, ThreadsXBlock)),
                [=](sycl::nd_item<3> item_ct1) {
                    transform3(wVect_dev, vVect_dev, localAstroMax,localAstroMax + other, t2, item_ct1);
                });
        }


        {
            queue.submit([&](sycl::handler &cgh) {
            sycl::accessor<double, 1, sycl::access::mode::read_write, sycl::access::target::local> shArr_acc_ct1(sycl::range<1>(blockSize), cgh);
            cgh.parallel_for(
                sycl::nd_range<3>(sycl::range<3>(1, 1, gridSize)*sycl::range<3>(1, 1, blockSize),sycl::range<3>(1, 1, blockSize)),
                [=](sycl::nd_item<3> item_ct1) {
                    dknorm_compute<blockSize>(dknorm_vec, wVect_dev,0,nAstroElements * nAstroPSolved,t3,item_ct1, shArr_acc_ct1.get_pointer());
                });

            });
        }
        queue.memcpy(&dknorm, dknorm_vec, sizeof(double)).wait();
        
        ///////////////////////////

        //      ------------------------------------------------------------------
        //      Monitor the norm of d_k, the update to x.
        //      dknorm = norm( d_k )
        //      dnorm  = norm( D_k ),        where   D_k = (d_1, d_2, ..., d_k )
        //      dxk    = norm( phi_k d_k ),  where new x = x_k + phi_k d_k.
        //      ------------------------------------------------------------------
        dknorm = sqrt( dknorm );
        dnorm  = d2norm( dnorm, dknorm );
        dxk    = fabs( phi * dknorm );
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
        arnorm =   alpha * fabs( tau );

        //      Now use these norms to estimate certain other quantities,
        //      some of which will be small near a solution.

        #ifdef VERBOSE 
            alfopt =   sqrt( rnorm / (dnorm * xnorm) );
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
    #else
        printf("Average iteration time: %lf \n", totTimeIteration.count()/itn);
        printf("Average kernel Aprod1Astro time: %lf \n", 1e-9*timekernel[0]/itn);
        printf("Average kernel Aprod1Att time: %lf \n", 1e-9*timekernel[1]/itn);
        printf("Average kernel Aprod1Instr time: %lf \n", 1e-9*timekernel[2]/itn);
        printf("Average kernel Aprod2Astro time: %lf \n", 1e-9*timekernel[3]/itn);
        printf("Average kernel Aprod2Att time: %lf \n", 1e-9*timekernel[4]/itn);
        printf("Average kernel Aprod2Instr time: %lf \n", 1e-9*timekernel[5]/itn);
    #endif
    //------------------------------------------------------------------------------------------------ 		

    queue.memcpy(xSolution, xSolution_dev, nunkSplit*sizeof(double)).wait();
    queue.memcpy(standardError, standardError_dev, nunkSplit*sizeof(double)).wait();


    if ( wantse ) {
        t    =   ONE;
        if (m > n)     t = m - n;
        if ( damped )  t = m;
        t    =   rnorm / sqrt( t );
      
        for (long i = 0; i < nunkSplit; i++)
            standardError[i]  = t * sqrt( standardError[i] );
        
    }


    *istop_out  = istop;
    *itn_out    = itn;
    *anorm_out  = anorm;
    *acond_out  = acond;
    *rnorm_out  = rnorm;
    *arnorm_out = test2;
    *xnorm_out  = xnorm;

    sycl::free(vVect_dev, queue);
    sycl::free(wVect_dev, queue);
    sycl::free(knownTerms_dev, queue);
    sycl::free(kAuxcopy_dev, queue);
    sycl::free(vAuxVect_dev, queue);
    sycl::free(instrCol_dev, queue);
    sycl::free(instrConstrIlung_dev, queue);
    sycl::free(dev_vVect_glob_sum, queue);
    sycl::free(dev_max_knownTerms, queue);
    sycl::free(dev_ssq_knownTerms, queue);
    sycl::free(xSolution_dev, queue);
    sycl::free(standardError_dev, queue);
    sycl::free(dknorm_vec, queue);


    sycl::free(dev_max_vVect, queue);
    sycl::free(dev_ssq_vVect, queue);
    sycl::free(matrixIndexAstro_dev, queue);
    sycl::free(matrixIndexAtt_dev, queue);
    sycl::free(startend_dev, queue);

    if(nAstroPSolved)   sycl::free(sysmatAstro_dev, queue);
    if(nAttP)           sycl::free(sysmatAtt_dev, queue);
    if(nInstrPSolved)   sycl::free(sysmatInstr_dev, queue);    
    if(nGlobP)          sycl::free(sysmatGloB_dev, queue);
    if(nTotConstraints) sycl::free(sysmatConstr_dev, queue);

    free(startend);



    return;
}
