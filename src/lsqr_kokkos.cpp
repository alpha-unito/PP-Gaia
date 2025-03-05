/*

KOKKOS Version

*/

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

#include <iostream>


#include <Kokkos_Core.hpp>


#ifdef USE_MPI
    #include <mpi.h>
#endif


#define ZERO   0.0
#define ONE    1.0
#define MONE    -1.0


//-----------------------------------------------------------------------------------------------------
//-----------------------------------------------------------------------------------------------------
//-----------------------------------------------------------------------------------------------------
inline double d2norm(const double& a, const double& b){

    double scale=std::fabs(a)+std::fabs(b);
    return (!scale) ? ZERO : scale * std::sqrt((a/scale)*(a/scale)+(b/scale)*(b/scale)); 

}

template<typename L>
inline void dload(double* x, const L n, const double alpha){

    #pragma omp parallel for 
    for(int i=0; i<n; ++i){
        x[i]=alpha;
    }
}

template<typename K>    //K should be a kokkos_view
inline void dscal(Kokkos::View<K> array, const double val, const long N, const double sign ){

    Kokkos::parallel_for("dscal", N, KOKKOS_LAMBDA (const long i) {
        array[i]=sign*(array[i]*val);
    });

}


template<typename K, typename L>
inline double maxCommMultiBlock_double(Kokkos::View<K> gArr, const L arraySize){

    double max{ZERO};
    
    Kokkos::parallel_reduce("max_reduce", arraySize, KOKKOS_LAMBDA(const L i, double& local_max) {
        if(fabs(gArr[i])>local_max) local_max=fabs(gArr[i]);
    }, Kokkos::Max<double>(max)); 

    return max;

}

template<typename K, typename L>
inline double sumCommMultiBlock_double(Kokkos::View<K> gArr, const L arraySize, const double max){

    const double d{1/max};
    double sum{ZERO};

    Kokkos::parallel_reduce("sumCommMultiBlock_double", arraySize, KOKKOS_LAMBDA (const L i, double& locsum) {
        locsum+=(gArr[i]*d)*(gArr[i]*d);
    },sum);
    return sum;
}

template<typename K, typename L>
inline double sumCommMultiBlock_double_start(Kokkos::View<K> gArr, const L arraySize, const double max, const L start){

    const double d{1/max};
    double sum{ZERO};

    Kokkos::parallel_reduce("sumCommMultiBlock_double_start", arraySize, KOKKOS_LAMBDA (const L i, double& locsum) {
        locsum+=(gArr[start+i]*d)*(gArr[start+i]*d);
    },sum);
    return sum;
}



template<typename K, typename L>
void vVect_Put_To_Zero_Kernel (Kokkos::View<K> vVect_dev, const L localAstroMax, const L nunkSplit)
{

    Kokkos::parallel_for("vVect_Put_To_Zero_Kernel", Kokkos::RangePolicy<>(localAstroMax, nunkSplit),
        KOKKOS_LAMBDA (const L i) {
        vVect_dev[i]=ZERO;
    });

}

template<typename K>
void cblas_dcopy_kernel(Kokkos::View<K> wVect_dev, const Kokkos::View<K> vVect_dev, const long nunkSplit)
{

    Kokkos::parallel_for("cblas_dcopy_kernel", nunkSplit, KOKKOS_LAMBDA (const long i) {
        wVect_dev[i]=vVect_dev[i];
    });
}

template<typename K, typename L, typename I>
void kAuxcopy_Kernel (Kokkos::View<K> knownTerms_dev, Kokkos::View<K> kAuxcopy_dev, const L nobs, const I N)
{

    Kokkos::parallel_for("kAuxcopy_Kernel", N, KOKKOS_LAMBDA (const I i) {
        kAuxcopy_dev[i]=knownTerms_dev[nobs+i];
        knownTerms_dev[nobs+i]=ZERO;
    });

}

template<typename K, typename L, typename I>
void kauxsum(Kokkos::View<K> knownTerms_dev,const Kokkos::View<K> kAuxcopy_dev, const L start, const I N){

    Kokkos::parallel_for("kauxsum", N, KOKKOS_LAMBDA (const long ix) {
        knownTerms_dev[start+ix] = knownTerms_dev[start+ix]+kAuxcopy_dev[ix];
    });
    
}

template<typename K, typename L>
void vAuxVect_Kernel (Kokkos::View<K> vVect_dev, Kokkos::View<K> vAuxVect_dev, const L N)
{

    Kokkos::parallel_for("vAuxVect_Kernel", N, KOKKOS_LAMBDA (const L i) {
        vAuxVect_dev[i]=vVect_dev[i];
        vVect_dev[i]=0;
    });

}

template<typename K, typename L>
void vaux_sum(Kokkos::View<K> vVect_dev,const Kokkos::View<K> vAuxVect_dev, const L localAstroMax){

    Kokkos::parallel_for("vaux_sum",localAstroMax,KOKKOS_LAMBDA (const L ix){
        vVect_dev[ix] += vAuxVect_dev[ix];
    });

}


//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
//                                                                                    APROD1
//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
//--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
template<typename K, typename I>
void aprod1_Kernel_astro(Kokkos::View<K> knownTerms_dev, 
                        const Kokkos::View<K> systemMatrix_dev, 
                        const Kokkos::View<K> vVect_dev, 
                        const Kokkos::View<I> matrixIndexAstro, 
                        const long offLocalAstro, 
                        const long nobs, 
                        const short nAstroPSolved){

    Kokkos::parallel_for("aprod1_Kernel_astro", nobs, KOKKOS_LAMBDA (const long ix) {
        double sum{ZERO};
        const long jstartAstro = matrixIndexAstro[ix] - offLocalAstro;
        
        for(short jx = 0; jx <   nAstroPSolved; jx++) {
            double matrixVal = systemMatrix_dev[ix * nAstroPSolved + jx];  
            double vectVal = vVect_dev[jstartAstro + jx];                  
            sum += matrixVal * vectVal;  
        }
        knownTerms_dev[ix]+=sum;
    });

}

template<typename K, typename I>
void aprod1_Kernel_att_AttAxis(Kokkos::View<K> knownTerms_dev, 
                                const Kokkos::View<K> systemMatrix_dev, 
                                const Kokkos::View<K> vVect_dev, 
                                const Kokkos::View<I> matrixIndexAtt, 
                                const long  nAttP, 
                                const long nobs, 
                                const long nDegFreedomAtt, 
                                const long offLocalAtt, 
                                const short nAttParAxis)
{

    Kokkos::parallel_for("aprod1_Kernel_att_AttAxis", nobs, KOKKOS_LAMBDA (const long ix) {
        double sum{ZERO};
        long jstartAtt_0{matrixIndexAtt[ix] + offLocalAtt}; 
        
        for(auto inpax = 0;inpax<nAttParAxis; ++inpax)
            sum += systemMatrix_dev[ix*nAttP + inpax ] * vVect_dev[jstartAtt_0 + inpax];
        jstartAtt_0 += nDegFreedomAtt;
        
        
        for(auto inpax = 0;inpax<nAttParAxis;++inpax)
            sum+=  systemMatrix_dev[ix*nAttP+nAttParAxis+inpax] * vVect_dev[jstartAtt_0+inpax];
        jstartAtt_0 += nDegFreedomAtt;

        
        for(auto inpax = 0;inpax<nAttParAxis;++inpax)
            sum+= systemMatrix_dev[ix*nAttP + nAttParAxis+nAttParAxis +inpax] * vVect_dev[jstartAtt_0+inpax];

        knownTerms_dev[ix]+=sum;
    });
}

template<typename K, typename I>
void aprod1_Kernel_instr (Kokkos::View<K> knownTerms_dev, 
                            const Kokkos::View<K> systemMatrix_dev, 
                            const Kokkos::View<K> vVect_dev, 
                            const Kokkos::View<I> instrCol_dev, 
                            const long nobs, 
                            const long offLocalInstr, 
                            const short nInstrPSolved){
        
    Kokkos::parallel_for("aprod1_Kernel_instr", nobs, KOKKOS_LAMBDA (const long ix) {
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

template<typename K>
void aprod1_Kernel_glob(Kokkos::View<K> knownTerms_dev, 
                        const Kokkos::View<K> systemMatrix_dev, 
                        const Kokkos::View<K> vVect_dev, 
                        const long offLocalGlob, 
                        const long nobs, 
                        const short nGlobP)
{        
    Kokkos::parallel_for("aprod1_Kernel_glob", nobs, KOKKOS_LAMBDA (const long ix) {
        double sum{ZERO};
        for(short inGlob=0;inGlob<nGlobP;inGlob++){
            sum+=systemMatrix_dev[ix * nGlobP + inGlob]*vVect_dev[offLocalGlob+inGlob];
        }
        knownTerms_dev[ix]+=sum;
    });
}




template<typename K>
void aprod1_Kernel_ExtConstr(Kokkos::View<K> knownTerms_dev, 
                            const Kokkos::View<K> systemMatrix_dev, 
                            const Kokkos::View<K> vVect_dev, 
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

    Kokkos::parallel_for("aprod1_Kernel_ExtConstr", nEqExtConstr, KOKKOS_LAMBDA (const int iexc) {

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
            Kokkos::atomic_add(&knownTerms_dev[nobs+iexc],sum);
    });

}

// /// BarConstr
template<typename K>
void aprod1_Kernel_BarConstr(Kokkos::View<K> knownTerms_dev, 
                            const Kokkos::View<K> systemMatrix_dev, 
                            const Kokkos::View<K> vVect_dev, 
                            const int nOfElextObs, 
                            const int nOfElBarObs, 
                            const int nEqExtConstr, 
                            const long nobs, 
                            const int nEqBarConstr, 
                            const int& numOfBarStar, 
                            const short nAstroPSolved){

    const long ktIx{nobs + nEqExtConstr};

    Kokkos::parallel_for("aprod1_Kernel_BarConstr", numOfBarStar*nAstroPSolved, KOKKOS_LAMBDA (const int j3) {
        for(int iexc=0;iexc<nEqBarConstr;iexc++ ){
            double sum{ZERO};
            const long offBarConstrIx=iexc*nOfElBarObs;
            sum = sum + systemMatrix_dev[offBarConstrIx+j3]*vVect_dev[j3];
            Kokkos::atomic_add(&knownTerms_dev[ktIx+iexc],sum);
        }
    });
}


/// InstrConstr
template<typename K, typename I>
void aprod1_Kernel_InstrConstr(Kokkos::View<K> knownTerms_dev, 
                                const Kokkos::View<K> systemMatrix_dev, 
                                const Kokkos::View<K> vVect_dev, 
                                const Kokkos::View<I> instrConstrIlung_dev, 
                                const Kokkos::View<I> instrCol_dev, 
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

        Kokkos::parallel_for("aprod1_Kernel_InstrConstr", Kokkos::RangePolicy<>(myid, nOfInstrConstr), KOKKOS_LAMBDA (const int i1) {

            if (i1 % nproc == myid) {
            double sum{ZERO};
            long offSetInstrInc{offSetInstrConstr};
            int offSetInstr{0};
            for(int m=0;m<i1;m++) 
            {
                offSetInstrInc+=instrConstrIlung_dev[m];
                offSetInstr+=instrConstrIlung_dev[m];
            }
            const long offvV{nobs*nInstrPSolved+offSetInstr};
            for(int j3 = 0; j3 < instrConstrIlung_dev[i1]; j3++)
                sum+=systemMatrix_dev[offSetInstrInc+j3]*vVect_dev[offSetInstrConstr1+instrCol_dev[offvV+j3]];
            Kokkos::atomic_add(&knownTerms_dev[ktIx+i1],sum);
            }
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
template<typename K, typename I>
void aprod2_Kernel_astro(Kokkos::View<K> vVect_dev, 
                        const Kokkos::View<K> systemMatrix_dev, 
                        const Kokkos::View<K> knownTerms_dev, 
                        const Kokkos::View<I> matrixIndexAstro, 
                        const Kokkos::View<I> startend, 
                        const long offLocalAstro, 
                        const long nobs, 
                        const short nAstroPSolved){

    Kokkos::parallel_for("aprod2_Kernel_astro", nobs, KOKKOS_LAMBDA (const long ix) {
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


template<typename K, typename I>
void aprod2_Kernel_att_AttAxis(Kokkos::View<K> vVect_dev, 
                                const Kokkos::View<K> systemMatrix_dev, 
                                const Kokkos::View<K> knownTerms_dev, 
                                const Kokkos::View<I> matrixIndexAtt, 
                                const long nAttP, 
                                const long nDegFreedomAtt, 
                                const long offLocalAtt, 
                                const long nobs, 
                                const short nAstroPSolved, 
                                const short nAttParAxis){

    Kokkos::parallel_for("aprod2_Kernel_att_AttAxis", nobs, KOKKOS_LAMBDA (const long ix) {

        long jstartAtt = matrixIndexAtt[ix] + offLocalAtt;
        
        for (auto inpax = 0; inpax < nAttParAxis; ++inpax){
            Kokkos::atomic_add(&vVect_dev[jstartAtt+inpax],systemMatrix_dev[ix*nAttP+inpax]*knownTerms_dev[ix]);
        }
        jstartAtt +=nDegFreedomAtt;
        
        for (auto inpax = 0; inpax < nAttParAxis; ++inpax){
            Kokkos::atomic_add(&vVect_dev[jstartAtt+inpax],systemMatrix_dev[ix*nAttP+nAttParAxis+inpax]*knownTerms_dev[ix]);
        }
        jstartAtt +=nDegFreedomAtt;
        
        for (auto inpax = 0; inpax < nAttParAxis; ++inpax){
            Kokkos::atomic_add(&vVect_dev[jstartAtt+inpax],systemMatrix_dev[ix*nAttP+nAttParAxis+nAttParAxis+inpax]*knownTerms_dev[ix]);
        }

    });

}

template<typename K, typename I>
void aprod2_Kernel_instr(Kokkos::View<K> vVect_dev,
                        const Kokkos::View<K> systemMatrix_dev, 
                        const Kokkos::View<K> knownTerms_dev, 
                        const Kokkos::View<I> instrCol_dev, 
                        const long offLocalInstr, 
                        const long nobs,  
                        const short nInstrPSolved){

    Kokkos::parallel_for("aprod2_Kernel_instr", nobs, KOKKOS_LAMBDA (const long ix) {
        
        for (auto inInstr = 0; inInstr < nInstrPSolved; inInstr++){
            double MatVal{systemMatrix_dev[ix*nInstrPSolved + inInstr]};
            double rhs{knownTerms_dev[ix]};
            double tmp=MatVal*rhs;
            Kokkos::atomic_add(&vVect_dev[offLocalInstr + instrCol_dev[ix*nInstrPSolved+inInstr]],tmp);
        }
    });
}


template<typename K>
void aprod2_Kernel_ExtConstr(Kokkos::View<K> vVect_dev, 
                            const Kokkos::View<K> systemMatrix_dev, 
                            const Kokkos::View<K> knownTerms_dev, 
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

    Kokkos::parallel_for("aprod2_Kernel_ExtConstr_1", numOfExtStar, KOKKOS_LAMBDA (const int i) {
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

    Kokkos::parallel_for("aprod2_Kernel_ExtConstr_2", numOfExtAttCol, KOKKOS_LAMBDA (const int i) {
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

template<typename K>
void aprod2_Kernel_BarConstr(Kokkos::View<K> vVect_dev, 
                            const Kokkos::View<K> systemMatrix_dev, 
                            const Kokkos::View<K> knownTerms_dev, 
                            const long nobs, 
                            const int nEqBarConstr, 
                            const int nEqExtConstr, 
                            const int nOfElextObs, 
                            const int nOfElBarObs, 
                            const int numOfBarStar, 
                            const short nAstroPSolved){
    

    Kokkos::parallel_for("aprod2_Kernel_ExtConstr_2", numOfBarStar, KOKKOS_LAMBDA (const int yx) {
        for(int ix=0;ix<nEqBarConstr;++ix){  
            const double yi{knownTerms_dev[nobs+nEqExtConstr+ix]};
            const long offBarStarConstrEq{nEqExtConstr*nOfElextObs+ix*nOfElBarObs};
            for(auto j2=0;j2<nAstroPSolved;j2++){
                vVect_dev[yx*nAstroPSolved + j2] += systemMatrix_dev[offBarStarConstrEq+yx*nAstroPSolved+j2]*yi;
            }
        }
    });

    Kokkos::fence();

}

template<typename K, typename I>
void aprod2_Kernel_InstrConstr(Kokkos::View<K> vVect, 
                                const Kokkos::View<K> systemMatrix, 
                                const Kokkos::View<K> knownTerms, 
                                const Kokkos::View<I> instrConstrIlung, 
                                const Kokkos::View<I> instrCol, 
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
        const long off3{nOfElextObs*nEqExtConstr+nOfElBarObs*nEqBarConstr};
        const long offInstrUnk{VrIdAstroPDimMax*nAstroPSolved+nDegFreedomAtt*nAttAxes};
        const long off2{mapNoss+nEqExtConstr+nEqBarConstr};
        const long off4{mapNoss*nInstrPSolved};

        Kokkos::parallel_for("aprod2_Kernel_InstrConstr", Kokkos::RangePolicy<>(myid, nOfInstrConstr), KOKKOS_LAMBDA (const int k1_Aux) {
            if (k1_Aux % nproc == myid) {
                const double yi{knownTerms[off2+k1_Aux]};
                int offSetInstr=0;
                for(int m=0;m<k1_Aux;++m){
                    offSetInstr=offSetInstr+instrConstrIlung[m];
                }
                const long off1{off3+offSetInstr};
                const long off5{off4+offSetInstr};
                for(int j=0;j<instrConstrIlung[k1_Aux];j++){
                    Kokkos::atomic_add(&vVect[offInstrUnk+instrCol[off5+j]],systemMatrix[off1+j]*yi);
                }
            }
        });
    }


}

template<typename T>
inline long create_startend_gpulist(const long* matrixIndexAstro,T startend_dev, const long mapNoss, const long nnz){
    long *startend=(long*)malloc(sizeof(long)*(nnz+1));
    long count=0, nnz2=0;
    startend[nnz2]=count;nnz2++;
    for(long i=0; i<mapNoss-1; ++i){
        if(matrixIndexAstro[i]!=matrixIndexAstro[i+1]){
            count++;
            startend[nnz2]=count;
            nnz2++;
        }else{
            count++;
        }
    }
    startend[nnz2]=count+1;

    Kokkos::deep_copy(startend_dev, Kokkos::View<long*, Kokkos::HostSpace>(startend, (nnz2+1)));

    free(startend);
    return nnz2;
}

// ---------------------------------------------------------------------
// LSQR
// ---------------------------------------------------------------------
void lsqr(
          long int m,
          long int n,
          double damp,
          double *knownTerms,     // len = m  reported as u
          double *vVect,     // len = n reported as v
          double *wVect,     // len = n  reported as w
          double *xSolution,     // len = n reported as x
          double *standardError,    // len at least n.  May be NULL. reported as se
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
          long *matrixIndexAstro, // reported as janew
          long *matrixIndexAtt, // reported as janew
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
//                aprod ( mode, m, n, x, y, UsrWrk )
//
//     which must perform the following functions:
//
//                If mode = 1, compute  y = y + A*x.
//                If mode = 2, compute  x = x + A(transpose)*y.
//
//     The vectors x and y are input parameters in both cases.
//     If  mode = 1,  y should be altered without changing x.
//     If  mode = 2,  x should be altered without changing y.
//     The parameter UsrWrk may be used for workspace as described
//     below.
//
//     The rhs vector b is input via u, and subsequently overwritten.
//
//
//     Note:  LSQR uses an iterative method to approximate the solution.
//     The number of iterations required to reach a certain accuracy
//     depends strongly on the scaling of the problem.  Poor scaling of
//     the rows or columns of A should therefore be avoided where
//     possible.
//
//     For example, in problem 1 the solution is unaltered by
//     row-scaling.  If a row of A is very small or large compared to
//     the other rows of A, the corresponding row of ( A  b ) should be
//     scaled up or down.
//
//     In problems 1 and 2, the solution x is easily recovered
//     following column-scaling.  Unless better information is known,
//     the nonzero columns of A should be scaled so that they all have
//     the same Euclidean norm (e.g., 1.0).
//
//     In problem 3, there is no freedom to re-scale if damp is
//     nonzero.  However, the value of damp should be assigned only
//     after attention has been paid to the scaling of A.
//
//     The parameter damp is intended to help regularize
//     ill-conditioned systems, by preventing the true solution from
//     being very large.  Another aid to regularization is provided by
//     the parameter acond, which may be used to terminate iterations
//     before the computed solution becomes very large.
//
//     Note that x is not an input parameter.
//     If some initial estimate x0 is known and if damp = 0,
//     one could proceed as follows:
//
//       1. Compute a residual vector     r0 = b - A*x0.
//       2. Use LSQR to solve the system  A*dx = r0.
//       3. Add the correction dx to obtain a final solution x = x0 + dx.
//
//     This requires that x0 be available before and after the call
//     to LSQR.  To judge the benefits, suppose LSQR takes k1 iterations
//     to solve A*x = b and k2 iterations to solve A*dx = r0.
//     If x0 is "good", norm(r0) will be smaller than norm(b).
//     If the same stopping tolerances atol and btol are used for each
//     system, k1 and k2 will be similar, but the final solution x0 + dx
//     should be more accurate.  The only way to reduce the total work
//     is to use a larger stopping tolerance for the second system.
//     If some value btol is suitable for A*x = b, the larger value
//     btol*norm(b)/norm(r0)  should be suitable for A*dx = r0.
//
//     Preconditioning is another way to reduce the number of iterations.
//     If it is possible to solve a related system M*x = b efficiently,
//     where M approximates A in some helpful way
//     (e.g. M - A has low rank or its elements are small relative to
//     those of A), LSQR may converge more rapidly on the system
//           A*M(inverse)*z = b,
//     after which x can be recovered by solving M*x = z.
//
//     NOTE: If A is symmetric, LSQR should not be used!
//     Alternatives are the symmetric conjugate-gradient method (cg)
//     and/or SYMMLQ.
//     SYMMLQ is an implementation of symmetric cg that applies to
//     any symmetric A and will converge more rapidly than LSQR.
//     If A is positive definite, there are other implementations of
//     symmetric cg that require slightly less work per iteration
//     than SYMMLQ (but will take the same number of iterations).
//
//
//     Notation
//     --------
//
//     The following quantities are used in discussing the subroutine
//     parameters:
//
//     Abar   =  (   A    ),          bbar  =  ( b )
//               ( damp*I )                    ( 0 )
//
//     r      =  b  -  A*x,           rbar  =  bbar  -  Abar*x
//
//     rnorm  =  sqrt( norm(r)**2  +  damp**2 * norm(x)**2 )
//            =  norm( rbar )
//
//     relpr  =  the relative precision of floating-point arithmetic
//               on the machine being used.  On most machines,
//               relpr is about 1.0e-7 and 1.0d-16 in single and double
//               precision respectively.
//
//     LSQR  minimizes the function rnorm with respect to x.
//
//
//     Parameters
//     ----------
//
//     m       input      m, the number of rows in A.
//
//     n       input      n, the number of columns in A.
//
//     aprod   external   See above.
//
//     damp    input      The damping parameter for problem 3 above.
//                        (damp should be 0.0 for problems 1 and 2.)
//                        If the system A*x = b is incompatible, values
//                        of damp in the range 0 to sqrt(relpr)*norm(A)
//                        will probably have a negligible effect.
//                        Larger values of damp will tend to decrease
//                        the norm of x and reduce the number of 
//                        iterations required by LSQR.
//
//                        The work per iteration and the storage needed
//                        by LSQR are the same for all values of damp.
//
//     rw      workspace  Transit pointer to user's workspace.
//                        Note:  LSQR  does not explicitly use this
//                        parameter, but passes it to subroutine aprod for
//                        possible use as workspace.
//
//     u(m)    input      The rhs vector b.  Beware that u is
//                        over-written by LSQR.
//
//     v(n)    workspace
//
//     w(n)    workspace
//
//     x(n)    output     Returns the computed solution x.
//
//     se(*)   output     If m .gt. n  or  damp .gt. 0,  the system is
//             (maybe)    overdetermined and the standard errors may be
//                        useful.  (See the first LSQR reference.)
//                        Otherwise (m .le. n  and  damp = 0) they do not
//                        mean much.  Some time and storage can be saved
//                        by setting  se = NULL.  In that case, se will
//                        not be touched.
//
//                        If se is not NULL, then the dimension of se must
//                        be n or more.  se(1:n) then returns standard error
//                        estimates for the components of x.
//                        For each i, se(i) is set to the value
//                           rnorm * sqrt( sigma(i,i) / t ),
//                        where sigma(i,i) is an estimate of the i-th
//                        diagonal of the inverse of Abar(transpose)*Abar
//                        and  t = 1      if  m .le. n,
//                             t = m - n  if  m .gt. n  and  damp = 0,
//                             t = m      if  damp .ne. 0.
//
//     atol    input      An estimate of the relative error in the data
//                        defining the matrix A.  For example,
//                        if A is accurate to about 6 digits, set
//                        atol = 1.0e-6 .
//
//     btol    input      An estimate of the relative error in the data
//                        defining the rhs vector b.  For example,
//                        if b is accurate to about 6 digits, set
//                        btol = 1.0e-6 .
//
//     conlim  input      An upper limit on cond(Abar), the apparent
//                        condition number of the matrix Abar.
//                        Iterations will be terminated if a computed
//                        estimate of cond(Abar) exceeds conlim.
//                        This is intended to prevent certain small or
//                        zero singular values of A or Abar from
//                        coming into effect and causing unwanted growth
//                        in the computed solution.
//
//                        conlim and damp may be used separately or
//                        together to regularize ill-conditioned systems.
//
//                        Normally, conlim should be in the range
//                        1000 to 1/relpr.
//                        Suggested value:
//                        conlim = 1/(100*relpr)  for compatible systems,
//                        conlim = 1/(10*sqrt(relpr)) for least squares.
//
//             Note:  If the user is not concerned about the parameters
//             atol, btol and conlim, any or all of them may be set
//             to zero.  The effect will be the same as the values
//             relpr, relpr and 1/relpr respectively.
//
//     itnlim  input      An upper limit on the number of iterations.
//                        Suggested value:
//                        itnlim = n/2   for well-conditioned systems
//                                       with clustered singular values,
//                        itnlim = 4*n   otherwise.
//
//     nout    input      File number for printed output.  If positive,
//                        a summary will be printed on file nout.
//
//     istop   output     An integer giving the reason for termination:
//
//                0       x = 0  is the exact solution.
//                        No iterations were performed.
//
//                1       The equations A*x = b are probably
//                        compatible.  Norm(A*x - b) is sufficiently
//                        small, given the values of atol and btol.
//
//                2       damp is zero.  The system A*x = b is probably
//                        not compatible.  A least-squares solution has
//                        been obtained that is sufficiently accurate,
//                        given the value of atol.
//
//                3       damp is nonzero.  A damped least-squares
//                        solution has been obtained that is sufficiently
//                        accurate, given the value of atol.
//
//                4       An estimate of cond(Abar) has exceeded
//                        conlim.  The system A*x = b appears to be
//                        ill-conditioned.  Otherwise, there could be an
//                        error in subroutine aprod.
//
//                5       The iteration limit itnlim was reached.
//
//     itn     output     The number of iterations performed.
//
//     anorm   output     An estimate of the Frobenius norm of  Abar.
//                        This is the square-root of the sum of squares
//                        of the elements of Abar.
//                        If damp is small and if the columns of A
//                        have all been scaled to have length 1.0,
//                        anorm should increase to roughly sqrt(n).
//                        A radically different value for anorm may
//                        indicate an error in subroutine aprod (there
//                        may be an inconsistency between modes 1 and 2).
//
//     acond   output     An estimate of cond(Abar), the condition
//                        number of Abar.  A very high value of acond
//                        may again indicate an error in aprod.
//
//     rnorm   output     An estimate of the final value of norm(rbar),
//                        the function being minimized (see notation
//                        above).  This will be small if A*x = b has
//                        a solution.
//
//     arnorm  output     An estimate of the final value of
//                        norm( Abar(transpose)*rbar ), the norm of
//                        the residual for the usual normal equations.
//                        This should be small in all cases.  (arnorm
//                        will often be smaller than the true value
//                        computed from the output vector x.)
//
//     xnorm   output     An estimate of the norm of the final
//                        solution vector x.
//
//
//     Subroutines and functions used              
//     ------------------------------
//
//     USER               aprod
//     CBLAS              dcopy, dnrm2, dscal (see Lawson et al. below)
//
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

//  Local copies of output variables.  Output vars are assigned at exit.
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

    Kokkos::initialize();
    {
        
        const int myid=comlsqr.myid;
        const long mapNoss=static_cast<long>(comlsqr.mapNoss[myid]);

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


        //--------------------------------------------------------------------------------------------------------



        // Allocate device view
        Kokkos::View<double*> sysmatAstro_dev("sysmatAstro_dev", mapNoss*nAstroPSolved);
        Kokkos::View<double*> sysmatAtt_dev("sysmatAtt_dev", mapNoss*nAttP);
        Kokkos::View<double*> sysmatInstr_dev("sysmatInstr_dev", mapNoss*nInstrPSolved);
        Kokkos::View<double*> sysmatGloB_dev("sysmatGloB_dev", mapNoss*nGlobP);
        Kokkos::View<double*> sysmatConstr_dev("sysmatConstr_dev", nOfElextObs*nEqExtConstr+nOfElBarObs*nEqBarConstr+nElemIC);

        //Copy to from host device
        Kokkos::deep_copy(sysmatAstro_dev, Kokkos::View<double*, Kokkos::HostSpace>(sysmatAstro, mapNoss*nAstroPSolved));
        Kokkos::deep_copy(sysmatAtt_dev, Kokkos::View<double*, Kokkos::HostSpace>(sysmatAtt, mapNoss*nAttP));
        Kokkos::deep_copy(sysmatInstr_dev, Kokkos::View<double*, Kokkos::HostSpace>(sysmatInstr, mapNoss*nInstrPSolved));
        Kokkos::deep_copy(sysmatGloB_dev, Kokkos::View<double*, Kokkos::HostSpace>(sysmatGloB, mapNoss*nGlobP));
        Kokkos::deep_copy(sysmatConstr_dev, Kokkos::View<double*, Kokkos::HostSpace>(sysmatConstr, nOfElextObs*nEqExtConstr+nOfElBarObs*nEqBarConstr+nElemIC));

        // New list    
        long nnz=1;
        for(long i=0; i<mapNoss-1; i++){
            if(matrixIndexAstro[i]!=matrixIndexAstro[i+1]){
                nnz++;
            }
        }


        Kokkos::View<long*> startend_dev("startend_dev", nnz+1);

        nnz=create_startend_gpulist(matrixIndexAstro,startend_dev,mapNoss,nnz);


        Kokkos::View<long*> matrixIndexAstro_dev("matrixIndexAstro_dev", mapNoss);
        Kokkos::View<long*> matrixIndexAtt_dev("matrixIndexAtt_dev", mapNoss);
        Kokkos::deep_copy(matrixIndexAstro_dev, Kokkos::View<long*, Kokkos::HostSpace>(matrixIndexAstro, mapNoss));
        Kokkos::deep_copy(matrixIndexAtt_dev, Kokkos::View<long*, Kokkos::HostSpace>(matrixIndexAtt, mapNoss));


        //--------------------------------------------------------------------------------------------------------

        Kokkos::View<double*> vVect_dev("vVect_dev", nunkSplit);
        Kokkos::View<double*> knownTerms_dev("knownTerms_dev", nElemKnownTerms);
        Kokkos::View<double*> wVect_dev("wVect_dev", nunkSplit);
        Kokkos::View<double*> kAuxcopy_dev("kAuxcopy_dev", (nEqExtConstr+nEqBarConstr+nOfInstrConstr));
        Kokkos::View<double*> vAuxVect_dev("vAuxVect_dev", localAstroMax);
        Kokkos::View<int*> instrCol_dev("instrCol_dev", (nInstrPSolved*mapNoss+nElemIC));
        Kokkos::View<int*> instrConstrIlung_dev("instrConstrIlung_dev", nOfInstrConstr);
        Kokkos::View<double*> xSolution_dev("xSolution_dev", nunkSplit);
        Kokkos::View<double*> standardError_dev("standardError_dev", nunkSplit);

        // Create a host view for the range
        Kokkos::View<double*, Kokkos::HostSpace> knownTerms_host(knownTerms, mapNoss+(nEqExtConstr+nEqBarConstr+nOfInstrConstr));
        // Create a device subview for the range on host
        auto knownTerms_host_subview = Kokkos::subview(knownTerms_host, std::make_pair(mapNoss, mapNoss + (nEqExtConstr + nEqBarConstr + nOfInstrConstr)));
        // Create a device subview for the range on device
        auto knownTerms_dev_subview = Kokkos::subview(knownTerms_dev, std::make_pair(mapNoss, mapNoss + (nEqExtConstr + nEqBarConstr + nOfInstrConstr)));

        for(auto i=0; i<nunkSplit;++i){
            xSolution[i]=0.0;
            standardError[i]=0.0;
        }

        Kokkos::deep_copy(xSolution_dev, Kokkos::View<double*, Kokkos::HostSpace>(xSolution, nunkSplit));
        Kokkos::deep_copy(standardError_dev, Kokkos::View<double*, Kokkos::HostSpace>(standardError, nunkSplit));

        //  Copies H2D:
        Kokkos::deep_copy(instrCol_dev, Kokkos::View<int*, Kokkos::HostSpace>(instrCol, (nInstrPSolved*mapNoss+nElemIC)));
        Kokkos::deep_copy(instrConstrIlung_dev, Kokkos::View<int*, Kokkos::HostSpace>(instrConstrIlung, nOfInstrConstr));
        
        /* First copy of knownTerms from the host to the device: */
        Kokkos::deep_copy(knownTerms_dev, Kokkos::View<double*, Kokkos::HostSpace>(knownTerms, nElemKnownTerms));

    ////////////////////////////////////////////// CUDA Definitions END
    other=(long)nAttParam + nInstrParam + nGlobalParam; 
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
        
    /* copy from the host to the device: */
    Kokkos::deep_copy(vVect_dev, Kokkos::View<double*, Kokkos::HostSpace>(vVect, nunkSplit));
        
    dload(xSolution,nunkSplit, ZERO);

    if ( wantse )   dload(standardError,nunkSplit, ZERO );

    alpha  =   ZERO;
        

    max_knownTerms = maxCommMultiBlock_double(knownTerms_dev,nElemKnownTerms);

    double betaLoc, betaLoc2;

    if(!myid){
        ssq_knownTerms = sumCommMultiBlock_double(knownTerms_dev, mapNoss + nEqExtConstr + nEqBarConstr+nOfInstrConstr, max_knownTerms);
        betaLoc = max_knownTerms*std::sqrt(ssq_knownTerms);
    }else{
        ssq_knownTerms = sumCommMultiBlock_double(knownTerms_dev, mapNoss, max_knownTerms);
        betaLoc = max_knownTerms*std::sqrt(ssq_knownTerms);
    }

    betaLoc2=betaLoc*betaLoc;



    //:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    //------------------------------------------------------------------------------------------------  TIME 2
    #ifdef USE_MPI
        starttime=get_time();
            MPI_Allreduce(MPI_IN_PLACE,&betaLoc2,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
        endtime=get_time();
        communicationtime=compute_time(endtime,starttime);
    #endif
    //------------------------------------------------------------------------------------------------
    beta=sqrt(betaLoc2);
        
        
    if (beta > ZERO) 
    {
        //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::: TIME3



            dscal(knownTerms_dev,1.0/beta,nElemKnownTerms,ONE);
            if(myid) vVect_Put_To_Zero_Kernel(vVect_dev,localAstroMax,nunkSplit);
            //APROD2 CALL BEFORE LSQR
            if(nAstroPSolved) aprod2_Kernel_astro(vVect_dev, sysmatAstro_dev, knownTerms_dev, matrixIndexAstro_dev, startend_dev, offLocalAstro, nnz, nAstroPSolved);
            if(nAttP) aprod2_Kernel_att_AttAxis(vVect_dev,sysmatAtt_dev,knownTerms_dev,matrixIndexAtt_dev,nAttP,nDegFreedomAtt,offLocalAtt,mapNoss,nAstroPSolved,nAttParAxis);
            if(nInstrPSolved) aprod2_Kernel_instr(vVect_dev,sysmatInstr_dev, knownTerms_dev, instrCol_dev, offLocalInstr, mapNoss, nInstrPSolved);
            // NOT OPTIMIZED YET
            if(nEqExtConstr) aprod2_Kernel_ExtConstr(vVect_dev,sysmatConstr_dev,knownTerms_dev,mapNoss,nDegFreedomAtt,VrIdAstroPDimMax,nEqExtConstr,nOfElextObs,numOfExtStar,startingAttColExtConstr,numOfExtAttCol,nAstroPSolved,nAttAxes);
            if(nEqBarConstr) aprod2_Kernel_BarConstr(vVect_dev,sysmatConstr_dev,knownTerms_dev,mapNoss,nEqBarConstr,nEqExtConstr,nOfElextObs,nOfElBarObs,numOfBarStar,nAstroPSolved);
            if(nOfInstrConstr) aprod2_Kernel_InstrConstr(vVect_dev, sysmatConstr_dev, knownTerms_dev, instrConstrIlung_dev, instrCol_dev,  VrIdAstroPDimMax, nDegFreedomAtt, mapNoss, nEqExtConstr, nEqBarConstr, nOfElextObs, nOfElBarObs, myid, nOfInstrConstr, nproc, nInstrPSolved, nAstroPSolved, nAttAxes);
            Kokkos::deep_copy(Kokkos::View<double*, Kokkos::HostSpace>(vVect, nunkSplit), vVect_dev);

            #ifdef USE_MPI
                //------------------------------------------------------------------------------------------------  TIME4
                starttime=get_time();
                MPI_Allreduce(MPI_IN_PLACE,&vVect[localAstroMax], nAttParam+nInstrParam+nGlobalParam,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
                if(nAstroPSolved) SumCirc(vVect,comlsqr);
                endtime=get_time();
                communicationtime=compute_time(endtime,starttime);
                //------------------------------------------------------------------------------------------------
            #endif    
    
            /* copy from the host to the device: */
            Kokkos::deep_copy(vVect_dev, Kokkos::View<double*, Kokkos::HostSpace>(vVect, nunkSplit));

            nAstroElements=comlsqr.mapStar[myid][1]-comlsqr.mapStar[myid][0] + 1;
            if(myid<nproc-1)
            {
                nAstroElements=comlsqr.mapStar[myid][1]-comlsqr.mapStar[myid][0] +1;
                if(comlsqr.mapStar[myid][1]==comlsqr.mapStar[myid+1][0]) nAstroElements--;
            }

            max_vVect = maxCommMultiBlock_double(vVect_dev, nunkSplit);
            ssq_vVect = sumCommMultiBlock_double(vVect_dev, nAstroElements*nAstroPSolved, max_vVect);


           double alphaLoc=0.0;
        
            alphaLoc = max_vVect*sqrt(ssq_vVect);

            alphaLoc2=alphaLoc*alphaLoc;
            if(myid==0) {
                double alphaOther2 = alphaLoc2;
                ssq_vVect = sumCommMultiBlock_double_start(vVect_dev, nunkSplit - localAstroMax, max_vVect, localAstroMax);
                alphaLoc = max_vVect*sqrt(ssq_vVect);
                alphaLoc2 = alphaLoc*alphaLoc;
                alphaLoc2 = alphaOther2 + alphaLoc2;
            }

        #ifdef USE_MPI
            //------------------------------------------------------------------------------------------------  TIME6
            starttime=get_time();
                MPI_Allreduce(MPI_IN_PLACE,&alphaLoc2,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
            endtime=get_time();
            communicationtime=compute_time(endtime,starttime);
            //------------------------------------------------------------------------------------------------
        #endif

        alpha=sqrt(alphaLoc2);
    }

    //     //::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::: TIME7


    if(alpha > ZERO){
        dscal(vVect_dev, 1/alpha, nunkSplit, ONE);
        cblas_dcopy_kernel(wVect_dev,vVect_dev,nunkSplit);
    }

        // copy from device to host 
        Kokkos::deep_copy(Kokkos::View<double*, Kokkos::HostSpace>(vVect, nunkSplit), vVect_dev);
        Kokkos::deep_copy(Kokkos::View<double*, Kokkos::HostSpace>(wVect, nunkSplit), wVect_dev);
        Kokkos::deep_copy(Kokkos::View<double*, Kokkos::HostSpace>(knownTerms, nElemKnownTerms), knownTerms_dev);



        arnorm  = alpha * beta;

        if (arnorm == ZERO){
            if (damped  &&  istop == 2) istop = 3;

            *istop_out  = istop;
            *itn_out    = itn;
            *anorm_out  = anorm;
            *acond_out  = acond;
            *rnorm_out  = rnorm;
            *arnorm_out = test2;
            *xnorm_out  = xnorm;

            Kokkos::finalize();
            
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


        // copy from host to device
        Kokkos::deep_copy(knownTerms_dev, Kokkos::View<double*, Kokkos::HostSpace>(knownTerms, nElemKnownTerms));
        Kokkos::deep_copy(vVect_dev, Kokkos::View<double*, Kokkos::HostSpace>(vVect, nunkSplit));
        Kokkos::deep_copy(wVect_dev, Kokkos::View<double*, Kokkos::HostSpace>(wVect, nunkSplit));


        //  ==================================================================
        //  Main iteration loop.
        //  ==================================================================
        
        if (myid == 0) printf("LSQR: START ITERATIONS\n");
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

            dscal(knownTerms_dev, alpha, nElemKnownTerms, MONE);
            kAuxcopy_Kernel(knownTerms_dev, kAuxcopy_dev, mapNoss, nEqExtConstr+nEqBarConstr+nOfInstrConstr);

        // //////////////////////////////////// APROD MODE 1
            #ifdef KERNELTIME
                milliseconds = 0.0;
                eventRecord(startAprod1Astro,0);
                    if(nAstroPSolved) aprod1_Kernel_astro(knownTerms_dev, sysmatAstro_dev, vVect_dev, matrixIndexAstro_dev, offLocalAstro, mapNoss, nAstroPSolved);            
                eventRecord(stopAprod1Astro,0);
                eventSynchronize(stopAprod1Astro);
                eventElapsedTime(&milliseconds, startAprod1Astro, stopAprod1Astro);
                timekernel[0]+=milliseconds;

                milliseconds = 0.0;
                eventRecord(startAprod1Att,0);
                    if(nAttP) aprod1_Kernel_att_AttAxis(knownTerms_dev,sysmatAtt_dev,vVect_dev,matrixIndexAtt_dev,nAttP,mapNoss,nDegFreedomAtt,offLocalAtt,nAttParAxis);
                eventRecord(stopAprod1Att,0);
                eventSynchronize(stopAprod1Att);
                eventElapsedTime(&milliseconds, startAprod1Att, stopAprod1Att);
                timekernel[1]+=milliseconds;

                milliseconds = 0.0;
                eventRecord(startAprod1Instr,0);
                    if(nInstrPSolved) aprod1_Kernel_instr(knownTerms_dev, sysmatInstr_dev, vVect_dev, instrCol_dev, mapNoss, offLocalInstr, nInstrPSolved);
                eventRecord(stopAprod1Instr,0);
                eventSynchronize(stopAprod1Instr);
                eventElapsedTime(&milliseconds, startAprod1Instr, stopAprod1Instr);
                timekernel[2]+=milliseconds;

                if(nGlobP) aprod1_Kernel_glob(knownTerms_dev, sysmatGloB_dev, vVect_dev, offLocalGlob, mapNoss, nGlobP);
                // //////////////////////////////////// CONSTRAINTS APROD MODE 1        
                if(nEqExtConstr) aprod1_Kernel_ExtConstr(knownTerms_dev,sysmatConstr_dev,vVect_dev,VrIdAstroPDimMax,mapNoss,nDegFreedomAtt,startingAttColExtConstr,nEqExtConstr,nOfElextObs,numOfExtStar,numOfExtAttCol,nAstroPSolved,nAttAxes);
                if(nEqBarConstr) aprod1_Kernel_BarConstr(knownTerms_dev, sysmatConstr_dev, vVect_dev, nOfElextObs, nOfElBarObs, nEqExtConstr, mapNoss, nEqBarConstr, numOfBarStar, nAstroPSolved );
                if(nOfInstrConstr) aprod1_Kernel_InstrConstr(knownTerms_dev,sysmatConstr_dev,vVect_dev,instrConstrIlung_dev,instrCol_dev,VrIdAstroPDimMax,mapNoss,nDegFreedomAtt,nOfElextObs,nEqExtConstr,nOfElBarObs,nEqBarConstr,myid,nOfInstrConstr,nproc,nAstroPSolved,nAttAxes,nInstrPSolved);

            #else
                if(nAstroPSolved) aprod1_Kernel_astro(knownTerms_dev, sysmatAstro_dev, vVect_dev, matrixIndexAstro_dev, offLocalAstro, mapNoss, nAstroPSolved);            
                if(nAttP) aprod1_Kernel_att_AttAxis(knownTerms_dev,sysmatAtt_dev,vVect_dev,matrixIndexAtt_dev,nAttP,mapNoss,nDegFreedomAtt,offLocalAtt,nAttParAxis);
                if(nInstrPSolved) aprod1_Kernel_instr(knownTerms_dev, sysmatInstr_dev, vVect_dev, instrCol_dev, mapNoss, offLocalInstr, nInstrPSolved);
                if(nGlobP) aprod1_Kernel_glob(knownTerms_dev, sysmatGloB_dev, vVect_dev, offLocalGlob, mapNoss, nGlobP);
                // //////////////////////////////////// CONSTRAINTS APROD MODE 1        
                if(nEqExtConstr) aprod1_Kernel_ExtConstr(knownTerms_dev,sysmatConstr_dev,vVect_dev,VrIdAstroPDimMax,mapNoss,nDegFreedomAtt,startingAttColExtConstr,nEqExtConstr,nOfElextObs,numOfExtStar,numOfExtAttCol,nAstroPSolved,nAttAxes);
                if(nEqBarConstr) aprod1_Kernel_BarConstr(knownTerms_dev, sysmatConstr_dev, vVect_dev, nOfElextObs, nOfElBarObs, nEqExtConstr, mapNoss, nEqBarConstr, numOfBarStar, nAstroPSolved );
                if(nOfInstrConstr) aprod1_Kernel_InstrConstr(knownTerms_dev,sysmatConstr_dev,vVect_dev,instrConstrIlung_dev,instrCol_dev,VrIdAstroPDimMax,mapNoss,nDegFreedomAtt,nOfElextObs,nEqExtConstr,nOfElBarObs,nEqBarConstr,myid,nOfInstrConstr,nproc,nAstroPSolved,nAttAxes,nInstrPSolved);
            #endif


            #ifdef USE_MPI
                // Copy data from device to host (range)
                Kokkos::deep_copy(knownTerms_host_subview, knownTerms_dev_subview);
                // Perform MPI_Allreduce on the full array (not just a scalar)
                starttime = get_time();
                MPI_Allreduce(MPI_IN_PLACE, knownTerms_host_subview.data(), nEqExtConstr + nEqBarConstr + nOfInstrConstr, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
                endtime=get_time();
                communicationtime=compute_time(endtime,starttime);
                // Copy the data back from host to device (range)
                Kokkos::deep_copy(knownTerms_dev_subview, knownTerms_host_subview);
            #endif

            kauxsum(knownTerms_dev,kAuxcopy_dev,mapNoss,nEqExtConstr+nEqBarConstr+nOfInstrConstr);

            max_knownTerms = maxCommMultiBlock_double(knownTerms_dev,nElemKnownTerms);

            if(!myid){
                ssq_knownTerms = sumCommMultiBlock_double(knownTerms_dev, mapNoss + nEqExtConstr + nEqBarConstr+nOfInstrConstr, max_knownTerms);
                betaLoc = max_knownTerms*std::sqrt(ssq_knownTerms);
                betaLoc2 = betaLoc*betaLoc;
            }else{
                ssq_knownTerms = sumCommMultiBlock_double(knownTerms_dev, mapNoss, max_knownTerms);
                betaLoc = max_knownTerms*std::sqrt(ssq_knownTerms);
                betaLoc2 = betaLoc*betaLoc;
            }

            #ifdef USE_MPI
                //------------------------------------------------------------------------------------------------
                starttime=get_time();
                    MPI_Allreduce(MPI_IN_PLACE,&betaLoc2,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
                endtime=get_time();
                communicationtime=compute_time(endtime,starttime);
                //------------------------------------------------------------------------------------------------
            #endif

            beta=sqrt(betaLoc2);
            //  Accumulate  anorm = || Bk || =  sqrt( sum of  alpha**2 + beta**2 + damp**2 ).
            temp   =   d2norm( alpha, beta );
            temp   =   d2norm( temp , damp );
            anorm  =   d2norm( anorm, temp );



            if (beta > ZERO) {

                dscal(knownTerms_dev,1/beta, nElemKnownTerms,ONE);
                dscal(vVect_dev,beta,nunkSplit,MONE);
                vAuxVect_Kernel(vVect_dev, vAuxVect_dev, localAstroMax);

                if (myid) {
                    vVect_Put_To_Zero_Kernel(vVect_dev,localAstroMax,nunkSplit);
                }

                #ifdef KERNELTIME
                    milliseconds = 0.0;
                    eventRecord(startAprod2Astro,0);
                        if(nAstroPSolved) aprod2_Kernel_astro(vVect_dev, sysmatAstro_dev, knownTerms_dev, matrixIndexAstro_dev, startend_dev, offLocalAstro, nnz, nAstroPSolved);
                    eventRecord(stopAprod2Astro,0);
                    eventSynchronize(stopAprod2Astro);
                    eventElapsedTime(&milliseconds, startAprod2Astro, stopAprod2Astro);
                    timekernel[3]+=milliseconds;

                    milliseconds = 0.0;
                    eventRecord(startAprod2Att,0);
                        if(nAttP) aprod2_Kernel_att_AttAxis(vVect_dev,sysmatAtt_dev,knownTerms_dev,matrixIndexAtt_dev,nAttP,nDegFreedomAtt,offLocalAtt,mapNoss,nAstroPSolved,nAttParAxis);
                    eventRecord(stopAprod2Att,0);
                    eventSynchronize(stopAprod2Att);
                    eventElapsedTime(&milliseconds, startAprod2Att, stopAprod2Att);
                    timekernel[4]+=milliseconds;

                    milliseconds = 0.0;
                    eventRecord(startAprod2Instr,0);
                        if(nInstrPSolved) aprod2_Kernel_instr(vVect_dev,sysmatInstr_dev, knownTerms_dev, instrCol_dev, offLocalInstr, mapNoss, nInstrPSolved);
                    eventRecord(stopAprod2Instr,0);
                    eventSynchronize(stopAprod2Instr);
                    eventElapsedTime(&milliseconds, startAprod2Instr, stopAprod2Instr);
                    timekernel[5]+=milliseconds;

                    // //////////////////////////////////// CONSTRAi_intS APROD MODE 2        
                    if(nEqExtConstr) aprod2_Kernel_ExtConstr(vVect_dev,sysmatConstr_dev,knownTerms_dev,mapNoss,nDegFreedomAtt,VrIdAstroPDimMax,nEqExtConstr,nOfElextObs,numOfExtStar,startingAttColExtConstr,numOfExtAttCol,nAstroPSolved,nAttAxes);
                    if(nEqBarConstr) aprod2_Kernel_BarConstr(vVect_dev,sysmatConstr_dev,knownTerms_dev,mapNoss,nEqBarConstr,nEqExtConstr,nOfElextObs,nOfElBarObs,numOfBarStar,nAstroPSolved);
                    if(nOfInstrConstr) aprod2_Kernel_InstrConstr(vVect_dev, sysmatConstr_dev, knownTerms_dev, instrConstrIlung_dev, instrCol_dev,  VrIdAstroPDimMax, nDegFreedomAtt, mapNoss, nEqExtConstr, nEqBarConstr, nOfElextObs, nOfElBarObs, myid, nOfInstrConstr, nproc, nInstrPSolved, nAstroPSolved, nAttAxes);


                #else
                    //APROD2 CALL BEFORE LSQR
                    if(nAstroPSolved) aprod2_Kernel_astro(vVect_dev, sysmatAstro_dev, knownTerms_dev, matrixIndexAstro_dev, startend_dev, offLocalAstro, nnz, nAstroPSolved);
                    if(nAttP) aprod2_Kernel_att_AttAxis(vVect_dev,sysmatAtt_dev,knownTerms_dev,matrixIndexAtt_dev,nAttP,nDegFreedomAtt,offLocalAtt,mapNoss,nAstroPSolved,nAttParAxis);
                    if(nInstrPSolved) aprod2_Kernel_instr(vVect_dev,sysmatInstr_dev, knownTerms_dev, instrCol_dev, offLocalInstr, mapNoss, nInstrPSolved);
                    // NOT OPTIMIZED YET
                    if(nEqExtConstr) aprod2_Kernel_ExtConstr(vVect_dev,sysmatConstr_dev,knownTerms_dev,mapNoss,nDegFreedomAtt,VrIdAstroPDimMax,nEqExtConstr,nOfElextObs,numOfExtStar,startingAttColExtConstr,numOfExtAttCol,nAstroPSolved,nAttAxes);
                    if(nEqBarConstr) aprod2_Kernel_BarConstr(vVect_dev,sysmatConstr_dev,knownTerms_dev,mapNoss,nEqBarConstr,nEqExtConstr,nOfElextObs,nOfElBarObs,numOfBarStar,nAstroPSolved);
                    if(nOfInstrConstr) aprod2_Kernel_InstrConstr(vVect_dev, sysmatConstr_dev, knownTerms_dev, instrConstrIlung_dev, instrCol_dev,  VrIdAstroPDimMax, nDegFreedomAtt, mapNoss, nEqExtConstr, nEqBarConstr, nOfElextObs, nOfElBarObs, myid, nOfInstrConstr, nproc, nInstrPSolved, nAstroPSolved, nAttAxes);

                #endif


                #ifdef USE_MPI
                    Kokkos::deep_copy(Kokkos::View<double*, Kokkos::HostSpace>(vVect, nunkSplit), vVect_dev);
                    //------------------------------------------------------------------------------------------------  TIME4
                    starttime=get_time();
                        MPI_Allreduce(MPI_IN_PLACE,&vVect[localAstroMax], nAttParam+nInstrParam+nGlobalParam,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
                    if(nAstroPSolved) SumCirc(vVect,comlsqr);
                    endtime=get_time();
                    communicationtime=compute_time(endtime,starttime);
                    //------------------------------------------------------------------------------------------------
                    /* copy from the host to the device: */
                    Kokkos::deep_copy(vVect_dev, Kokkos::View<double*, Kokkos::HostSpace>(vVect, nunkSplit));
                #endif

    
                vaux_sum(vVect_dev,vAuxVect_dev,localAstroMax);


                max_vVect = maxCommMultiBlock_double(vVect_dev, nunkSplit);
                ssq_vVect = sumCommMultiBlock_double(vVect_dev, nAstroElements*nAstroPSolved, max_vVect);

                double alphaLoc = 0.0;
                alphaLoc = max_vVect*sqrt(ssq_vVect);
                alphaLoc2=alphaLoc*alphaLoc;
            
                if(!myid){
                    double alphaOther2 = alphaLoc2;
                    ssq_vVect = sumCommMultiBlock_double_start(vVect_dev, nunkSplit - localAstroMax, max_vVect,localAstroMax);
                    alphaLoc = max_vVect*std::sqrt(ssq_vVect);
                    alphaLoc2 = alphaLoc*alphaLoc;
                    alphaLoc2 = alphaOther2 + alphaLoc2;
                }


                #ifdef USE_MPI
                    //------------------------------------------------------------------------------------------------
                    starttime=get_time();
                        MPI_Allreduce(MPI_IN_PLACE,&alphaLoc2,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
                    endtime=get_time();
                    communicationtime=compute_time(endtime,starttime);
                    //------------------------------------------------------------------------------------------------
                #endif

                alpha=std::sqrt(alphaLoc2);
                        
                if (alpha > ZERO) {
                    dscal(vVect_dev,1/alpha,nunkSplit,ONE);
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



            Kokkos::parallel_reduce("compute_dknorm", nAstroElements * nAstroPSolved, KOKKOS_LAMBDA(const long i, double& lsum) {
                double t = wVect_dev(i);
                t = (t3 * t) * (t3 * t);
                lsum += t;
            }, dknorm);

            #ifdef USE_MPI
                //------------------------------------------------------------------------------------------------
                starttime=get_time();
                    MPI_Allreduce(MPI_IN_PLACE,&dknorm,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
                endtime=get_time();
                communicationtime=compute_time(endtime,starttime);
                //------------------------------------------------------------------------------------------------ 		
            #endif

            // First transformation
            Kokkos::parallel_for("transform_wVect_xSolution", localAstro, KOKKOS_LAMBDA(const int i) {
                xSolution_dev(i) = t1 * wVect_dev(i) + xSolution_dev(i);
            });

            if(wantse) {
                Kokkos::parallel_for("transform_wVect_standardError", localAstro, KOKKOS_LAMBDA(const int i) {
                    standardError_dev(i) = standardError_dev(i) + (t3 * wVect_dev(i)) * (t3 * wVect_dev(i));
                });
            }

            // Third transformation
            Kokkos::parallel_for("transform_vVect_wVect", localAstro, KOKKOS_LAMBDA(const int i) {
                wVect_dev(i) = vVect_dev(i) + t2 * wVect_dev(i);
            });

            // Fourth transformation for range [localAstroMax, localAstroMax + other)
            Kokkos::parallel_for("transform_wVect_xSolution_max", other, KOKKOS_LAMBDA(const int i) {
                const int index = localAstroMax + i;
                xSolution_dev(index) = t1 * wVect_dev(index) + xSolution_dev(index);
            });

            if(wantse) {
                Kokkos::parallel_for("transform_wVect_standardError_max", other, KOKKOS_LAMBDA(const int i) {
                    const int index = localAstroMax + i;
                    standardError_dev(index) = standardError_dev(index) + (t3 * wVect_dev(index)) * (t3 * wVect_dev(index));
                });
            }

            // Sixth transformation for range [localAstroMax, localAstroMax + other)
            Kokkos::parallel_for("transform_vVect_wVect_max", other, KOKKOS_LAMBDA(const int i) {
                const int index = localAstroMax + i;
                wVect_dev(index) = vVect_dev(index) + t2 * wVect_dev(index);
            });

            Kokkos::parallel_reduce("transform_reduce_dknorm", other, KOKKOS_LAMBDA(const int i, double& lsum) {
                const int index = localAstroMax + i;
                lsum += (t3 * wVect_dev(index)) * (t3 * wVect_dev(index));
            }, dknorm);

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


            if (test3 <= ctol) istop = 4;
            if (test2 <= atol) istop = 2;
            if (test1 <= rtol) istop = 1;   //(Michael Friedlander had this commented out)


        //------------------------------------------------------------------------------------------------
        #ifdef USE_MPI
            endCycleTime=get_time(startCycleTime);
            totTimeIteration+=endCycleTime;
            if(myid==0) printf("lsqr: Iteration number %d. Iteration seconds %lf. Global Seconds %lf \n",itn,endCycleTime,totTimeIteration);
        #elif defined(KERNELTIME)
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
            printf("Average kernel Aprod1Astro time: %lf \n", 1e-3*timekernel[0]/itn);
            printf("Average kernel Aprod1Att time: %lf \n", 1e-3*timekernel[1]/itn);
            printf("Average kernel Aprod1Instr time: %lf \n", 1e-3*timekernel[2]/itn);
            printf("Average kernel Aprod2Astro time: %lf \n", 1e-3*timekernel[3]/itn);
            printf("Average kernel Aprod2Att time: %lf \n", 1e-3*timekernel[4]/itn);
            printf("Average kernel Aprod2Instr time: %lf \n", 1e-3*timekernel[5]/itn);
        #endif
        //------------------------------------------------------------------------------------------------ 		


        Kokkos::deep_copy(Kokkos::View<double*, Kokkos::HostSpace>(xSolution, nunkSplit), xSolution_dev);
        Kokkos::deep_copy(Kokkos::View<double*, Kokkos::HostSpace>(standardError, nunkSplit), standardError_dev);

        if ( wantse ) {
            t    =   ONE;
            if (m > n)     t = m - n;
            if ( damped )  t = m;
            t    =   rnorm / sqrt( t );
        
            for (long i = 0; i < nunkSplit; i++)
                standardError[i]  = t * sqrt( standardError[i] );
            
        }





        //  Assign output variables from local copies.
        *istop_out  = istop;
        *itn_out    = itn;
        *anorm_out  = anorm;
        *acond_out  = acond;
        *rnorm_out  = rnorm;
        *arnorm_out = test2;
        *xnorm_out  = xnorm;


    }
    Kokkos::finalize();

    return;
}


