#ifndef __UTIL_H
#define __UTIL_H
#define DEFAULT_TIMECPR  1380
#define DEFAULT_TIMELIMIT  7200
#define DEFAULT_ITNCPR  2000
#define DEFAULT_ITNLIMIT  4000 
#define DEFAULT_NATTPARAXIS 4
#define DEFAULT_NINSTRINDEXES 4  // Unconsistent model if this figure is changed
#define DEFAULT_NINSTRVALUES 6  // Unconsistent model if this figure is changed
#define DEFAULT_NASTROP 5
#define DEFAULT_NCOLSFITS 17 // present number of columns in the files GsrSystemRow*.fits
#define DEFAULT_EXTCONSTROWS 6 // number of extended contraints rows
#define DEFAULT_BARCONSTROWS 0 //6 // number of extended contraints rows
// define per inserimento funzione ran2
#define IM1 2147483563
#define IM2 2147483399
#define AM (1.0/IM1)
#define IMM1 (IM1-1)
#define IA1 40014
#define IA2 40692
#define IQ1 53668
#define IQ2 52774
#define IR1 12211
#define IR2 3791
#define NTAB 32
#define NDIV (1+IMM1/NTAB)
#define EPS 1.2e-7
#define RNMX (1.0-EPS)
#define VER "8.0"
//

#include <dirent.h>
#include <string.h>
#include <unistd.h>
#include <time.h>
#include "cblas.h"
#include <algorithm>

#include <chrono>

#include <fstream> // Required for std::ofstream


#ifdef USE_MPI
    #include <mpi.h>
#endif

#define MIN(a,b) (((a)<(b))?(a):(b))
#define MAX(a,b) (((a)>(b))?(a):(b))


#ifdef USE_MPI
    using time_point =float;
    
    inline float get_time(){
        MPI_Barrier(MPI_COMM_WORLD);
        return MPI_Wtime();
    }


    template<typename T>
    inline float get_time(const T& t){
        MPI_Barrier(MPI_COMM_WORLD);
        return MPI_Wtime()-t;
    }

    template<typename S>
    inline float compute_time(const S& end, const S& start){
        return end-start;
    }
#else
    using time_point = std::chrono::time_point<std::chrono::high_resolution_clock>;

    inline time_point get_time(){
        return  std::chrono::high_resolution_clock::now();
    }

    inline time_point get_time(const time_point& t) {
        auto now = std::chrono::high_resolution_clock::now(); // Get current time
        auto elapsed = now - t;
        return now; 
    }

    template<typename S>
    inline float compute_time(const S& end, const S& start){
        return std::chrono::duration<float>(end-start).count();
    }
#endif

#ifdef KERNELTIME
    #if defined(__NVIDIA90__) || defined(__NVIDIA80__) || defined(__NVIDIA70__)
        #include <cuda_runtime.h>    // CUDA runtime for NVIDIA GPUs
        #include <cuda.h>            // Optional: CUDA driver API
        
        using event = cudaEvent_t;
        inline void eventCreate(event* ev){
            cudaEventCreate(ev);
        }
        inline void eventRecord(cudaEvent_t event, cudaStream_t stream = 0){
            cudaEventRecord(event,stream);
        }
        inline void eventSynchronize(cudaEvent_t event){
            cudaEventSynchronize(event);
        }
        inline void eventElapsedTime(float* ms, cudaEvent_t start, cudaEvent_t end){
            cudaEventElapsedTime(ms, start, end);
        }

    #elif defined(__MI250X__) || defined(__MI100X__)
        #include <hip/hip_runtime.h>  // HIP runtime for AMD GPUs
        #include <hip/hip_runtime_api.h> // Optional: HIP runtime API

        using event = hipEvent_t;
        inline void eventCreate(event* ev){
            hipEventCreate(ev);
        }
        inline void eventRecord(hipEvent_t event, hipStream_t stream = 0){
            hipEventRecord(event,stream);
        }
        inline void eventSynchronize(hipEvent_t event){
            hipEventSynchronize(event);
        }
        inline void eventElapsedTime(float* ms, hipEvent_t start, hipEvent_t end){
            hipEventElapsedTime(ms, start, end);
        }

    #else
        #error "Unknown platform"
    #endif

#endif


/* Called in an error during malloc occurs. Returns 1. */
int err_malloc(const char *s, int id);



struct nullSpace {
    double vectNorm[6];
    double compMin[6];
    double compMax[6];
    double compVar[6];
    double compAvg[6];
    
};
struct comData {
	int myid;
	int nproc;
	long int * mapNoss;
	long int * mapNcoeff;
	long int mapNossBefore, mapNossAfter;
	long int nvinc;
	long int nobs;
	long int parOss;   
	long nunk, nunkSplit;
	long nDegFreedomAtt;
	long offsetAttParam,offsetInstrParam , offsetGlobParam, VroffsetAttParam; 
	long nStar;
    short nAstroP, nAttP, nInstrP, nGlobP, nAstroPSolved, nInstrPSolved; // number of astrometric, attitude, instrument,
    int lsInstrFlag,ssInstrFlag,nuInstrFlag,maInstrFlag;
    int nElemIC,nOfInstrConstr;
    long cCDLSAACZP;
    int nElemICSS;
    int nElemICLSAL;
    int nElemICLSAC;
    short nAttAxes, nAttParAxis;
	int nAttParam,nInstrParam,nGlobalParam;
	int **mapStar;
	long VrIdAstroPDim;
	long VrIdAstroPDimMax;
	int *constrPE;
	int setBound[4];
	int instrConst[4];  // instrConst[0]=nFovs instrConst[1]= nCCDs instrConst[2]=nPixelColumns instrConst[3]=nTimeIntervals
	int timeCPR, timeLimit, itnCPR,itnCPRstop,itnCPRend, itnLimit,itn,noCPR;
    long offsetCMag,offsetCnu,offsetCdelta_eta,offsetCDelta_eta_1,offsetCDelta_eta_2;
    long offsetCDelta_eta_3,offsetCdelta_zeta,offsetCDelta_zeta_1,offsetCDelta_zeta_2;
	int nthreads;
	long **mapForThread;
	int nSubsetAtt, nSubsetInstr;
	int NOnSubsetAtt, NOnSubsetInstr;
	int Test;
	// int multMI;
	// int debugMode;
    int extConstraint,nEqExtConstr,numOfExtStar,numOfExtAttCol,startingAttColExtConstr;
    int barConstraint,nEqBarConstr,numOfBarStar;
    long lastStarConstr;
    long firstStarConstr;
    int nOfElextObs;
    int nOfElBarObs;
    int noCov;
    long EleCov, ElewVect;
    char *outputDir;
    double nullSpaceAttfact,extConstrW,barConstrW;
    time_t totSec;
    long *covStarStar,*covStarOther, *covOtherOther, *covUnresolved;
    int covStarStarCounter, covStarOtherCounter, covOtherOtherCounter, covUnresolvedCounter;
    double *coVariance;
};
 	
 


void SumCirc(double *vectToSum, struct comData comlsqr);

void SumCirc2(double *vectToSum,struct comData comlsqr, double* communicationtime);

void initThread(struct comData *comlsqr);


void precondSystemMatrix(double *sysmatAstro,double *sysmatAtt,double *sysmatInstr,double *sysmatGloB,double *sysmatConstr, double *preCondVect, long  *matrixIndexAstro,long  *matrixIndexAtt,int *instrCol,struct comData comlsqr);


double gauss(double ave, double sigma, long init2);


double ran2(long *idum);



struct nullSpace cknullSpace(double *sysmatAstro,double *sysmatAtt,double *sysmatInstr,double *sysmatGloB,double *sysmatConstr,long* matrixIndexAstro,long * matrixIndexAtt,double *attNS,struct comData  comlsqr);

double legendre(int deg, double x);
int computeInstrConstr (struct comData comlsqr,double * instrCoeffConstr,int * instrColsConstr,int * instrConstrIlung);
float simfullram(long *nStar, long *nobs, float memGlobal, int nparams, int nAttParam, int nInstrParam);



#endif
