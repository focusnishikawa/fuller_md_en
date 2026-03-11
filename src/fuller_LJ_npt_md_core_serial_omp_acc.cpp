// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2025, Takeshi Nishikawa
/*===========================================================================
  fuller_LJ_npt_md_core_serial_omp_acc.cpp
  C60 Fullerene Crystal NPT Molecular Dynamics Simulation
  (LJ Rigid-body Model, Core Version — Serial / OpenMP / OpenACC GPU Unified)

  Compilation:
    Serial (Single-threaded):
      g++ -std=c++17 -O3 -Wno-unknown-pragmas \
          -o fuller_LJ_core_serial fuller_LJ_npt_md_core_serial_omp_acc.cpp -lm

    OpenMP (Multi-threaded):
      g++ -std=c++17 -O3 -fopenmp -Wno-unknown-pragmas \
          -o fuller_LJ_core_omp fuller_LJ_npt_md_core_serial_omp_acc.cpp -lm

    OpenACC GPU:
      nvc++ -std=c++17 -O3 -acc -gpu=cc80 -Minfo=accel \
          -o fuller_LJ_core_gpu fuller_LJ_npt_md_core_serial_omp_acc.cpp -lm

  Runtime Options:
    ./fuller_LJ_core_serial [nc]
    ./fuller_LJ_core_omp    [nc]
    ./fuller_LJ_core_gpu    [nc]

    nc (integer, default: 3, max: 8)
      Number of FCC unit cell repetitions. Number of molecules N = 4*nc^3.
      nc=3 -> N=108, nc=4 -> N=256, nc=5 -> N=500

    Specification methods:
      ./fuller_LJ_core_serial 3         # positional argument
      ./fuller_LJ_core_serial --cell=5  # keyword argument

  Execution Examples:
    # Serial: default (3x3x3, N=108)
    ./fuller_LJ_core_serial

    # OpenMP: large system
    ./fuller_LJ_core_omp 5

    # OpenACC GPU
    ./fuller_LJ_core_gpu 4

  Fixed Parameters (modify in source code):
    Temperature T = 300 K
    Pressure Pe   = 0.0 GPa
    Time step dt  = 1.0 fs
    Number of steps = 1000
    Output interval = 100 steps
    Neighbor list update = 25 steps

  Features:
    - NPT-MD simulation of C60 rigid-body molecules with LJ intermolecular interaction
    - Nose-Hoover thermostat + Parrinello-Rahman pressure control
    - Rigid-body rotation via quaternions
    - Velocity-Verlet time integration
    - Automatic generation of FCC crystal initial configuration
    - Compile-time switch for Serial/OpenMP/OpenACC selection
    - No restart capability, no OVITO output (core version)

  GPU Optimization Strategy:
    1. forces(): gang parallelism (molecules) x vector parallelism (intra-molecular atoms ai)
       - Symmetric full list: no Newton's 3rd law -> conflict avoidance
       - Vectorize ai atom loop: up to 128 threads for warp parallelism
       - Expand virial 9 components into scalar reduction variables
    2. Data residency: keep arrays on GPU with #pragma acc data
    3. OpenMP mode: multi-threaded with schedule(dynamic) + atomic

  Unit system: A (distance), amu (mass), eV (energy), fs (time), K (temperature), GPa (pressure)
===========================================================================*/

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <random>
#include <chrono>
#include <string>

#ifdef _OPENACC
  #include <openacc.h>
#endif
#ifdef _OPENMP
  #include <omp.h>
#else
  inline int omp_get_max_threads(){ return 1; }
#endif


/* =============== Physical Constants and Unit Conversion =============== */
constexpr double CONV       = 9.64853321e-3;
constexpr double kB         = 8.617333262e-5;
constexpr double eV2GPa     = 160.21766208;
constexpr double eV2kcalmol = 23.06054783;

/* =============== LJ Potential Parameters =============== */
constexpr double sigma_LJ = 3.431;
constexpr double eps_LJ   = 2.635e-3;
constexpr double RCUT      = 3.0*sigma_LJ;
constexpr double RCUT2     = RCUT*RCUT;
constexpr double sig2_LJ   = sigma_LJ*sigma_LJ;
constexpr double mC        = 12.011;

/* VSHFT: constexpr (guaranteed usable on device) */
constexpr double _sr_v  = 1.0/3.0;
constexpr double _sr2_v = _sr_v * _sr_v;
constexpr double _sr6_v = _sr2_v * _sr2_v * _sr2_v;
constexpr double VSHFT  = 4.0*eps_LJ*(_sr6_v*_sr6_v - _sr6_v);

/* =============== Molecular Parameters =============== */
constexpr int C60_NATOM = 60;
constexpr int MAX_NATOM = 84;
constexpr double MC60   = C60_NATOM * mC;
constexpr double RC60   = 3.55;
constexpr double RMCUT   = RCUT + 2*RC60 + 1.0;
constexpr double RMCUT2  = RMCUT*RMCUT;

constexpr int MAX_NEIGH = 80;
constexpr int VECTOR_LENGTH = 128;


/* =============== H-matrix Flat 9-component Operations =============== */
#define H_(h,i,j) ((h)[3*(i)+(j)])

#pragma acc routine seq
static inline double mat_det9(const double* h){
    return H_(h,0,0)*(H_(h,1,1)*H_(h,2,2)-H_(h,1,2)*H_(h,2,1))
          -H_(h,0,1)*(H_(h,1,0)*H_(h,2,2)-H_(h,1,2)*H_(h,2,0))
          +H_(h,0,2)*(H_(h,1,0)*H_(h,2,1)-H_(h,1,1)*H_(h,2,0));
}

#pragma acc routine seq
static inline double mat_tr9(const double* h){
    return H_(h,0,0)+H_(h,1,1)+H_(h,2,2);
}

static void mat_inv9(const double* h, double* hi){
    double d=mat_det9(h), id=1.0/d;
    hi[0]=id*(H_(h,1,1)*H_(h,2,2)-H_(h,1,2)*H_(h,2,1));
    hi[1]=id*(H_(h,0,2)*H_(h,2,1)-H_(h,0,1)*H_(h,2,2));
    hi[2]=id*(H_(h,0,1)*H_(h,1,2)-H_(h,0,2)*H_(h,1,1));
    hi[3]=id*(H_(h,1,2)*H_(h,2,0)-H_(h,1,0)*H_(h,2,2));
    hi[4]=id*(H_(h,0,0)*H_(h,2,2)-H_(h,0,2)*H_(h,2,0));
    hi[5]=id*(H_(h,0,2)*H_(h,1,0)-H_(h,0,0)*H_(h,1,2));
    hi[6]=id*(H_(h,1,0)*H_(h,2,1)-H_(h,1,1)*H_(h,2,0));
    hi[7]=id*(H_(h,0,1)*H_(h,2,0)-H_(h,0,0)*H_(h,2,1));
    hi[8]=id*(H_(h,0,0)*H_(h,1,1)-H_(h,0,1)*H_(h,1,0));
}


/* =============== Minimum Image Convention (Device Function) =============== */
#pragma acc routine seq
static inline void mimg_flat(double &dx, double &dy, double &dz,
                             const double* hi, const double* h){
    double s0=hi[0]*dx+hi[1]*dy+hi[2]*dz;
    double s1=hi[3]*dx+hi[4]*dy+hi[5]*dz;
    double s2=hi[6]*dx+hi[7]*dy+hi[8]*dz;
    s0-=round(s0); s1-=round(s1); s2-=round(s2);
    dx=h[0]*s0+h[1]*s1+h[2]*s2;
    dy=h[3]*s0+h[4]*s1+h[5]*s2;
    dz=h[6]*s0+h[7]*s1+h[8]*s2;
}


/* =============== Quaternion Operations (Device Functions) =============== */

#pragma acc routine seq
static inline void q2R_flat(const double* q, double* R){
    double w=q[0],x=q[1],y=q[2],z=q[3];
    R[0]=1-2*(y*y+z*z); R[1]=2*(x*y-w*z);   R[2]=2*(x*z+w*y);
    R[3]=2*(x*y+w*z);   R[4]=1-2*(x*x+z*z); R[5]=2*(y*z-w*x);
    R[6]=2*(x*z-w*y);   R[7]=2*(y*z+w*x);   R[8]=1-2*(x*x+y*y);
}

#pragma acc routine seq
static inline void qmul_flat(const double* a, const double* b, double* out){
    out[0]=a[0]*b[0]-a[1]*b[1]-a[2]*b[2]-a[3]*b[3];
    out[1]=a[0]*b[1]+a[1]*b[0]+a[2]*b[3]-a[3]*b[2];
    out[2]=a[0]*b[2]-a[1]*b[3]+a[2]*b[0]+a[3]*b[1];
    out[3]=a[0]*b[3]+a[1]*b[2]-a[2]*b[1]+a[3]*b[0];
}

#pragma acc routine seq
static inline void qnorm_flat(double* q){
    double n=sqrt(q[0]*q[0]+q[1]*q[1]+q[2]*q[2]+q[3]*q[3]);
    double inv=1.0/n;
    q[0]*=inv; q[1]*=inv; q[2]*=inv; q[3]*=inv;
}

#pragma acc routine seq
static inline void omega2dq_flat(double wx, double wy, double wz,
                                 double dt, double* dq){
    double wm=sqrt(wx*wx+wy*wy+wz*wz);
    double th=wm*dt*0.5;
    if(th<1e-14){
        dq[0]=1.0; dq[1]=0.5*dt*wx; dq[2]=0.5*dt*wy; dq[3]=0.5*dt*wz;
    } else {
        double s=sin(th)/wm;
        dq[0]=cos(th); dq[1]=s*wx; dq[2]=s*wy; dq[3]=s*wz;
    }
}


/* =============== C60 Coordinate Generation (Host) =============== */
struct C60Data { double coords[60*3]; double I0, Mmol, Rmol; };

static C60Data generate_c60(){
    C60Data d; d.Mmol=MC60;
    double phi=(1.0+sqrt(5.0))/2.0;
    int n=0, cyc[3][3]={{0,1,2},{1,2,0},{2,0,1}};
    double tmp[60][3]={};
    for(int p=0;p<3;p++) for(int s2:{-1,1}) for(int s3:{-1,1}){
        tmp[n][cyc[p][1]]=s2; tmp[n][cyc[p][2]]=s3*3*phi; n++; }
    for(int p=0;p<3;p++) for(int s1:{-1,1}) for(int s2:{-1,1}) for(int s3:{-1,1}){
        tmp[n][cyc[p][0]]=s1*2; tmp[n][cyc[p][1]]=s2*(1+2*phi); tmp[n][cyc[p][2]]=s3*phi; n++; }
    for(int p=0;p<3;p++) for(int s1:{-1,1}) for(int s2:{-1,1}) for(int s3:{-1,1}){
        tmp[n][cyc[p][0]]=s1; tmp[n][cyc[p][1]]=s2*(2+phi); tmp[n][cyc[p][2]]=s3*2*phi; n++; }
    double cm[3]={0,0,0};
    for(int i=0;i<60;i++) for(int a=0;a<3;a++) cm[a]+=tmp[i][a];
    for(int a=0;a<3;a++) cm[a]/=60.0;
    for(int i=0;i<60;i++) for(int a=0;a<3;a++) tmp[i][a]=(tmp[i][a]-cm[a])*0.72;
    d.Rmol=0; double Isum=0;
    for(int i=0;i<60;i++){
        double r2=tmp[i][0]*tmp[i][0]+tmp[i][1]*tmp[i][1]+tmp[i][2]*tmp[i][2];
        double r=sqrt(r2); if(r>d.Rmol) d.Rmol=r;
        Isum+=mC*r2;
    }
    d.I0=Isum*2.0/3.0;
    for(int i=0;i<60;i++) for(int a=0;a<3;a++) d.coords[i*3+a]=tmp[i][a];
    return d;
}


/* =============== FCC Crystal Generation (Host) =============== */
static int make_fcc(double a, int nc, double* pos, double* h){
    double bas[4][3]={{0,0,0},{.5*a,.5*a,0},{.5*a,0,.5*a},{0,.5*a,.5*a}};
    int n=0;
    for(int ix=0;ix<nc;ix++) for(int iy=0;iy<nc;iy++) for(int iz=0;iz<nc;iz++)
        for(int b=0;b<4;b++){
            pos[n*3+0]=a*ix+bas[b][0];
            pos[n*3+1]=a*iy+bas[b][1];
            pos[n*3+2]=a*iz+bas[b][2];
            n++;
        }
    for(int i=0;i<9;i++) h[i]=0.0;
    h[0]=h[4]=h[8]=nc*a;
    return n;
}


/* =============== Neighbor List Construction (Host, Symmetric Full List) =============== */
static void nlist_build_sym(const double* pos, const double* h, const double* hi,
                            int N, double rmcut, int* nl_count, int* nl_list){
    double rc2=(rmcut+3.0)*(rmcut+3.0);
    for(int i=0;i<N;i++) nl_count[i]=0;

    for(int i=0;i<N;i++){
        for(int j=i+1;j<N;j++){
            double dx=pos[j*3]-pos[i*3];
            double dy=pos[j*3+1]-pos[i*3+1];
            double dz=pos[j*3+2]-pos[i*3+2];
            mimg_flat(dx,dy,dz,hi,h);
            double r2=dx*dx+dy*dy+dz*dz;
            if(r2<rc2){
                int ci=nl_count[i], cj=nl_count[j];
                if(ci<MAX_NEIGH){ nl_list[i*MAX_NEIGH+ci]=j; nl_count[i]++; }
                else { printf("WARNING: nl overflow mol %d (count=%d)\n",i,ci); }
                if(cj<MAX_NEIGH){ nl_list[j*MAX_NEIGH+cj]=i; nl_count[j]++; }
                else { printf("WARNING: nl overflow mol %d (count=%d)\n",j,cj); }
            }
        }
    }
}


/* =============== Apply PBC =============== */
static void apply_pbc(double* pos, const double* h, const double* hi, int N){
#ifdef _OPENACC
    #pragma acc parallel loop present(pos,h,hi)
#elif defined(_OPENMP)
    #pragma omp parallel for schedule(static)
#endif
    for(int i=0;i<N;i++){
        double px=pos[i*3], py=pos[i*3+1], pz=pos[i*3+2];
        double s0=hi[0]*px+hi[1]*py+hi[2]*pz;
        double s1=hi[3]*px+hi[4]*py+hi[5]*pz;
        double s2=hi[6]*px+hi[7]*py+hi[8]*pz;
        s0-=floor(s0); s1-=floor(s1); s2-=floor(s2);
        pos[i*3  ]=h[0]*s0+h[1]*s1+h[2]*s2;
        pos[i*3+1]=h[3]*s0+h[4]*s1+h[5]*s2;
        pos[i*3+2]=h[6]*s0+h[7]*s1+h[8]*s2;
    }
}


/* =============== Force, Torque, and Virial Calculation (Main Kernel) =============== */
static double forces(double* Fv, double* Tv, double* Wm9,
                     const double* pos, const double* qv,
                     const double* body, const double* h, const double* hi,
                     const int* nl_count, const int* nl_list,
                     int N, int natom, double rmcut2,
                     double* lab)
{
    /* --- Compute lab-frame coordinates --- */
#ifdef _OPENACC
    #pragma acc parallel loop gang vector_length(VECTOR_LENGTH) present(qv,body,lab)
#elif defined(_OPENMP)
    #pragma omp parallel for schedule(static)
#endif
    for(int i=0;i<N;i++){
        double R[9];
        q2R_flat(&qv[i*4], R);
#ifdef _OPENACC
        #pragma acc loop vector
#endif
        for(int a=0;a<natom;a++){
            double bx=body[a*3], by=body[a*3+1], bz=body[a*3+2];
            int idx=i*natom*3+a*3;
            lab[idx  ]=R[0]*bx+R[1]*by+R[2]*bz;
            lab[idx+1]=R[3]*bx+R[4]*by+R[5]*bz;
            lab[idx+2]=R[6]*bx+R[7]*by+R[8]*bz;
        }
    }

    /* --- Zero initialization --- */
#ifdef _OPENACC
    #pragma acc parallel loop present(Fv,Tv)
#elif defined(_OPENMP)
    #pragma omp parallel for schedule(static)
#endif
    for(int i=0;i<N*3;i++){ Fv[i]=0.0; Tv[i]=0.0; }
#ifdef _OPENACC
    #pragma acc parallel loop present(Wm9)
#elif defined(_OPENMP)
    #pragma omp parallel for schedule(static)
#endif
    for(int i=0;i<9;i++) Wm9[i]=0.0;

    /* --- LJ force calculation main kernel --- */
    double Ep=0.0;

#ifdef _OPENACC
    #pragma acc parallel loop gang vector_length(VECTOR_LENGTH) \
        present(pos,lab,Fv,Tv,Wm9,h,hi,nl_count,nl_list) \
        reduction(+:Ep)
#elif defined(_OPENMP)
    #pragma omp parallel for schedule(dynamic,1) reduction(+:Ep)
#endif
    for(int i=0;i<N;i++){
        double fi0=0, fi1=0, fi2=0;
        double ti0=0, ti1=0, ti2=0;
        double my_Ep=0;
        double w00=0,w01=0,w02=0, w10=0,w11=0,w12=0, w20=0,w21=0,w22=0;

        int nni=nl_count[i];
        for(int k=0;k<nni;k++){
            int j=nl_list[i*MAX_NEIGH+k];
            double dmx=pos[j*3]-pos[i*3];
            double dmy=pos[j*3+1]-pos[i*3+1];
            double dmz=pos[j*3+2]-pos[i*3+2];
            mimg_flat(dmx,dmy,dmz,hi,h);
            if(dmx*dmx+dmy*dmy+dmz*dmz > rmcut2) continue;

#ifdef _OPENACC
            #pragma acc loop vector \
                reduction(+:fi0,fi1,fi2,ti0,ti1,ti2,my_Ep, \
                           w00,w01,w02,w10,w11,w12,w20,w21,w22)
#endif
            for(int ai=0;ai<natom;ai++){
                int ia=i*natom*3+ai*3;
                double rax=lab[ia], ray=lab[ia+1], raz=lab[ia+2];

                for(int bj=0;bj<natom;bj++){
                    int jb=j*natom*3+bj*3;
                    double rbx=lab[jb], rby=lab[jb+1], rbz=lab[jb+2];
                    double ddx=dmx+rbx-rax;
                    double ddy=dmy+rby-ray;
                    double ddz=dmz+rbz-raz;
                    double r2=ddx*ddx+ddy*ddy+ddz*ddz;

                    if(r2<RCUT2){
                        if(r2<0.25) r2=0.25;
                        double ri2=1.0/r2;
                        double sr2=sig2_LJ*ri2;
                        double sr6=sr2*sr2*sr2;
                        double sr12=sr6*sr6;
                        double fm=24.0*eps_LJ*(2.0*sr12-sr6)*ri2;
                        double fx=fm*ddx, fy=fm*ddy, fz=fm*ddz;

                        fi0-=fx; fi1-=fy; fi2-=fz;
                        ti0-=(ray*fz-raz*fy);
                        ti1-=(raz*fx-rax*fz);
                        ti2-=(rax*fy-ray*fx);
                        my_Ep+=0.5*(4.0*eps_LJ*(sr12-sr6)-VSHFT);
                        w00+=0.5*ddx*fx; w01+=0.5*ddx*fy; w02+=0.5*ddx*fz;
                        w10+=0.5*ddy*fx; w11+=0.5*ddy*fy; w12+=0.5*ddy*fz;
                        w20+=0.5*ddz*fx; w21+=0.5*ddz*fy; w22+=0.5*ddz*fz;
                    }
                }
            }
        }

        Fv[i*3]=fi0; Fv[i*3+1]=fi1; Fv[i*3+2]=fi2;
        Tv[i*3]=ti0; Tv[i*3+1]=ti1; Tv[i*3+2]=ti2;

#if defined(_OPENACC)
        #pragma acc atomic update
#elif defined(_OPENMP)
        #pragma omp atomic update
#endif
        Wm9[0]+=w00;
#if defined(_OPENACC)
        #pragma acc atomic update
#elif defined(_OPENMP)
        #pragma omp atomic update
#endif
        Wm9[1]+=w01;
#if defined(_OPENACC)
        #pragma acc atomic update
#elif defined(_OPENMP)
        #pragma omp atomic update
#endif
        Wm9[2]+=w02;
#if defined(_OPENACC)
        #pragma acc atomic update
#elif defined(_OPENMP)
        #pragma omp atomic update
#endif
        Wm9[3]+=w10;
#if defined(_OPENACC)
        #pragma acc atomic update
#elif defined(_OPENMP)
        #pragma omp atomic update
#endif
        Wm9[4]+=w11;
#if defined(_OPENACC)
        #pragma acc atomic update
#elif defined(_OPENMP)
        #pragma omp atomic update
#endif
        Wm9[5]+=w12;
#if defined(_OPENACC)
        #pragma acc atomic update
#elif defined(_OPENMP)
        #pragma omp atomic update
#endif
        Wm9[6]+=w20;
#if defined(_OPENACC)
        #pragma acc atomic update
#elif defined(_OPENMP)
        #pragma omp atomic update
#endif
        Wm9[7]+=w21;
#if defined(_OPENACC)
        #pragma acc atomic update
#elif defined(_OPENMP)
        #pragma omp atomic update
#endif
        Wm9[8]+=w22;

        Ep+=my_Ep;
    }
    return Ep;
}


/* =============== Kinetic Energy =============== */
static double ke_trans(const double* vel, int N, double Mmol){
    double s=0;
#ifdef _OPENACC
    #pragma acc parallel loop present(vel) reduction(+:s)
#elif defined(_OPENMP)
    #pragma omp parallel for reduction(+:s)
#endif
    for(int i=0;i<N;i++)
        s+=vel[i*3]*vel[i*3]+vel[i*3+1]*vel[i*3+1]+vel[i*3+2]*vel[i*3+2];
    return 0.5*Mmol*s/CONV;
}

static double ke_rot(const double* omg, int N, double I0){
    double s=0;
#ifdef _OPENACC
    #pragma acc parallel loop present(omg) reduction(+:s)
#elif defined(_OPENMP)
    #pragma omp parallel for reduction(+:s)
#endif
    for(int i=0;i<N;i++)
        s+=omg[i*3]*omg[i*3]+omg[i*3+1]*omg[i*3+1]+omg[i*3+2]*omg[i*3+2];
    return 0.5*I0*s/CONV;
}

static inline double inst_T(double KE, int Nf){return 2*KE/(Nf*kB);}
static inline double inst_P(const double* W, double KEt, double V){
    return (2*KEt+W[0]+W[4]+W[8])/(3*V)*eV2GPa;
}


/* =============== NPT State Variables =============== */
struct NPTState { double xi,Q,Vg[9],W,Pe,Tt; int Nf; };

static NPTState make_npt(double T, double Pe, int N){
    int Nf=6*N-3;
    NPTState s;
    s.xi=0; s.Q=std::max(Nf*kB*T*100.0*100.0,1e-20);
    for(int i=0;i<9;i++) s.Vg[i]=0;
    s.W=std::max((Nf+9)*kB*T*1000.0*1000.0,1e-20);
    s.Pe=Pe; s.Tt=T; s.Nf=Nf;
    return s;
}


/* =============== NPT Velocity-Verlet One Step =============== */
static std::pair<double,double>
step_npt(double* pos, double* vel, double* qv, double* omg,
         double* Fv, double* Tv, double* Wm9,
         double* h, double* hi,
         const double* body, double I0, double Mmol,
         int N, int natom, double rmcut2, double dt, NPTState& npt,
         const int* nl_count, const int* nl_list, double* lab)
{
    double hdt=0.5*dt;
    mat_inv9(h,hi);
#ifdef _OPENACC
    #pragma acc update device(hi[0:9])
#endif

    double V=fabs(mat_det9(h));
    double kt=ke_trans(vel,N,Mmol), kr=ke_rot(omg,N,I0), KE=kt+kr;

    /* (A) Thermostat first half */
    npt.xi+=hdt*(2*KE-npt.Nf*kB*npt.Tt)/npt.Q;
    npt.xi=std::clamp(npt.xi,-0.1,0.1);

    /* (B) Barostat first half */
    double dP=inst_P(Wm9,kt,V)-npt.Pe;
    for(int a=0;a<3;a++) npt.Vg[a*4]+=hdt*V*dP/(npt.W*eV2GPa);
    for(int a=0;a<3;a++) npt.Vg[a*4]=std::clamp(npt.Vg[a*4],-0.01,0.01);

    double eps_tr=npt.Vg[0]*hi[0]+npt.Vg[4]*hi[4]+npt.Vg[8]*hi[8];
    double sc_nh=exp(-hdt*npt.xi);
    double sc_pr=exp(-hdt*eps_tr/3.0);
    double sc_v=sc_nh*sc_pr;
    double cF=CONV/Mmol, cT=CONV/I0;

    /* (C) First half velocity update */
#ifdef _OPENACC
    #pragma acc parallel loop present(vel,omg,Fv,Tv)
#elif defined(_OPENMP)
    #pragma omp parallel for schedule(static)
#endif
    for(int i=0;i<N;i++){
        for(int a=0;a<3;a++){
            vel[i*3+a]=vel[i*3+a]*sc_v+hdt*Fv[i*3+a]*cF;
            omg[i*3+a]=omg[i*3+a]*sc_nh+hdt*Tv[i*3+a]*cT;
        }
    }

    /* (D) Position update (integrate in fractional coordinates + PBC) */
#ifdef _OPENACC
    #pragma acc parallel loop present(pos,vel,hi)
#elif defined(_OPENMP)
    #pragma omp parallel for schedule(static)
#endif
    for(int i=0;i<N;i++){
        double px=pos[i*3],py=pos[i*3+1],pz=pos[i*3+2];
        double vx=vel[i*3],vy=vel[i*3+1],vz=vel[i*3+2];
        double sx=hi[0]*px+hi[1]*py+hi[2]*pz;
        double sy=hi[3]*px+hi[4]*py+hi[5]*pz;
        double sz=hi[6]*px+hi[7]*py+hi[8]*pz;
        double vsx=hi[0]*vx+hi[1]*vy+hi[2]*vz;
        double vsy=hi[3]*vx+hi[4]*vy+hi[5]*vz;
        double vsz=hi[6]*vx+hi[7]*vy+hi[8]*vz;
        sx+=dt*vsx; sy+=dt*vsy; sz+=dt*vsz;
        sx-=floor(sx); sy-=floor(sy); sz-=floor(sz);
        pos[i*3]=sx; pos[i*3+1]=sy; pos[i*3+2]=sz;
    }

    /* (E) Update cell H-matrix (host) -> transfer to device */
    for(int a=0;a<3;a++) for(int b=0;b<3;b++) h[a*3+b]+=dt*npt.Vg[a*3+b];
#ifdef _OPENACC
    #pragma acc update device(h[0:9])
#endif

    /* (F) Fractional coordinates -> Cartesian coordinates */
#ifdef _OPENACC
    #pragma acc parallel loop present(pos,h)
#elif defined(_OPENMP)
    #pragma omp parallel for schedule(static)
#endif
    for(int i=0;i<N;i++){
        double sx=pos[i*3],sy=pos[i*3+1],sz=pos[i*3+2];
        pos[i*3  ]=h[0]*sx+h[1]*sy+h[2]*sz;
        pos[i*3+1]=h[3]*sx+h[4]*sy+h[5]*sz;
        pos[i*3+2]=h[6]*sx+h[7]*sy+h[8]*sz;
    }

    /* (G) Quaternion update */
#ifdef _OPENACC
    #pragma acc parallel loop present(qv,omg)
#elif defined(_OPENMP)
    #pragma omp parallel for schedule(static)
#endif
    for(int i=0;i<N;i++){
        double dq[4],tmp[4];
        omega2dq_flat(omg[i*3],omg[i*3+1],omg[i*3+2],dt,dq);
        qmul_flat(&qv[i*4],dq,tmp);
        qv[i*4]=tmp[0]; qv[i*4+1]=tmp[1]; qv[i*4+2]=tmp[2]; qv[i*4+3]=tmp[3];
        qnorm_flat(&qv[i*4]);
    }

    /* (H) Recalculate forces */
    mat_inv9(h,hi);
#ifdef _OPENACC
    #pragma acc update device(hi[0:9])
#endif
    double Ep=forces(Fv,Tv,Wm9,pos,qv,body,h,hi,nl_count,nl_list,
                     N,natom,rmcut2,lab);

    /* (I) Second half velocity update */
    double eps_tr2=npt.Vg[0]*hi[0]+npt.Vg[4]*hi[4]+npt.Vg[8]*hi[8];
    double sc_v2=sc_nh*exp(-hdt*eps_tr2/3.0);
#ifdef _OPENACC
    #pragma acc parallel loop present(vel,omg,Fv,Tv)
#elif defined(_OPENMP)
    #pragma omp parallel for schedule(static)
#endif
    for(int i=0;i<N;i++){
        for(int a=0;a<3;a++){
            vel[i*3+a]=(vel[i*3+a]+hdt*Fv[i*3+a]*cF)*sc_v2;
            omg[i*3+a]=(omg[i*3+a]+hdt*Tv[i*3+a]*cT)*sc_nh;
        }
    }

    /* (J)(K) Thermostat/Barostat second half update */
    kt=ke_trans(vel,N,Mmol); kr=ke_rot(omg,N,I0); KE=kt+kr;
    npt.xi+=hdt*(2*KE-npt.Nf*kB*npt.Tt)/npt.Q;
    npt.xi=std::clamp(npt.xi,-0.1,0.1);
    double V2=fabs(mat_det9(h));
#ifdef _OPENACC
    #pragma acc update self(Wm9[0:9])
#endif
    dP=inst_P(Wm9,kt,V2)-npt.Pe;
    for(int a=0;a<3;a++) npt.Vg[a*4]+=hdt*V2*dP/(npt.W*eV2GPa);
    for(int a=0;a<3;a++) npt.Vg[a*4]=std::clamp(npt.Vg[a*4],-0.01,0.01);

    return {Ep,KE};
}


/* =============== Main Program =============== */
int main(int argc, char** argv){
    /* --- Parameter settings --- */
    int nc=3;
    for(int i=1;i<argc;i++){
        std::string a=argv[i];
        if(a.substr(0,7)=="--cell=") nc=std::atoi(a.substr(7).c_str());
        else if(a[0]!='-') nc=atoi(argv[i]);
    }
    if(nc<1||nc>8){ printf("Error: nc must be 1-8 (got %d)\n",nc); return 1; }

    constexpr int nsteps=1000;
    constexpr int mon=100;
    constexpr int nlup=25;
    constexpr double T=300.0;
    constexpr double Pe=0.0;
    constexpr double dt=1.0;
    constexpr double a0=14.17;
    int avg_from=nsteps-nsteps/4;

    /* --- Generate C60 molecular coordinates --- */
    C60Data c60=generate_c60();
    int natom=C60_NATOM;

    /* --- Allocate arrays (raw arrays: for OpenACC data management) --- */
    int Nmax=4*nc*nc*nc;
    double* pos      =new double[Nmax*3]();
    double* vel      =new double[Nmax*3]();
    double* omg      =new double[Nmax*3]();
    double* qv       =new double[Nmax*4]();
    double* Fv       =new double[Nmax*3]();
    double* Tv       =new double[Nmax*3]();
    double* lab      =new double[Nmax*natom*3]();
    double  h[9]={}, hi[9]={}, Wm9[9]={};
    double* body     =new double[natom*3];
    int*    nl_count =new int[Nmax]();
    int*    nl_list  =new int[Nmax*MAX_NEIGH]();

    for(int i=0;i<natom*3;i++) body[i]=c60.coords[i];

    int N=make_fcc(a0,nc,pos,h);
    mat_inv9(h,hi);

    /* --- Display banner --- */
    printf("================================================================\n");
#if defined(_OPENACC)
    printf("  C60 LJ NPT-MD Core (OpenACC GPU)\n");
#elif defined(_OPENMP)
    printf("  C60 LJ NPT-MD Core (OpenMP, %d threads)\n", omp_get_max_threads());
#else
    printf("  C60 LJ NPT-MD Core (Serial)\n");
#endif
    printf("================================================================\n");
#ifdef _OPENACC
    int ndev=acc_get_num_devices(acc_device_nvidia);
    printf("  GPU devices     : %d\n", ndev);
    if(ndev>0){
        acc_set_device_num(0, acc_device_nvidia);
        printf("  Active device   : %d (NVIDIA)\n", acc_get_device_num(acc_device_nvidia));
    }
#endif
    printf("  FCC cell        : %dx%dx%d  N=%d molecules\n", nc,nc,nc,N);
    printf("  Atoms/molecule  : %d\n", natom);
    printf("  a0=%.2f A  T=%.0f K  P=%.1f GPa  dt=%.1f fs  steps=%d\n",
           a0,T,Pe,dt,nsteps);
    printf("  MAX_NEIGH=%d  VECTOR_LENGTH=%d\n", MAX_NEIGH, VECTOR_LENGTH);

    double mem_lab = (double)N*natom*3*8/1024/1024;
    double mem_nl  = (double)N*MAX_NEIGH*4/1024/1024;
    double mem_total = (double)(N*3*8*5 + N*4*8 + N*natom*3*8 + natom*3*8
                                + N*4 + N*MAX_NEIGH*4 + 9*8*3) / 1024/1024;
    printf("  Memory          : lab=%.2f MB  nl=%.2f MB  total=%.2f MB\n",
           mem_lab, mem_nl, mem_total);
    printf("================================================================\n\n");

    /* --- Initial velocities (Maxwell-Boltzmann distribution) --- */
    std::mt19937 rng(42);
    std::normal_distribution<double> gauss(0,1);
    double sv=sqrt(kB*T*CONV/c60.Mmol), sw=sqrt(kB*T*CONV/c60.I0);
    for(int i=0;i<N;i++){
        for(int a=0;a<3;a++){vel[i*3+a]=sv*gauss(rng); omg[i*3+a]=sw*gauss(rng);}
        for(int a=0;a<4;a++) qv[i*4+a]=gauss(rng);
        double n=sqrt(qv[i*4]*qv[i*4]+qv[i*4+1]*qv[i*4+1]
                     +qv[i*4+2]*qv[i*4+2]+qv[i*4+3]*qv[i*4+3]);
        for(int a=0;a<4;a++) qv[i*4+a]/=n;
    }
    double vcm[3]={0,0,0};
    for(int i=0;i<N;i++) for(int a=0;a<3;a++) vcm[a]+=vel[i*3+a];
    for(int a=0;a<3;a++) vcm[a]/=N;
    for(int i=0;i<N;i++) for(int a=0;a<3;a++) vel[i*3+a]-=vcm[a];

    NPTState npt=make_npt(T,Pe,N);

    /* --- Initial neighbor list construction (host) --- */
    nlist_build_sym(pos,h,hi,N,RMCUT,nl_count,nl_list);

    /* === Data region (placed on GPU only in ACC mode) === */
#ifdef _OPENACC
    #pragma acc data \
        copy(pos[0:N*3], vel[0:N*3], omg[0:N*3], qv[0:N*4]) \
        copy(Fv[0:N*3], Tv[0:N*3]) \
        copyin(body[0:natom*3]) \
        copy(h[0:9], hi[0:9], Wm9[0:9]) \
        create(lab[0:N*natom*3]) \
        copyin(nl_count[0:N], nl_list[0:N*MAX_NEIGH])
    {
#endif

    /* Apply PBC + initial force calculation */
    apply_pbc(pos,h,hi,N);
    forces(Fv,Tv,Wm9,pos,qv,body,h,hi,nl_count,nl_list,
           N,natom,RMCUT2,lab);
#ifdef _OPENACC
    #pragma acc update self(Wm9[0:9])
#endif

    double sT=0,sP=0,sa=0,sEp=0; int nav=0;
    auto t0=std::chrono::steady_clock::now();
    printf("%8s %7s %9s %8s %10s %7s\n",
           "step","T[K]","P[GPa]","a[A]","Ecoh[eV]","t[s]");

    /* === MD main loop === */
    for(int g=1;g<=nsteps;g++){

        /* Rebuild neighbor list (host) */
        if(g%nlup==0){
#ifdef _OPENACC
            #pragma acc update self(pos[0:N*3])
#endif
            mat_inv9(h,hi);
            nlist_build_sym(pos,h,hi,N,RMCUT,nl_count,nl_list);
#ifdef _OPENACC
            #pragma acc update device(nl_count[0:N], nl_list[0:N*MAX_NEIGH])
            #pragma acc update device(hi[0:9])
#endif
        }

        /* One step of time evolution */
        auto [Ep,KE]=step_npt(pos,vel,qv,omg,Fv,Tv,Wm9,
                               h,hi,body,c60.I0,c60.Mmol,
                               N,natom,RMCUT2,dt,npt,
                               nl_count,nl_list,lab);

        /* Instantaneous physical quantities */
        double kt=ke_trans(vel,N,c60.Mmol);
        double V=fabs(mat_det9(h));
        double Tn=inst_T(KE,npt.Nf), Pn=inst_P(Wm9,kt,V);
        double Ec=Ep/N, an=h[0]/nc;
        if(g>=avg_from){sT+=Tn;sP+=Pn;sa+=an;sEp+=Ec;nav++;}

        /* Monitoring output */
        if(g%mon==0||g==nsteps){
            double el=std::chrono::duration<double>(
                std::chrono::steady_clock::now()-t0).count();
            printf("%8d %7.1f %9.3f %8.3f %10.5f %7.0f\n",
                   g,Tn,Pn,an,Ec,el);
        }
    }

    if(nav>0) printf("Avg(%d): T=%.2f P=%.4f a=%.4f Ecoh=%.5f\n",
                      nav,sT/nav,sP/nav,sa/nav,sEp/nav);
    printf("Done %.1fs\n",std::chrono::duration<double>(
        std::chrono::steady_clock::now()-t0).count());

#ifdef _OPENACC
    } /* end #pragma acc data */
#endif

    /* --- Cleanup --- */
    delete[] pos; delete[] vel; delete[] omg; delete[] qv;
    delete[] Fv;  delete[] Tv;  delete[] lab; delete[] body;
    delete[] nl_count; delete[] nl_list;
    return 0;
}
