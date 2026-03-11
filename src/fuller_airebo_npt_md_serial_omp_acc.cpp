// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2025, Takeshi Nishikawa
/*===========================================================================
  fuller_airebo_npt_md_serial_omp_acc.cpp — Fullerene Crystal NPT-MD
  (AIREBO, Serial / OpenMP / OpenACC GPU)

  REBO-II (Brenner 2002) + LJ (Stuart 2000) / NH+PR / 3D PBC

  Compilation:
    Serial:  g++ -std=c++17 -O3 -Wno-unknown-pragmas \
               -o fuller_airebo_npt_md_serial fuller_airebo_npt_md_serial_omp_acc.cpp -lm
    OpenMP:  g++ -std=c++17 -O3 -fopenmp -Wno-unknown-pragmas \
               -o fuller_airebo_npt_md_omp fuller_airebo_npt_md_serial_omp_acc.cpp -lm
    OpenACC: nvc++ -std=c++17 -O3 -acc -gpu=cc80 -Minfo=accel \
               -o fuller_airebo_npt_md_gpu fuller_airebo_npt_md_serial_omp_acc.cpp -lm

  Runtime Options (all in --key=value format):
    --help                  Show this help
    --fullerene=<name>      Fullerene species (default: C60)
    --crystal=<fcc|hcp|bcc> Crystal structure (default: fcc)
    --cell=<nc>             Unit cell repeat count (default: 3)
    --temp=<T_K>            Target temperature [K] (default: 298.0)
    --pres=<P_GPa>          Target pressure [GPa] (default: 0.0)
    --step=<N>              Production steps (default: 10000)
    --dt=<fs>               Time step [fs] (default: 0.5)
    --init_scale=<s>        Lattice constant scale factor (default: 1.0)
    --seed=<n>              Random seed (default: 42)
    --coldstart=<N>         Cold start (4K) steps (default: 0)
    --warmup=<N>            Warmup steps 4K->T (default: 0)
    --from=<step>           Averaging start step (default: 3/4 point of production)
    --to=<step>             Averaging end step (default: nsteps)
    --mon=<N>               Monitoring output interval (default: auto)
    --warmup_mon=<mode>     Warmup output frequency norm|freq|some (default: norm)
    --ovito=<N>             OVITO XYZ output interval (0=disabled, default: 0)
    --restart=<N>           Restart save interval (0=disabled, default: 0)
    --resfile=<path>        Resume from restart file
    --libdir=<path>         Fullerene library (default: FullereneLib)

  Execution Examples:
    # Basic run (C60 FCC 3x3x3, 298K, 10000 steps)
    ./fuller_airebo_npt_md_serial

    # Long run with specified temperature and pressure
    ./fuller_airebo_npt_md_omp --temp=500 --pres=1.0 --step=50000

    # Cold start + Warmup + Production
    ./fuller_airebo_npt_md_serial --coldstart=5000 --warmup=5000 --step=20000

    # OVITO output (write to XYZ file every 100 steps)
    ./fuller_airebo_npt_md_omp --step=10000 --ovito=100

    # Restart save (every 5000 steps + final step)
    ./fuller_airebo_npt_md_serial --step=50000 --restart=5000

    # Resume from restart file
    ./fuller_airebo_npt_md_serial --resfile=restart_airebo_serial_00025000.rst

    # OVITO + Restart used simultaneously
    ./fuller_airebo_npt_md_gpu --step=100000 --ovito=500 --restart=10000

    # C84 fullerene
    ./fuller_airebo_npt_md_omp --fullerene=C84 --cell=4 --step=20000

  Stop Control:
    Create the following files in the current directory during execution to control behavior:
    - abort.md: Stop immediately (saves restart if restart is enabled, then exits)
    - stop.md:  Stop at the next restart checkpoint

  Unit System: A, amu, eV, fs, K, GPa
===========================================================================*/
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <string>
#include <vector>
#include <array>
#include <map>
#include <algorithm>
#include <random>
#include <dirent.h>
#include <fstream>
#include <sstream>
#include <chrono>
#include <sys/stat.h>

static inline bool dir_exists(const char* path){
    struct stat st;
    return stat(path,&st)==0 && S_ISDIR(st.st_mode);
}

#ifdef _OPENACC
  #include <openacc.h>
#endif
#ifdef _OPENMP
  #include <omp.h>
#else
  inline int omp_get_max_threads(){return 1;}
#endif

using Vec3=std::array<double,3>;
using Mat3=std::array<std::array<double,3>,3>;

/* ═══════════ Constants ═══════════ */
constexpr double CONV=9.64853321e-3,kB_=8.617333262e-5;
constexpr double eV2GPa=160.21766208,eV2kcalmol=23.06054783,PI_=3.14159265358979323846;
constexpr double mC_=12.011;

/* REBO-II C-C */
constexpr double Q_CC=0.3134602960833,A_CC=10953.544162170,alpha_CC=4.7465390606595;
constexpr double B1_CC=12388.79197798,beta1_CC=4.7204523127;
constexpr double B2_CC=17.56740646509,beta2_CC=1.4332132499;
constexpr double B3_CC=30.71493208065,beta3_CC=1.3826912506;
constexpr double Dmin_CC=1.7,Dmax_CC=2.0,BO_DELTA=0.5;
constexpr double G_a0=0.00020813,G_c0=330.0,G_d0=3.5;
constexpr double REBO_RCUT=Dmax_CC+0.3;

/* LJ intermolecular */
constexpr double sig_LJ=3.40,eps_LJ_=2.84e-3;
constexpr double LJ_RCUT=3.0*sig_LJ,LJ_RCUT2=LJ_RCUT*LJ_RCUT;
constexpr double _sr_v=sig_LJ/LJ_RCUT;
constexpr double _sr2=_sr_v*_sr_v,_sr6=_sr2*_sr2*_sr2;
constexpr double LJ_VSHFT=4*eps_LJ_*(_sr6*_sr6-_sr6);
constexpr double sig2_LJ=sig_LJ*sig_LJ;

constexpr int MAX_REBO_NEIGH=12;
constexpr int MAX_LJ_NEIGH=400;

/* ═══════════ Flat Mat3 ops ═══════════ */
#define H9(h,i,j) ((h)[3*(i)+(j)])
#pragma acc routine seq
static inline double mat_det9(const double*h){
    return H9(h,0,0)*(H9(h,1,1)*H9(h,2,2)-H9(h,1,2)*H9(h,2,1))
          -H9(h,0,1)*(H9(h,1,0)*H9(h,2,2)-H9(h,1,2)*H9(h,2,0))
          +H9(h,0,2)*(H9(h,1,0)*H9(h,2,1)-H9(h,1,1)*H9(h,2,0));}
static void mat_inv9(const double*h,double*hi){
    double d=mat_det9(h),id=1.0/d;
    hi[0]=id*(h[4]*h[8]-h[5]*h[7]);hi[1]=id*(h[2]*h[7]-h[1]*h[8]);hi[2]=id*(h[1]*h[5]-h[2]*h[4]);
    hi[3]=id*(h[5]*h[6]-h[3]*h[8]);hi[4]=id*(h[0]*h[8]-h[2]*h[6]);hi[5]=id*(h[2]*h[3]-h[0]*h[5]);
    hi[6]=id*(h[3]*h[7]-h[4]*h[6]);hi[7]=id*(h[1]*h[6]-h[0]*h[7]);hi[8]=id*(h[0]*h[4]-h[1]*h[3]);}

/* ═══════════ PBC (device) ═══════════ */
#pragma acc routine seq
static inline void mimg9(double&dx,double&dy,double&dz,const double*hi,const double*h){
    double s0=hi[0]*dx+hi[1]*dy+hi[2]*dz,s1=hi[3]*dx+hi[4]*dy+hi[5]*dz,s2=hi[6]*dx+hi[7]*dy+hi[8]*dz;
    s0-=round(s0);s1-=round(s1);s2-=round(s2);
    dx=h[0]*s0+h[1]*s1+h[2]*s2;dy=h[3]*s0+h[4]*s1+h[5]*s2;dz=h[6]*s0+h[7]*s1+h[8]*s2;}

static void apply_pbc(double*pos,const double*h,const double*hi,int N){
#ifdef _OPENACC
    #pragma acc parallel loop present(pos,h,hi)
#elif defined(_OPENMP)
    #pragma omp parallel for
#endif
    for(int i=0;i<N;i++){
        double px=pos[i*3],py=pos[i*3+1],pz=pos[i*3+2];
        double s0=hi[0]*px+hi[1]*py+hi[2]*pz,s1=hi[3]*px+hi[4]*py+hi[5]*pz,s2=hi[6]*px+hi[7]*py+hi[8]*pz;
        s0-=floor(s0);s1-=floor(s1);s2-=floor(s2);
        pos[i*3]=h[0]*s0+h[1]*s1+h[2]*s2;
        pos[i*3+1]=h[3]*s0+h[4]*s1+h[5]*s2;
        pos[i*3+2]=h[6]*s0+h[7]*s1+h[8]*s2;}}

/* ═══════════ REBO device functions ═══════════ */
#pragma acc routine seq
static inline double fc_d(double r,double dmin,double dmax){
    if(r<=dmin)return 1.0;if(r>=dmax)return 0.0;
    return 0.5*(1.0+cos(PI_*(r-dmin)/(dmax-dmin)));}
#pragma acc routine seq
static inline double dfc_d(double r,double dmin,double dmax){
    if(r<=dmin||r>=dmax)return 0.0;
    return -0.5*PI_/(dmax-dmin)*sin(PI_*(r-dmin)/(dmax-dmin));}
#pragma acc routine seq
static inline double VR_CC_d(double r){return(1.0+Q_CC/r)*A_CC*exp(-alpha_CC*r);}
#pragma acc routine seq
static inline double dVR_CC_d(double r){
    double ex=A_CC*exp(-alpha_CC*r);
    return(-Q_CC/(r*r))*ex+(1.0+Q_CC/r)*(-alpha_CC)*ex;}
#pragma acc routine seq
static inline double VA_CC_d(double r){
    return B1_CC*exp(-beta1_CC*r)+B2_CC*exp(-beta2_CC*r)+B3_CC*exp(-beta3_CC*r);}
#pragma acc routine seq
static inline double dVA_CC_d(double r){
    return-beta1_CC*B1_CC*exp(-beta1_CC*r)-beta2_CC*B2_CC*exp(-beta2_CC*r)-beta3_CC*B3_CC*exp(-beta3_CC*r);}
#pragma acc routine seq
static inline double G_C_d(double x){
    double c2=G_c0*G_c0,d2=G_d0*G_d0,h=1.0+x;
    return G_a0*(1.0+c2/d2-c2/(d2+h*h));}
#pragma acc routine seq
static inline double dG_C_d(double x){
    double c2=G_c0*G_c0,d2=G_d0*G_d0,h=1.0+x,dn=d2+h*h;
    return G_a0*2.0*c2*h/(dn*dn);}

/* ═══════════ cc1 + resolver + crystal (host, same as LJ) ═══════════ */
struct MolData{std::vector<Vec3> coords;int natom;double Rmol,Dmol;};
MolData load_cc1(const std::string&path){
    MolData md;std::ifstream f(path);if(!f){fprintf(stderr,"Cannot open %s\n",path.c_str());exit(1);}
    f>>md.natom;md.coords.resize(md.natom);
    for(int i=0;i<md.natom;i++){std::string line;std::getline(f,line);
        while(line.empty()&&f.good())std::getline(f,line);
        std::istringstream ss(line);std::string el;int idx;double x,y,z;int fl;
        ss>>el>>idx>>x>>y>>z>>fl;md.coords[i]={x,y,z};}
    Vec3 cm={0,0,0};for(auto&c:md.coords)for(int a=0;a<3;a++)cm[a]+=c[a];
    for(int a=0;a<3;a++)cm[a]/=md.natom;
    for(auto&c:md.coords)for(int a=0;a<3;a++)c[a]-=cm[a];
    md.Rmol=0;md.Dmol=0;
    for(int i=0;i<md.natom;i++){double r=sqrt(md.coords[i][0]*md.coords[i][0]+md.coords[i][1]*md.coords[i][1]+md.coords[i][2]*md.coords[i][2]);
        md.Rmol=std::max(md.Rmol,r);
        for(int j=i+1;j<md.natom;j++){double dx=md.coords[i][0]-md.coords[j][0],dy=md.coords[i][1]-md.coords[j][1],dz=md.coords[i][2]-md.coords[j][2];
            md.Dmol=std::max(md.Dmol,sqrt(dx*dx+dy*dy+dz*dz));}}
    return md;}
std::pair<std::string,std::string> resolve_fullerene(const std::string&spec,const std::string&lib="FullereneLib"){
    std::string sl=spec;std::transform(sl.begin(),sl.end(),sl.begin(),::tolower);
    if(sl=="buckyball"||sl=="c60"||sl=="c60:ih")return{lib+"/C60-76/C60-Ih.cc1","C60(Ih)"};
    if(sl=="c70"||sl=="c70:d5h")return{lib+"/C60-76/C70-D5h.cc1","C70(D5h)"};
    if(sl=="c72"||sl=="c72:d6d")return{lib+"/C60-76/C72-D6d.cc1","C72(D6d)"};
    if(sl=="c74"||sl=="c74:d3h")return{lib+"/C60-76/C74-D3h.cc1","C74(D3h)"};
    if(sl.substr(0,4)=="c76:"&&sl.size()>4){std::string sym=spec.substr(4);return{lib+"/C60-76/C76-"+sym+".cc1","C76("+sym+")"};}
    if(sl.substr(0,4)=="c84:"){std::string rest=spec.substr(4);auto c=rest.find(':');
        if(c!=std::string::npos){int n=std::atoi(rest.substr(0,c).c_str());std::string sym=rest.substr(c+1);
            char buf[64];snprintf(buf,64,"C84-No.%02d-%s.cc1",n,sym.c_str());return{lib+"/C84/"+buf,"C84 No."+std::to_string(n)};}
        else{int n=std::atoi(rest.c_str());char pfx[32];snprintf(pfx,32,"C84-No.%02d-",n);
            std::string dpath=lib+"/C84";DIR*dp=opendir(dpath.c_str());
            if(dp){struct dirent*ep;while((ep=readdir(dp))){std::string fn=ep->d_name;
                if(fn.find(pfx)==0&&fn.size()>4&&fn.substr(fn.size()-4)==".cc1"){closedir(dp);return{dpath+"/"+fn,"C84 No."+std::to_string(n)};}}
                closedir(dp);}}}
    fprintf(stderr,"Unknown: %s\n",spec.c_str());exit(1);}

static int make_fcc(double a,int nc,double*pos,double*h){
    double bas[4][3]={{0,0,0},{.5*a,.5*a,0},{.5*a,0,.5*a},{0,.5*a,.5*a}};int n=0;
    for(int x=0;x<nc;x++)for(int y=0;y<nc;y++)for(int z=0;z<nc;z++)
        for(int b=0;b<4;b++){pos[n*3]=a*x+bas[b][0];pos[n*3+1]=a*y+bas[b][1];pos[n*3+2]=a*z+bas[b][2];n++;}
    for(int i=0;i<9;i++)h[i]=0;h[0]=h[4]=h[8]=nc*a;return n;}
static int make_hcp(double a,int nc,double*pos,double*h){
    double c=a*sqrt(8.0/3.0),a1[3]={a,0,0},a2[3]={a/2,a*sqrt(3.0)/2,0},a3[3]={0,0,c};
    double bas[2][3]={{0,0,0},{1.0/3,2.0/3,0.5}};int n=0;
    for(int x=0;x<nc;x++)for(int y=0;y<nc;y++)for(int z=0;z<nc;z++)
        for(int b=0;b<2;b++){double fx=x+bas[b][0],fy=y+bas[b][1],fz=z+bas[b][2];
            pos[n*3]=fx*a1[0]+fy*a2[0]+fz*a3[0];pos[n*3+1]=fx*a1[1]+fy*a2[1]+fz*a3[1];
            pos[n*3+2]=fx*a1[2]+fy*a2[2]+fz*a3[2];n++;}
    for(int i=0;i<9;i++)h[i]=0;h[0]=nc*a1[0];h[3]=nc*a1[1];h[1]=nc*a2[0];h[4]=nc*a2[1];h[8]=nc*a3[2];return n;}
static int make_bcc(double a,int nc,double*pos,double*h){
    double bas[2][3]={{0,0,0},{.5*a,.5*a,.5*a}};int n=0;
    for(int x=0;x<nc;x++)for(int y=0;y<nc;y++)for(int z=0;z<nc;z++)
        for(int b=0;b<2;b++){pos[n*3]=a*x+bas[b][0];pos[n*3+1]=a*y+bas[b][1];pos[n*3+2]=a*z+bas[b][2];n++;}
    for(int i=0;i<9;i++)h[i]=0;h[0]=h[4]=h[8]=nc*a;return n;}
static double default_a0(double dmax,const std::string&st,double s){
    double m=1.4,a0;if(st=="FCC")a0=dmax*sqrt(2.0)*m;else if(st=="HCP")a0=dmax*m;else a0=dmax*2.0/sqrt(3.0)*m;
    return a0*s;}

/* ═══════════ Neighbor list build (host) ═══════════ */
static void build_nlist_rebo(const double*pos,const double*h,const double*hi,
    int Na,const int*mol_id,int*nlc,int*nll){
    double rc2=REBO_RCUT*REBO_RCUT;
    for(int i=0;i<Na;i++)nlc[i]=0;
    for(int i=0;i<Na-1;i++){int mi=mol_id[i];
        for(int j=i+1;j<Na;j++){if(mol_id[j]!=mi)continue;
            double dx=pos[j*3]-pos[i*3],dy=pos[j*3+1]-pos[i*3+1],dz=pos[j*3+2]-pos[i*3+2];
            mimg9(dx,dy,dz,hi,h);
            if(dx*dx+dy*dy+dz*dz<rc2){
                if(nlc[i]<MAX_REBO_NEIGH){nll[i*MAX_REBO_NEIGH+nlc[i]]=j;nlc[i]++;}
                if(nlc[j]<MAX_REBO_NEIGH){nll[j*MAX_REBO_NEIGH+nlc[j]]=i;nlc[j]++;}}}}}

static void build_nlist_lj(const double*pos,const double*h,const double*hi,
    int Na,const int*mol_id,int*nlc,int*nll){
    double rc2=(LJ_RCUT+2.0)*(LJ_RCUT+2.0);
    for(int i=0;i<Na;i++)nlc[i]=0;
    for(int i=0;i<Na-1;i++){int mi=mol_id[i];
        for(int j=i+1;j<Na;j++){if(mol_id[j]==mi)continue;
            double dx=pos[j*3]-pos[i*3],dy=pos[j*3+1]-pos[i*3+1],dz=pos[j*3+2]-pos[i*3+2];
            mimg9(dx,dy,dz,hi,h);
            if(dx*dx+dy*dy+dz*dz<rc2){
                if(nlc[i]<MAX_LJ_NEIGH){nll[i*MAX_LJ_NEIGH+nlc[i]]=j;nlc[i]++;}}}}}

/* ═══════════ REBO-II force (atomics for F[j]/F[k]/F[l]) ═══════════ */
static double compute_rebo(double*F,double*vir9,
    const double*pos,const double*h,const double*hi,
    const int*nlc,const int*nll,int Na)
{
    double Ep=0;
#ifdef _OPENACC
    #pragma acc parallel loop gang \
        present(F,vir9,pos,h,hi,nlc,nll) reduction(+:Ep)
#elif defined(_OPENMP)
    #pragma omp parallel for schedule(dynamic,4) reduction(+:Ep)
#endif
    for(int i=0;i<Na;i++){
        int nni=nlc[i];
        for(int jn=0;jn<nni;jn++){
            int j=nll[i*MAX_REBO_NEIGH+jn];
            if(j<=i)continue;
            double dx=pos[j*3]-pos[i*3],dy=pos[j*3+1]-pos[i*3+1],dz=pos[j*3+2]-pos[i*3+2];
            mimg9(dx,dy,dz,hi,h);
            double rij=sqrt(dx*dx+dy*dy+dz*dz);if(rij>Dmax_CC)continue;
            double fcut=fc_d(rij,Dmin_CC,Dmax_CC),dfcut=dfc_d(rij,Dmin_CC,Dmax_CC);
            if(fcut<1e-15&&dfcut==0)continue;
            double vr=VR_CC_d(rij),dvr=dVR_CC_d(rij),va=VA_CC_d(rij),dva=dVA_CC_d(rij);
            double rij_inv=1.0/rij;
            double rhat0=dx*rij_inv,rhat1=dy*rij_inv,rhat2=dz*rij_inv;

            /* b_ij */
            double Gs_ij=0;
            for(int kn=0;kn<nni;kn++){int k=nll[i*MAX_REBO_NEIGH+kn];if(k==j)continue;
                double dkx=pos[k*3]-pos[i*3],dky=pos[k*3+1]-pos[i*3+1],dkz=pos[k*3+2]-pos[i*3+2];
                mimg9(dkx,dky,dkz,hi,h);
                double rik=sqrt(dkx*dkx+dky*dky+dkz*dkz);if(rik>Dmax_CC)continue;
                double fc_ik=fc_d(rik,Dmin_CC,Dmax_CC);if(fc_ik<1e-15)continue;
                double costh=(dx*dkx+dy*dky+dz*dkz)/(rij*rik);
                if(costh<-1.0)costh=-1.0;if(costh>1.0)costh=1.0;
                Gs_ij+=fc_ik*G_C_d(costh);}
            double bij=pow(1.0+Gs_ij,-BO_DELTA);

            /* b_ji */
            double Gs_ji=0;
            int nnj=nlc[j];
            for(int ln=0;ln<nnj;ln++){int l=nll[j*MAX_REBO_NEIGH+ln];if(l==i)continue;
                double dlx=pos[l*3]-pos[j*3],dly=pos[l*3+1]-pos[j*3+1],dlz=pos[l*3+2]-pos[j*3+2];
                mimg9(dlx,dly,dlz,hi,h);
                double rjl=sqrt(dlx*dlx+dly*dly+dlz*dlz);if(rjl>Dmax_CC)continue;
                double fc_jl=fc_d(rjl,Dmin_CC,Dmax_CC);if(fc_jl<1e-15)continue;
                double costh=(-dx*dlx-dy*dly-dz*dlz)/(rij*rjl);
                if(costh<-1.0)costh=-1.0;if(costh>1.0)costh=1.0;
                Gs_ji+=fc_jl*G_C_d(costh);}
            double bji=pow(1.0+Gs_ji,-BO_DELTA);
            double bbar=0.5*(bij+bji);

            Ep+=fcut*(vr-bbar*va);
            double fpair=(dfcut*(vr-bbar*va)+fcut*(dvr-bbar*dva))*rij_inv;

            /* Pair force: F[i] safe, F[j] atomic */
            F[i*3]+=fpair*dx; F[i*3+1]+=fpair*dy; F[i*3+2]+=fpair*dz;
#if defined(_OPENACC)
            #pragma acc atomic update
#elif defined(_OPENMP)
            #pragma omp atomic update
#endif
            F[j*3]-=fpair*dx;
#if defined(_OPENACC)
            #pragma acc atomic update
#elif defined(_OPENMP)
            #pragma omp atomic update
#endif
            F[j*3+1]-=fpair*dy;
#if defined(_OPENACC)
            #pragma acc atomic update
#elif defined(_OPENMP)
            #pragma omp atomic update
#endif
            F[j*3+2]-=fpair*dz;

            /* Pair virial (atomic) */
            for(int a=0;a<3;a++){double da=(a==0?dx:a==1?dy:dz);
                for(int b=0;b<3;b++){double db=(b==0?dx:b==1?dy:dz);
#if defined(_OPENACC)
                    #pragma acc atomic update
#elif defined(_OPENMP)
                    #pragma omp atomic update
#endif
                    vir9[a*3+b]-=da*fpair*db;}}

            /* 3-body: db_ij/dr_k */
            if(fabs(Gs_ij)>1e-20&&va>1e-20){
                double dbp=-BO_DELTA*pow(1.0+Gs_ij,-BO_DELTA-1.0);
                double vh=0.5*fcut*va;
                for(int kn=0;kn<nni;kn++){int k=nll[i*MAX_REBO_NEIGH+kn];if(k==j)continue;
                    double dkx=pos[k*3]-pos[i*3],dky=pos[k*3+1]-pos[i*3+1],dkz=pos[k*3+2]-pos[i*3+2];
                    mimg9(dkx,dky,dkz,hi,h);
                    double rik=sqrt(dkx*dkx+dky*dky+dkz*dkz);if(rik>Dmax_CC)continue;
                    double fc_ik=fc_d(rik,Dmin_CC,Dmax_CC);if(fc_ik<1e-15)continue;
                    double dfc_ik=dfc_d(rik,Dmin_CC,Dmax_CC);
                    double rik_inv=1.0/rik;
                    double rhat_ik0=dkx*rik_inv,rhat_ik1=dky*rik_inv,rhat_ik2=dkz*rik_inv;
                    double costh=(dx*dkx+dy*dky+dz*dkz)/(rij*rik);
                    if(costh<-1.0)costh=-1.0;if(costh>1.0)costh=1.0;
                    double gv=G_C_d(costh),dgv=dG_C_d(costh),coeff=-vh*dbp;
                    double rh[3]={rhat0,rhat1,rhat2},rk[3]={rhat_ik0,rhat_ik1,rhat_ik2};
                    for(int a=0;a<3;a++){
                        double fk1=coeff*dfc_ik*gv*rk[a];
                        double dc=(rh[a]-costh*rk[a])*rik_inv;
                        double fk=fk1+coeff*fc_ik*dgv*dc;
#if defined(_OPENACC)
                        #pragma acc atomic update
#elif defined(_OPENMP)
                        #pragma omp atomic update
#endif
                        F[k*3+a]+=fk;
                        F[i*3+a]-=fk;}}}

            /* 3-body: db_ji/dr_l */
            if(fabs(Gs_ji)>1e-20&&va>1e-20){
                double dbp=-BO_DELTA*pow(1.0+Gs_ji,-BO_DELTA-1.0);
                double vh=0.5*fcut*va;
                for(int ln=0;ln<nnj;ln++){int l=nll[j*MAX_REBO_NEIGH+ln];if(l==i)continue;
                    double dlx=pos[l*3]-pos[j*3],dly=pos[l*3+1]-pos[j*3+1],dlz=pos[l*3+2]-pos[j*3+2];
                    mimg9(dlx,dly,dlz,hi,h);
                    double rjl=sqrt(dlx*dlx+dly*dly+dlz*dlz);if(rjl>Dmax_CC)continue;
                    double fc_jl=fc_d(rjl,Dmin_CC,Dmax_CC);if(fc_jl<1e-15)continue;
                    double dfc_jl=dfc_d(rjl,Dmin_CC,Dmax_CC);
                    double rjl_inv=1.0/rjl;
                    double rhat_jl0=dlx*rjl_inv,rhat_jl1=dly*rjl_inv,rhat_jl2=dlz*rjl_inv;
                    double costh=(-dx*dlx-dy*dly-dz*dlz)/(rij*rjl);
                    if(costh<-1.0)costh=-1.0;if(costh>1.0)costh=1.0;
                    double gv=G_C_d(costh),dgv=dG_C_d(costh),coeff=-vh*dbp;
                    double rji[3]={-rhat0,-rhat1,-rhat2},rl[3]={rhat_jl0,rhat_jl1,rhat_jl2};
                    for(int a=0;a<3;a++){
                        double fl1=coeff*dfc_jl*gv*rl[a];
                        double dc=(rji[a]-costh*rl[a])*rjl_inv;
                        double fl=fl1+coeff*fc_jl*dgv*dc;
#if defined(_OPENACC)
                        #pragma acc atomic update
#elif defined(_OPENMP)
                        #pragma omp atomic update
#endif
                        F[l*3+a]+=fl;
#if defined(_OPENACC)
                        #pragma acc atomic update
#elif defined(_OPENMP)
                        #pragma omp atomic update
#endif
                        F[j*3+a]-=fl;}}}
        }}
    return Ep;}

/* ═══════════ LJ intermolecular ═══════════ */
static double compute_lj(double*F,double*vir9,
    const double*pos,const double*h,const double*hi,
    const int*nlc,const int*nll,int Na)
{
    double Ep=0;
#ifdef _OPENACC
    #pragma acc parallel loop gang \
        present(F,vir9,pos,h,hi,nlc,nll) reduction(+:Ep)
#elif defined(_OPENMP)
    #pragma omp parallel for schedule(dynamic,4) reduction(+:Ep)
#endif
    for(int i=0;i<Na;i++){
        int nni=nlc[i];
        for(int jn=0;jn<nni;jn++){
            int j=nll[i*MAX_LJ_NEIGH+jn];
            double dx=pos[j*3]-pos[i*3],dy=pos[j*3+1]-pos[i*3+1],dz=pos[j*3+2]-pos[i*3+2];
            mimg9(dx,dy,dz,hi,h);
            double r2=dx*dx+dy*dy+dz*dz;if(r2>LJ_RCUT2)continue;
            if(r2<0.25)r2=0.25;
            double ri2=1.0/r2,sr2=sig2_LJ*ri2,sr6=sr2*sr2*sr2,sr12=sr6*sr6;
            double fm=24*eps_LJ_*(2*sr12-sr6)*ri2;
            Ep+=4*eps_LJ_*(sr12-sr6)-LJ_VSHFT;
            F[i*3]-=fm*dx; F[i*3+1]-=fm*dy; F[i*3+2]-=fm*dz;
#if defined(_OPENACC)
            #pragma acc atomic update
#elif defined(_OPENMP)
            #pragma omp atomic update
#endif
            F[j*3]+=fm*dx;
#if defined(_OPENACC)
            #pragma acc atomic update
#elif defined(_OPENMP)
            #pragma omp atomic update
#endif
            F[j*3+1]+=fm*dy;
#if defined(_OPENACC)
            #pragma acc atomic update
#elif defined(_OPENMP)
            #pragma omp atomic update
#endif
            F[j*3+2]+=fm*dz;
            for(int a=0;a<3;a++){double da=(a==0?dx:a==1?dy:dz);
                for(int b=0;b<3;b++){double db=(b==0?dx:b==1?dy:dz);
#if defined(_OPENACC)
                    #pragma acc atomic update
#elif defined(_OPENMP)
                    #pragma omp atomic update
#endif
                    vir9[a*3+b]+=da*fm*db;}}}}
    return Ep;}

/* ═══════════ Total forces ═══════════ */
struct ForceResult{double Ep,Ep_rebo,Ep_lj;};
static ForceResult compute_forces(double*F,double*vir9,
    const double*pos,const double*h,const double*hi,
    const int*nlc_r,const int*nll_r,const int*nlc_l,const int*nll_l,int Na)
{
#ifdef _OPENACC
    #pragma acc parallel loop present(F)
#elif defined(_OPENMP)
    #pragma omp parallel for
#endif
    for(int i=0;i<Na*3;i++) F[i]=0;
#ifdef _OPENACC
    #pragma acc parallel loop present(vir9)
#elif defined(_OPENMP)
    #pragma omp parallel for
#endif
    for(int i=0;i<9;i++) vir9[i]=0;
    double Er=compute_rebo(F,vir9,pos,h,hi,nlc_r,nll_r,Na);
    double El=compute_lj(F,vir9,pos,h,hi,nlc_l,nll_l,Na);
    return{Er+El,Er,El};}

/* ═══════════ KE ═══════════ */
static double ke_total(const double*vel,const double*mass,int Na){
    double s=0;
#ifdef _OPENACC
    #pragma acc parallel loop present(vel,mass) reduction(+:s)
#elif defined(_OPENMP)
    #pragma omp parallel for reduction(+:s)
#endif
    for(int i=0;i<Na;i++) s+=mass[i]*(vel[i*3]*vel[i*3]+vel[i*3+1]*vel[i*3+1]+vel[i*3+2]*vel[i*3+2]);
    return 0.5*s/CONV;}
static double inst_T(double KE,int Nf){return 2*KE/(Nf*kB_);}
static double inst_P(const double*W,double KE,double V){return(2*KE+W[0]+W[4]+W[8])/(3*V)*eV2GPa;}

/* ═══════════ NPT ═══════════ */
struct NPTState{double xi,Q,Vg[9],W_,Pe,Tt;int Nf;};
static NPTState make_npt(double T,double Pe,int Na){
    int Nf=3*Na-3;NPTState s;s.xi=0;s.Q=std::max(Nf*kB_*T*100.0*100.0,1e-20);
    for(int i=0;i<9;i++)s.Vg[i]=0;
    s.W_=std::max((Nf+9)*kB_*T*1000.0*1000.0,1e-20);s.Pe=Pe;s.Tt=T;s.Nf=Nf;return s;}

struct StepResult{double Ep,Ep_rebo,Ep_lj,KE;};
static StepResult step_npt(double*pos,double*vel,double*F,double*vir9,
    double*h,double*hi,const double*mass,int Na,double dt,NPTState&npt,
    const int*nlc_r,const int*nll_r,const int*nlc_l,const int*nll_l)
{
    double hdt=0.5*dt,V=fabs(mat_det9(h));
    double KE=ke_total(vel,mass,Na);
    npt.xi+=hdt*(2*KE-npt.Nf*kB_*npt.Tt)/npt.Q; npt.xi=std::clamp(npt.xi,-0.05,0.05);
#ifdef _OPENACC
    #pragma acc update self(vir9[0:9])
#endif
    double dP=inst_P(vir9,KE,V)-npt.Pe;
    for(int a=0;a<3;a++)npt.Vg[a*4]+=hdt*V*dP/(npt.W_*eV2GPa);
    for(int a=0;a<3;a++)npt.Vg[a*4]=std::clamp(npt.Vg[a*4],-0.005,0.005);
    double eps=npt.Vg[0]*hi[0]+npt.Vg[4]*hi[4]+npt.Vg[8]*hi[8];
    double sc_nh=exp(-hdt*npt.xi),sc_pr=exp(-hdt*eps/3.0),sc_v=sc_nh*sc_pr;
#ifdef _OPENACC
    #pragma acc parallel loop present(vel,F,mass)
#elif defined(_OPENMP)
    #pragma omp parallel for
#endif
    for(int i=0;i<Na;i++){double mi=CONV/mass[i];
        vel[i*3  ]=vel[i*3  ]*sc_v+hdt*F[i*3  ]*mi;
        vel[i*3+1]=vel[i*3+1]*sc_v+hdt*F[i*3+1]*mi;
        vel[i*3+2]=vel[i*3+2]*sc_v+hdt*F[i*3+2]*mi;}
#ifdef _OPENACC
    #pragma acc parallel loop present(pos,vel,hi)
#elif defined(_OPENMP)
    #pragma omp parallel for
#endif
    for(int i=0;i<Na;i++){
        double px=pos[i*3],py=pos[i*3+1],pz=pos[i*3+2];
        double vx=vel[i*3],vy=vel[i*3+1],vz=vel[i*3+2];
        double sx=hi[0]*px+hi[1]*py+hi[2]*pz,sy=hi[3]*px+hi[4]*py+hi[5]*pz,sz=hi[6]*px+hi[7]*py+hi[8]*pz;
        double vsx=hi[0]*vx+hi[1]*vy+hi[2]*vz,vsy=hi[3]*vx+hi[4]*vy+hi[5]*vz,vsz=hi[6]*vx+hi[7]*vy+hi[8]*vz;
        sx+=dt*vsx;sy+=dt*vsy;sz+=dt*vsz;sx-=floor(sx);sy-=floor(sy);sz-=floor(sz);
        pos[i*3]=sx;pos[i*3+1]=sy;pos[i*3+2]=sz;}
    for(int a=0;a<3;a++)for(int b=0;b<3;b++)h[a*3+b]+=dt*npt.Vg[a*3+b];
    mat_inv9(h,hi);
#ifdef _OPENACC
    #pragma acc update device(h[0:9],hi[0:9])
#endif
#ifdef _OPENACC
    #pragma acc parallel loop present(pos,h)
#elif defined(_OPENMP)
    #pragma omp parallel for
#endif
    for(int i=0;i<Na;i++){
        double sx=pos[i*3],sy=pos[i*3+1],sz=pos[i*3+2];
        pos[i*3]=h[0]*sx+h[1]*sy+h[2]*sz;
        pos[i*3+1]=h[3]*sx+h[4]*sy+h[5]*sz;
        pos[i*3+2]=h[6]*sx+h[7]*sy+h[8]*sz;}
    auto fr=compute_forces(F,vir9,pos,h,hi,nlc_r,nll_r,nlc_l,nll_l,Na);
    double eps2=npt.Vg[0]*hi[0]+npt.Vg[4]*hi[4]+npt.Vg[8]*hi[8];
    double sc_v2=sc_nh*exp(-hdt*eps2/3.0);
#ifdef _OPENACC
    #pragma acc parallel loop present(vel,F,mass)
#elif defined(_OPENMP)
    #pragma omp parallel for
#endif
    for(int i=0;i<Na;i++){double mi=CONV/mass[i];
        vel[i*3  ]=(vel[i*3  ]+hdt*F[i*3  ]*mi)*sc_v2;
        vel[i*3+1]=(vel[i*3+1]+hdt*F[i*3+1]*mi)*sc_v2;
        vel[i*3+2]=(vel[i*3+2]+hdt*F[i*3+2]*mi)*sc_v2;}
    KE=ke_total(vel,mass,Na);
    npt.xi+=hdt*(2*KE-npt.Nf*kB_*npt.Tt)/npt.Q;npt.xi=std::clamp(npt.xi,-0.05,0.05);
    double V2=fabs(mat_det9(h));
#ifdef _OPENACC
    #pragma acc update self(vir9[0:9])
#endif
    dP=inst_P(vir9,KE,V2)-npt.Pe;
    for(int a=0;a<3;a++)npt.Vg[a*4]+=hdt*V2*dP/(npt.W_*eV2GPa);
    for(int a=0;a<3;a++)npt.Vg[a*4]=std::clamp(npt.Vg[a*4],-0.005,0.005);
    return{fr.Ep,fr.Ep_rebo,fr.Ep_lj,KE};}

/* ═══════════ OVITO (host) ═══════════ */
static void write_ovito(FILE*fp,int step,double dt,const double*pos,
    const double*vel,const int*mol_id,const double*h,int Na){
    fprintf(fp,"%d\n",Na);
    fprintf(fp,"Lattice=\"%.8f %.8f %.8f %.8f %.8f %.8f %.8f %.8f %.8f\" "
        "Properties=species:S:1:pos:R:3:c_mol:I:1:vx:R:1:vy:R:1:vz:R:1 Time=%.4f Step=%d pbc=\"T T T\"\n",
        h[0],h[3],h[6],h[1],h[4],h[7],h[2],h[5],h[8],step*dt,step);
    for(int i=0;i<Na;i++)
        fprintf(fp,"C %14.8f %14.8f %14.8f %5d %14.8e %14.8e %14.8e\n",
            pos[i*3],pos[i*3+1],pos[i*3+2],mol_id[i],vel[i*3],vel[i*3+1],vel[i*3+2]);}

/* ═══════════ CLI (same as CPU) ═══════════ */
std::map<std::string,std::string> parse_args(int argc,char**argv){
    std::map<std::string,std::string> o;
    for(int i=1;i<argc;i++){std::string a=argv[i];if(a.substr(0,2)!="--")continue;
        auto eq=a.find('=');std::string k=(eq!=std::string::npos)?a.substr(2,eq-2):a.substr(2);
        std::string v=(eq!=std::string::npos)?a.substr(eq+1):"";
        std::transform(k.begin(),k.end(),k.begin(),::tolower);o[k]=v;}return o;}
std::string gopt(const std::map<std::string,std::string>&o,const std::string&k,const std::string&d){
    auto it=o.find(k);return it!=o.end()?it->second:d;}
static bool file_exists(const std::string&p){struct stat st;return stat(p.c_str(),&st)==0;}
std::string unique_file(const std::string&b,const std::string&e){
    std::string p=b+e;if(!file_exists(p))return p;
    for(int n=1;n<=9999;n++){std::string c=b+"_"+std::to_string(n)+e;if(!file_exists(c))return c;}
    return b+"_9999"+e;}

/* ═══════════ Restart ═══════════ */
std::string restart_filename(const std::string&oname,int istep,int nsteps){
    std::string b=oname;auto dt=b.rfind('.');if(dt!=std::string::npos)b=b.substr(0,dt);
    auto p=b.find("ovito_traj");if(p!=std::string::npos)b.replace(p,10,"restart");
    if(istep==nsteps)return b+".rst";
    int dg=1;for(int x=nsteps;x>=10;x/=10)dg++;
    char buf[64];snprintf(buf,64,"_%0*d",dg,istep);return b+buf+".rst";}

struct RestartAirebo{int istep,Na;double h[9];NPTState npt;
    std::vector<double>pos,vel;std::vector<int>mol_id;std::vector<double>mass;bool ok;};

void write_restart_airebo(const std::string&fname,int istep,
    const std::map<std::string,std::string>&opts,
    const std::string&st,int nc,double T,double Pe,int nsteps,double dt,int seed,
    const std::string&fspec,double init_scale,
    const double*h,const NPTState&npt,
    const double*pos,const double*vel,
    const int*mol_id,const double*mass,
    int Na,int Nmol,int natom_mol){
    FILE*f=fopen(fname.c_str(),"w");
    if(!f){fprintf(stderr,"Cannot write restart: %s\n",fname.c_str());return;}
    fprintf(f,"# RESTART fuller_airebo_npt_md_serial_omp_acc\n# OPTIONS:");
    for(auto&[k,v]:opts)fprintf(f," --%s=%s",k.c_str(),v.c_str());
    fprintf(f,"\n");
    fprintf(f,"STEP %d\nNSTEPS %d\nDT %.15e\nTEMP %.15e\nPRES %.15e\n",istep,nsteps,dt,T,Pe);
    fprintf(f,"CRYSTAL %s\nNC %d\nFULLERENE %s\nINIT_SCALE %.15e\nSEED %d\n",
            st.c_str(),nc,fspec.c_str(),init_scale,seed);
    fprintf(f,"NMOL %d\nNATOM_MOL %d\nNATOM %d\n",Nmol,natom_mol,Na);
    fprintf(f,"H");for(int i=0;i<9;i++)fprintf(f," %.15e",h[i]);
    fprintf(f,"\nNPT %.15e %.15e %.15e %.15e %.15e %d\n",npt.xi,npt.Q,npt.W_,npt.Pe,npt.Tt,npt.Nf);
    fprintf(f,"VG");for(int i=0;i<9;i++)fprintf(f," %.15e",npt.Vg[i]);fprintf(f,"\n");
    for(int i=0;i<Na;i++){
        fprintf(f,"ATOM %d %d %.15e",i+1,mol_id[i],mass[i]);
        fprintf(f," %.15e %.15e %.15e",pos[i*3],pos[i*3+1],pos[i*3+2]);
        fprintf(f," %.15e %.15e %.15e\n",vel[i*3],vel[i*3+1],vel[i*3+2]);}
    fprintf(f,"END\n");fclose(f);}

RestartAirebo read_restart_airebo(const std::string&fname){
    RestartAirebo rd;rd.ok=false;
    std::ifstream f(fname);if(!f){fprintf(stderr,"Cannot read: %s\n",fname.c_str());return rd;}
    std::string line;
    for(int i=0;i<9;i++)rd.h[i]=0;rd.npt={};
    while(std::getline(f,line)){
        if(line.empty()||line[0]=='#')continue;
        std::istringstream ss(line);std::string tag;ss>>tag;
        if(tag=="STEP")ss>>rd.istep;
        else if(tag=="NATOM")ss>>rd.Na;
        else if(tag=="H"){for(int i=0;i<9;i++)ss>>rd.h[i];}
        else if(tag=="NPT"){ss>>rd.npt.xi>>rd.npt.Q>>rd.npt.W_>>rd.npt.Pe>>rd.npt.Tt>>rd.npt.Nf;}
        else if(tag=="VG"){for(int i=0;i<9;i++)ss>>rd.npt.Vg[i];}
        else if(tag=="ATOM"){
            int idx,mid;double m;ss>>idx>>mid>>m;
            double px,py,pz,vx,vy,vz;ss>>px>>py>>pz>>vx>>vy>>vz;
            rd.pos.push_back(px);rd.pos.push_back(py);rd.pos.push_back(pz);
            rd.vel.push_back(vx);rd.vel.push_back(vy);rd.vel.push_back(vz);
            rd.mol_id.push_back(mid);rd.mass.push_back(m);}
        else if(tag=="END")break;}
    rd.Na=(int)rd.pos.size()/3;rd.ok=(rd.Na>0);
    if(rd.ok)printf("  Restart loaded: %s (step %d, %d atoms)\n",fname.c_str(),rd.istep,rd.Na);
    return rd;}


/* ═══════════ MAIN ═══════════ */
int main(int argc,char**argv){
    auto opts=parse_args(argc,argv);
    if(opts.count("help")){
        printf(
            "fuller_airebo_npt_md_serial_omp_acc — AIREBO fullerene NPT-MD\n\n"
            "Options:\n"
            "  --help                  Show this help\n"
            "  --fullerene=<name>      Fullerene species (default: C60)\n"
            "  --crystal=<fcc|hcp|bcc> Crystal structure (default: fcc)\n"
            "  --cell=<nc>             Unit cell repeats (default: 3)\n"
            "  --temp=<K>              Target temperature [K] (default: 298.0)\n"
            "  --pres=<GPa>            Target pressure [GPa] (default: 0.0)\n"
            "  --step=<N>              Production steps (default: 10000)\n"
            "  --dt=<fs>               Time step [fs] (default: 0.5)\n"
            "  --init_scale=<s>        Lattice scale factor (default: 1.0)\n"
            "  --seed=<n>              Random seed (default: 42)\n"
            "  --coldstart=<N>         Cold-start steps at 4K (default: 0)\n"
            "  --warmup=<N>            Warm-up ramp steps 4K->T (default: 0)\n"
            "  --from=<step>           Averaging start step (default: auto)\n"
            "  --to=<step>             Averaging end step (default: nsteps)\n"
            "  --mon=<N>               Monitor print interval (default: auto)\n"
            "  --warmup_mon=<mode>     Warmup monitor: norm|freq|some (default: norm)\n"
            "  --ovito=<N>             OVITO XYZ output interval, 0=off (default: 0)\n"
            "  --restart=<N>           Restart save interval, 0=off (default: 0)\n"
            "  --resfile=<path>        Resume from restart file\n"
            "  --libdir=<path>         Fullerene library dir (default: FullereneLib)\n\n"
            "Examples:\n"
            "  ./prog --temp=500 --pres=1.0 --step=50000\n"
            "  ./prog --coldstart=5000 --warmup=5000 --step=20000\n"
            "  ./prog --step=10000 --ovito=100\n"
            "  ./prog --step=50000 --restart=5000\n"
            "  ./prog --resfile=restart_airebo_serial_00025000.rst\n");
        return 0;}
    std::string crystal=gopt(opts,"crystal","fcc");
    std::string st=crystal;std::transform(st.begin(),st.end(),st.begin(),::toupper);
    int nc=std::atoi(gopt(opts,"cell","3").c_str());
    double T=std::atof(gopt(opts,"temp","298.0").c_str());
    double Pe=std::atof(gopt(opts,"pres","0.0").c_str());
    int nsteps=std::atoi(gopt(opts,"step","10000").c_str());
    double dt=std::atof(gopt(opts,"dt","0.5").c_str());
    int seed=std::atoi(gopt(opts,"seed","42").c_str());
    double init_scale=std::atof(gopt(opts,"init_scale","1.0").c_str());
    std::string fspec=gopt(opts,"fullerene","C60");
    std::string libdir=gopt(opts,"libdir","FullereneLib");
    int coldstart=std::atoi(gopt(opts,"coldstart","0").c_str());
    int warmup=std::atoi(gopt(opts,"warmup","0").c_str());
    int avg_from=std::atoi(gopt(opts,"from","0").c_str());
    int avg_to=std::atoi(gopt(opts,"to","0").c_str());
    int nrec_o=std::atoi(gopt(opts,"ovito","0").c_str());
    int nrec_rst=std::atoi(gopt(opts,"restart","0").c_str());
    std::string resfile=gopt(opts,"resfile","");
    int start_step=0;
    int mon_interval=std::atoi(gopt(opts,"mon","0").c_str());
    std::string warmup_mon_mode=gopt(opts,"warmup_mon","norm");
    constexpr double T_cold=4.0;

    if(avg_to<=0)avg_to=nsteps;if(avg_from<=0)avg_from=std::max(1,nsteps-nsteps/4);
    int total_steps=coldstart+warmup+nsteps;
    int gavg_from=coldstart+warmup+avg_from,gavg_to=coldstart+warmup+avg_to;

    auto[fpath,label]=resolve_fullerene(fspec,libdir);
    MolData mol=load_cc1(fpath);
    double a0=default_a0(mol.Dmol,st,init_scale);

    /* Max molecules for crystal type */
    int Nmol_max=(st=="FCC")?4*nc*nc*nc:(st=="HCP")?2*nc*nc*nc:2*nc*nc*nc;
    int Na_max=Nmol_max*mol.natom;

    /* GPU memory check */
    long bpa=(long)3*8*3+8+4+MAX_REBO_NEIGH*4+MAX_LJ_NEIGH*4; /* per atom */
    long mem_req=(long)Na_max*bpa+9*8*2;
    long gpu_mem=0;
#ifdef _OPENACC
    int ndev=acc_get_num_devices(acc_device_nvidia);
    if(ndev>0){acc_set_device_num(0,acc_device_nvidia);
        gpu_mem=(long)acc_get_property(0,acc_device_nvidia,acc_property_memory);
        if(gpu_mem>0&&mem_req>(long)(gpu_mem*0.80)){
            printf("Error: GPU memory insufficient (need %.1f GB, have %.1f GB)\n",
                mem_req/1024.0/1024.0/1024.0,gpu_mem/1024.0/1024.0/1024.0);return 1;}}
#endif

    /* Build molecular centers -> atom positions */
    double*mol_centers=new double[Nmol_max*3];
    double h[9]={},hi[9]={},vir9[9]={};
    int Nmol;
    if(st=="FCC")Nmol=make_fcc(a0,nc,mol_centers,h);
    else if(st=="HCP")Nmol=make_hcp(a0,nc,mol_centers,h);
    else Nmol=make_bcc(a0,nc,mol_centers,h);
    int Na=Nmol*mol.natom;

    double*pos=new double[Na*3]();
    double*vel=new double[Na*3]();
    double*F  =new double[Na*3]();
    double*mass=new double[Na];
    int*mol_id=new int[Na];
    int*nlc_r=new int[Na]();
    int*nll_r=new int[Na*MAX_REBO_NEIGH]();
    int*nlc_l=new int[Na]();
    int*nll_l=new int[Na*MAX_LJ_NEIGH]();

    for(int m=0;m<Nmol;m++)for(int a=0;a<mol.natom;a++){
        int idx=m*mol.natom+a;
        pos[idx*3]=mol_centers[m*3]+mol.coords[a][0];
        pos[idx*3+1]=mol_centers[m*3+1]+mol.coords[a][1];
        pos[idx*3+2]=mol_centers[m*3+2]+mol.coords[a][2];
        mass[idx]=mC_;mol_id[idx]=m+1;}
    delete[]mol_centers;

    printf("========================================================================\n");
#if defined(_OPENACC)
    printf("  Fullerene Crystal NPT-MD — AIREBO (OpenACC GPU)\n");
#elif defined(_OPENMP)
    printf("  Fullerene Crystal NPT-MD — AIREBO (OpenMP)\n");
#else
    printf("  Fullerene Crystal NPT-MD — AIREBO (Serial)\n");
#endif
    printf("========================================================================\n");
#ifdef _OPENACC
    if(gpu_mem>0)printf("  GPU VRAM        : %.2f GB (need %.2f MB)\n",gpu_mem/1024.0/1024.0/1024.0,mem_req/1024.0/1024.0);
#endif
    printf("  Fullerene       : %s (%d atoms/mol)\n",label.c_str(),mol.natom);
    printf("  Crystal         : %s %dx%dx%d  Nmol=%d  Natom=%d\n",st.c_str(),nc,nc,nc,Nmol,Na);
    printf("  a0=%.3f A  T=%.1f K  P=%.4f GPa  dt=%.3f fs\n",a0,T,Pe,dt);
    printf("  Production      : %d steps  avg=%d-%d  Total=%d\n",nsteps,avg_from,avg_to,total_steps);
    printf("========================================================================\n\n");

    mat_inv9(h,hi);
    double T_init=(coldstart>0||warmup>0)?T_cold:T;
    std::mt19937 rng(seed);std::normal_distribution<double> g(0,1);
    for(int i=0;i<Na;i++){double sv=sqrt(kB_*T_init*CONV/mass[i]);
        for(int a=0;a<3;a++)vel[i*3+a]=sv*g(rng);}
    double vcm[3]={0,0,0};
    for(int i=0;i<Na;i++)for(int a=0;a<3;a++)vcm[a]+=vel[i*3+a];
    for(int a=0;a<3;a++)vcm[a]/=Na;
    for(int i=0;i<Na;i++)for(int a=0;a<3;a++)vel[i*3+a]-=vcm[a];

    NPTState npt=make_npt(T,Pe,Na);npt.Tt=T_init;

    /* --- Restart file loading --- */
    if(!resfile.empty()){
        auto rd=read_restart_airebo(resfile);
        if(rd.ok){
            start_step=rd.istep;for(int i=0;i<9;i++)h[i]=rd.h[i];mat_inv9(h,hi);npt=rd.npt;
            for(int i=0;i<Na&&i<rd.Na;i++){
                pos[i*3]=rd.pos[i*3];pos[i*3+1]=rd.pos[i*3+1];pos[i*3+2]=rd.pos[i*3+2];
                vel[i*3]=rd.vel[i*3];vel[i*3+1]=rd.vel[i*3+1];vel[i*3+2]=rd.vel[i*3+2];
                mol_id[i]=rd.mol_id[i];mass[i]=rd.mass[i];}
            printf("  Restarting from global step %d\n",start_step);}}

    build_nlist_rebo(pos,h,hi,Na,mol_id,nlc_r,nll_r);
    build_nlist_lj(pos,h,hi,Na,mol_id,nlc_l,nll_l);

    /* Mode tag for filenames */
#if defined(_OPENACC)
    std::string mode_tag="gpu";
#elif defined(_OPENMP)
    std::string mode_tag="omp";
#else
    std::string mode_tag="serial";
#endif

    /* ═══ OpenACC data region ═══ */
#ifdef _OPENACC
    #pragma acc data \
        copy(pos[0:Na*3],vel[0:Na*3],F[0:Na*3]) \
        copyin(mass[0:Na],mol_id[0:Na]) \
        copy(h[0:9],hi[0:9],vir9[0:9]) \
        copyin(nlc_r[0:Na],nll_r[0:Na*MAX_REBO_NEIGH]) \
        copyin(nlc_l[0:Na],nll_l[0:Na*MAX_LJ_NEIGH])
    {
#endif
        apply_pbc(pos,h,hi,Na);
        compute_forces(F,vir9,pos,h,hi,nlc_r,nll_r,nlc_l,nll_l,Na);

        int prn=(mon_interval>0)?mon_interval:std::max(1,total_steps/50);
        int prn_pre=prn;
        if(coldstart+warmup>0){int div=100;
            if(warmup_mon_mode=="freq")div=10;else if(warmup_mon_mode=="some")div=1000;
            prn_pre=std::max(1,(coldstart+warmup)/div);}
        int nlup=20;
        double sT=0,sP=0,sa=0,sR=0,sL=0,sE=0;int nav=0;
        auto t0=std::chrono::steady_clock::now();
        std::string ovito_file=unique_file("ovito_traj_airebo_"+mode_tag,".xyz");
        FILE*io_o=nrec_o>0?fopen(ovito_file.c_str(),"w"):nullptr;

        printf("  %8s %5s %7s %9s %8s %11s %11s %11s %7s\n",
            "step","phase","T[K]","P[GPa]","a[A]","E_REBO","E_LJ","E_total","t[s]");

        bool stop_requested=false;
        std::string rst_base=nrec_o>0?ovito_file:("restart_airebo_"+mode_tag);
        for(int gstep=start_step+1;gstep<=total_steps;gstep++){
            const char*phase=gstep<=coldstart?"COLD":gstep<=coldstart+warmup?"WARM":"PROD";
            int cur_prn=(gstep<=coldstart+warmup)?prn_pre:prn;

            if(gstep<=coldstart)npt.Tt=T_cold;
            else if(gstep<=coldstart+warmup)npt.Tt=T_cold+(T-T_cold)*double(gstep-coldstart)/double(warmup);
            else npt.Tt=T;
            if(coldstart>0&&gstep==coldstart+1){npt.xi=0;for(int i=0;i<9;i++)npt.Vg[i]=0;}
            if(gstep<=coldstart){for(int i=0;i<9;i++)npt.Vg[i]=0;}

            if(gstep%nlup==0){
#ifdef _OPENACC
                #pragma acc update self(pos[0:Na*3])
#endif
                mat_inv9(h,hi);
                build_nlist_rebo(pos,h,hi,Na,mol_id,nlc_r,nll_r);
                build_nlist_lj(pos,h,hi,Na,mol_id,nlc_l,nll_l);
#ifdef _OPENACC
                #pragma acc update device(nlc_r[0:Na],nll_r[0:Na*MAX_REBO_NEIGH])
                #pragma acc update device(nlc_l[0:Na],nll_l[0:Na*MAX_LJ_NEIGH])
                #pragma acc update device(hi[0:9])
#endif
            }

            auto sr=step_npt(pos,vel,F,vir9,h,hi,mass,Na,dt,npt,nlc_r,nll_r,nlc_l,nll_l);
            double V=fabs(mat_det9(h)),Tn=inst_T(sr.KE,npt.Nf),Pn=inst_P(vir9,sr.KE,V);

            if((gstep<=coldstart||gstep<=coldstart+warmup)&&Tn>0.1){
                double tgt=(gstep<=coldstart)?T_cold:npt.Tt;
                double scale=sqrt(std::max(tgt,0.1)/Tn);
#ifdef _OPENACC
                #pragma acc parallel loop present(vel)
#elif defined(_OPENMP)
                #pragma omp parallel for
#endif
                for(int i=0;i<Na;i++){vel[i*3]*=scale;vel[i*3+1]*=scale;vel[i*3+2]*=scale;}
                sr.KE=ke_total(vel,mass,Na);Tn=inst_T(sr.KE,npt.Nf);
                npt.xi=0;if(gstep<=coldstart)for(int i=0;i<9;i++)npt.Vg[i]=0;}

            double an=h[0]/nc;
            if(gstep>=gavg_from&&gstep<=gavg_to){
                sT+=Tn;sP+=Pn;sa+=an;sR+=sr.Ep_rebo/Nmol;sL+=sr.Ep_lj/Nmol;sE+=sr.Ep/Nmol;nav++;}

            if(io_o&&gstep%nrec_o==0){
#ifdef _OPENACC
                #pragma acc update self(pos[0:Na*3],vel[0:Na*3])
#endif
                write_ovito(io_o,gstep,dt,pos,vel,mol_id,h,Na);fflush(io_o);}

            /* restart save */
            if(nrec_rst>0&&(gstep%nrec_rst==0||gstep==total_steps)){
#ifdef _OPENACC
                #pragma acc update self(pos[0:Na*3],vel[0:Na*3])
#endif
                auto rfn=restart_filename(rst_base,gstep,total_steps);
                write_restart_airebo(rfn,gstep,opts,st,nc,T,Pe,nsteps,dt,seed,
                    fspec,init_scale,h,npt,pos,vel,mol_id,mass,Na,Nmol,mol.natom);
                if(stop_requested){
                    printf("\n  *** Stopped at restart checkpoint (global step %d) ***\n",gstep);
                    break;}}

            if(gstep%cur_prn==0||gstep==total_steps){
                if(dir_exists("abort.md")){
                    printf("\n  *** abort.md detected at step %d ***\n",gstep);
                    if(nrec_rst>0){
#ifdef _OPENACC
                        #pragma acc update self(pos[0:Na*3],vel[0:Na*3])
#endif
                        auto rfn=restart_filename(rst_base,gstep,total_steps);
                        write_restart_airebo(rfn,gstep,opts,st,nc,T,Pe,nsteps,dt,seed,
                            fspec,init_scale,h,npt,pos,vel,mol_id,mass,Na,Nmol,mol.natom);}
                    break;}
                if(!stop_requested&&dir_exists("stop.md")){
                    stop_requested=true;
                    printf("\n  *** stop.md detected at step %d — will stop at next checkpoint ***\n",gstep);
                    if(nrec_rst==0)break;}
                double el=std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count();
                printf("  %8d %5s %7.1f %9.3f %8.3f %11.4f %11.4f %11.4f %7.0f\n",
                    gstep,phase,Tn,Pn,an,sr.Ep_rebo/Nmol,sr.Ep_lj/Nmol,sr.Ep/Nmol,el);}
        }
        if(io_o)fclose(io_o);

        if(nav>0){printf("\n  Averages (%d): T=%.2f P=%.4f a=%.4f REBO=%.4f LJ=%.4f Total=%.4f\n",
            nav,sT/nav,sP/nav,sa/nav,sR/nav,sL/nav,sE/nav);}
        printf("  Done (%.1f sec)\n",std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count());
#ifdef _OPENACC
    } /* end acc data */
#endif

    delete[]pos;delete[]vel;delete[]F;delete[]mass;delete[]mol_id;
    delete[]nlc_r;delete[]nll_r;delete[]nlc_l;delete[]nll_l;
    return 0;
}
