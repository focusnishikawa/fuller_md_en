// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2025, Takeshi Nishikawa
/*===========================================================================
  fuller_LJ_npt_md_serial_omp_acc.cpp — Fullerene Crystal NPT-MD
  (Rigid-body LJ, Serial / OpenMP / OpenACC GPU)

  Compile:
    Serial:  g++ -std=c++17 -O3 -Wno-unknown-pragmas \
               -o fuller_LJ_npt_md_serial fuller_LJ_npt_md_serial_omp_acc.cpp -lm
    OpenMP:  g++ -std=c++17 -O3 -fopenmp -Wno-unknown-pragmas \
               -o fuller_LJ_npt_md_omp fuller_LJ_npt_md_serial_omp_acc.cpp -lm
    OpenACC: nvc++ -std=c++17 -O3 -acc -gpu=cc80 -Minfo=accel \
               -o fuller_LJ_npt_md_gpu fuller_LJ_npt_md_serial_omp_acc.cpp -lm

  Runtime Options (all in --key=value format):
    --help                  Show this help
    --fullerene=<name>      Fullerene species (default: C60)
    --crystal=<fcc|hcp|bcc> Crystal structure (default: fcc)
    --cell=<nc>             Unit cell repeat count (default: 3)
    --temp=<T_K>            Target temperature [K] (default: 298.0)
    --pres=<P_GPa>          Target pressure [GPa] (default: 0.0)
    --step=<N>              Number of production steps (default: 10000)
    --dt=<fs>               Time step [fs] (default: 1.0)
    --init_scale=<s>        Lattice constant scale factor (default: 1.0)
    --seed=<n>              Random seed (default: 42)
    --coldstart=<N>         Cold start (4K) steps (default: 0)
    --warmup=<N>            Warmup steps 4K->T (default: 0)
    --from=<step>           Averaging start step (default: 3/4 point of production)
    --to=<step>             Averaging end step (default: nsteps)
    --mon=<N>               Monitoring output interval (default: auto)
    --warmup_mon=<mode>     Output frequency during warmup norm|freq|some (default: norm)
    --ovito=<N>             OVITO XYZ output interval (0=disabled, default: 0)
    --ofile=<filename>      OVITO output filename (default: auto-generated)
    --restart=<N>           Restart save interval (0=disabled, default: 0)
    --resfile=<path>        Resume from restart file
    --libdir=<path>         Fullerene library (default: FullereneLib)

  Execution Examples:
    # Basic run (C60 FCC 3x3x3, 298K, 10000 steps)
    ./fuller_LJ_npt_md_serial

    # Specify temperature, pressure, and number of steps
    ./fuller_LJ_npt_md_omp --temp=500 --pres=1.0 --step=50000

    # Cold start + warmup + production
    ./fuller_LJ_npt_md_serial --coldstart=2000 --warmup=3000 --step=20000

    # OVITO output (write to XYZ file every 100 steps)
    ./fuller_LJ_npt_md_omp --step=10000 --ovito=100

    # Restart save (every 5000 steps + final step)
    ./fuller_LJ_npt_md_serial --step=50000 --restart=5000

    # Resume from restart file
    ./fuller_LJ_npt_md_serial --resfile=restart_LJ_serial_00010000.rst

    # Use OVITO + restart simultaneously
    ./fuller_LJ_npt_md_gpu --step=100000 --ovito=500 --restart=10000

    # C84 fullerene, large cell
    ./fuller_LJ_npt_md_omp --fullerene=C84 --cell=5 --step=20000

  Stop Control:
    Create the following in the current directory during execution to control behavior:
    - abort.md: Stop immediately (saves restart if enabled, then exits)
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

/* ═══════════ Constants ═══════════ */
constexpr double CONV=9.64853321e-3, kB=8.617333262e-5;
constexpr double eV2GPa=160.21766208, eV2kcalmol=23.06054783;
constexpr double sigma_LJ=3.431, eps_LJ=2.635e-3;
constexpr double RCUT=3.0*sigma_LJ, RCUT2=RCUT*RCUT;
constexpr double sig2_LJ=sigma_LJ*sigma_LJ, mC=12.011;
constexpr double _sr_v=1.0/3.0, _sr2_v=_sr_v*_sr_v, _sr6_v=_sr2_v*_sr2_v*_sr2_v;
constexpr double VSHFT=4.0*eps_LJ*(_sr6_v*_sr6_v-_sr6_v);
constexpr int MAX_NEIGH=80, VECTOR_LENGTH=128;
constexpr int MAX_NATOM=84; /* C84 support */

/* ═══════════ Flat H-matrix ops ═══════════ */
#define H_(h,i,j) ((h)[3*(i)+(j)])
#pragma acc routine seq
static inline double mat_det9(const double*h){
    return H_(h,0,0)*(H_(h,1,1)*H_(h,2,2)-H_(h,1,2)*H_(h,2,1))
          -H_(h,0,1)*(H_(h,1,0)*H_(h,2,2)-H_(h,1,2)*H_(h,2,0))
          +H_(h,0,2)*(H_(h,1,0)*H_(h,2,1)-H_(h,1,1)*H_(h,2,0));}
#pragma acc routine seq
static inline double mat_tr9(const double*h){return h[0]+h[4]+h[8];}
static void mat_inv9(const double*h,double*hi){
    double d=mat_det9(h),id=1.0/d;
    hi[0]=id*(h[4]*h[8]-h[5]*h[7]); hi[1]=id*(h[2]*h[7]-h[1]*h[8]); hi[2]=id*(h[1]*h[5]-h[2]*h[4]);
    hi[3]=id*(h[5]*h[6]-h[3]*h[8]); hi[4]=id*(h[0]*h[8]-h[2]*h[6]); hi[5]=id*(h[2]*h[3]-h[0]*h[5]);
    hi[6]=id*(h[3]*h[7]-h[4]*h[6]); hi[7]=id*(h[1]*h[6]-h[0]*h[7]); hi[8]=id*(h[0]*h[4]-h[1]*h[3]);}

/* ═══════════ PBC (device) ═══════════ */
#pragma acc routine seq
static inline void mimg_flat(double&dx,double&dy,double&dz,const double*hi,const double*h){
    double s0=hi[0]*dx+hi[1]*dy+hi[2]*dz;
    double s1=hi[3]*dx+hi[4]*dy+hi[5]*dz;
    double s2=hi[6]*dx+hi[7]*dy+hi[8]*dz;
    s0-=round(s0);s1-=round(s1);s2-=round(s2);
    dx=h[0]*s0+h[1]*s1+h[2]*s2;
    dy=h[3]*s0+h[4]*s1+h[5]*s2;
    dz=h[6]*s0+h[7]*s1+h[8]*s2;}

static void apply_pbc(double*pos,const double*h,const double*hi,int N){
#ifdef _OPENACC
    #pragma acc parallel loop present(pos,h,hi)
#elif defined(_OPENMP)
    #pragma omp parallel for
#endif
    for(int i=0;i<N;i++){
        double px=pos[i*3],py=pos[i*3+1],pz=pos[i*3+2];
        double s0=hi[0]*px+hi[1]*py+hi[2]*pz;
        double s1=hi[3]*px+hi[4]*py+hi[5]*pz;
        double s2=hi[6]*px+hi[7]*py+hi[8]*pz;
        s0-=floor(s0);s1-=floor(s1);s2-=floor(s2);
        pos[i*3]=h[0]*s0+h[1]*s1+h[2]*s2;
        pos[i*3+1]=h[3]*s0+h[4]*s1+h[5]*s2;
        pos[i*3+2]=h[6]*s0+h[7]*s1+h[8]*s2;}}

/* ═══════════ Quaternion (device) ═══════════ */
#pragma acc routine seq
static inline void q2R_flat(const double*q,double*R){
    double w=q[0],x=q[1],y=q[2],z=q[3];
    R[0]=1-2*(y*y+z*z);R[1]=2*(x*y-w*z);R[2]=2*(x*z+w*y);
    R[3]=2*(x*y+w*z);R[4]=1-2*(x*x+z*z);R[5]=2*(y*z-w*x);
    R[6]=2*(x*z-w*y);R[7]=2*(y*z+w*x);R[8]=1-2*(x*x+y*y);}
#pragma acc routine seq
static inline void qmul_flat(const double*a,const double*b,double*o){
    o[0]=a[0]*b[0]-a[1]*b[1]-a[2]*b[2]-a[3]*b[3];
    o[1]=a[0]*b[1]+a[1]*b[0]+a[2]*b[3]-a[3]*b[2];
    o[2]=a[0]*b[2]-a[1]*b[3]+a[2]*b[0]+a[3]*b[1];
    o[3]=a[0]*b[3]+a[1]*b[2]-a[2]*b[1]+a[3]*b[0];}
#pragma acc routine seq
static inline void qnorm_flat(double*q){
    double n=sqrt(q[0]*q[0]+q[1]*q[1]+q[2]*q[2]+q[3]*q[3]),inv=1.0/n;
    q[0]*=inv;q[1]*=inv;q[2]*=inv;q[3]*=inv;}
#pragma acc routine seq
static inline void omega2dq_flat(double wx,double wy,double wz,double dt,double*dq){
    double wm=sqrt(wx*wx+wy*wy+wz*wz),th=wm*dt*0.5;
    if(th<1e-14){dq[0]=1.0;dq[1]=0.5*dt*wx;dq[2]=0.5*dt*wy;dq[3]=0.5*dt*wz;}
    else{double s=sin(th)/wm;dq[0]=cos(th);dq[1]=s*wx;dq[2]=s*wy;dq[3]=s*wz;}}

/* ═══════════ cc1 reader (host) ═══════════ */
struct MolData{double coords[MAX_NATOM*3]; int natom; double Rmol,Dmol,I0,Mmol;};
MolData load_cc1(const std::string&path){
    MolData md; std::ifstream f(path);
    if(!f){fprintf(stderr,"Error: cannot open %s\n",path.c_str());exit(1);}
    f>>md.natom;
    if(md.natom>MAX_NATOM){fprintf(stderr,"Error: natom=%d > MAX_NATOM=%d\n",md.natom,MAX_NATOM);exit(1);}
    double tmp[MAX_NATOM][3];
    for(int i=0;i<md.natom;i++){
        std::string line; std::getline(f,line);
        while(line.empty()&&f.good()) std::getline(f,line);
        std::istringstream ss(line);
        std::string elem; int idx; double x,y,z; int flag;
        ss>>elem>>idx>>x>>y>>z>>flag; tmp[i][0]=x;tmp[i][1]=y;tmp[i][2]=z;}
    double cm[3]={0,0,0};
    for(int i=0;i<md.natom;i++) for(int a=0;a<3;a++) cm[a]+=tmp[i][a];
    for(int a=0;a<3;a++) cm[a]/=md.natom;
    for(int i=0;i<md.natom;i++) for(int a=0;a<3;a++) tmp[i][a]-=cm[a];
    md.Rmol=0; md.Dmol=0;
    for(int i=0;i<md.natom;i++){
        double r2=tmp[i][0]*tmp[i][0]+tmp[i][1]*tmp[i][1]+tmp[i][2]*tmp[i][2];
        double r=sqrt(r2); if(r>md.Rmol) md.Rmol=r;
        for(int j=i+1;j<md.natom;j++){
            double dx=tmp[i][0]-tmp[j][0],dy=tmp[i][1]-tmp[j][1],dz=tmp[i][2]-tmp[j][2];
            double d=sqrt(dx*dx+dy*dy+dz*dz); if(d>md.Dmol) md.Dmol=d;}}
    double Ixx=0,Iyy=0,Izz=0;
    for(int i=0;i<md.natom;i++){
        double r2=tmp[i][0]*tmp[i][0]+tmp[i][1]*tmp[i][1]+tmp[i][2]*tmp[i][2];
        Ixx+=mC*(r2-tmp[i][0]*tmp[i][0]); Iyy+=mC*(r2-tmp[i][1]*tmp[i][1]); Izz+=mC*(r2-tmp[i][2]*tmp[i][2]);}
    md.I0=(Ixx+Iyy+Izz)/3.0; md.Mmol=md.natom*mC;
    for(int i=0;i<md.natom;i++) for(int a=0;a<3;a++) md.coords[i*3+a]=tmp[i][a];
    return md;}

/* ═══════════ Fullerene resolver (host) ═══════════ */
std::pair<std::string,std::string> resolve_fullerene(const std::string&spec,const std::string&lib="FullereneLib"){
    std::string sl=spec; std::transform(sl.begin(),sl.end(),sl.begin(),::tolower);
    if(sl=="buckyball"||sl=="c60"||sl=="c60:ih") return{lib+"/C60-76/C60-Ih.cc1","C60(Ih)"};
    if(sl=="c70"||sl=="c70:d5h") return{lib+"/C60-76/C70-D5h.cc1","C70(D5h)"};
    if(sl=="c72"||sl=="c72:d6d") return{lib+"/C60-76/C72-D6d.cc1","C72(D6d)"};
    if(sl=="c74"||sl=="c74:d3h") return{lib+"/C60-76/C74-D3h.cc1","C74(D3h)"};
    if(sl.substr(0,4)=="c76:"&&sl.size()>4){std::string sym=spec.substr(4);return{lib+"/C60-76/C76-"+sym+".cc1","C76("+sym+")"};}
    if(sl.substr(0,4)=="c84:"){
        std::string rest=spec.substr(4); auto c=rest.find(':');
        if(c!=std::string::npos){int n=std::atoi(rest.substr(0,c).c_str());std::string sym=rest.substr(c+1);
            char buf[64];snprintf(buf,64,"C84-No.%02d-%s.cc1",n,sym.c_str());return{lib+"/C84/"+buf,"C84 No."+std::to_string(n)};}
        else{int n=std::atoi(rest.c_str());char pfx[32];snprintf(pfx,32,"C84-No.%02d-",n);
            std::string dpath=lib+"/C84";DIR*dp=opendir(dpath.c_str());
            if(dp){struct dirent*ep;while((ep=readdir(dp))){std::string fn=ep->d_name;
                if(fn.find(pfx)==0&&fn.size()>4&&fn.substr(fn.size()-4)==".cc1"){closedir(dp);return{dpath+"/"+fn,"C84 No."+std::to_string(n)};}}
                closedir(dp);}}}
    fprintf(stderr,"Unknown fullerene: %s\n",spec.c_str()); exit(1);}

/* ═══════════ Crystal structures (host, flat output) ═══════════ */
static int make_fcc(double a,int nc,double*pos,double*h){
    double bas[4][3]={{0,0,0},{.5*a,.5*a,0},{.5*a,0,.5*a},{0,.5*a,.5*a}};
    int n=0;
    for(int ix=0;ix<nc;ix++)for(int iy=0;iy<nc;iy++)for(int iz=0;iz<nc;iz++)
        for(int b=0;b<4;b++){pos[n*3]=a*ix+bas[b][0];pos[n*3+1]=a*iy+bas[b][1];pos[n*3+2]=a*iz+bas[b][2];n++;}
    for(int i=0;i<9;i++)h[i]=0; h[0]=h[4]=h[8]=nc*a; return n;}
static int make_hcp(double a,int nc,double*pos,double*h){
    double c=a*sqrt(8.0/3.0);
    double a1[3]={a,0,0},a2[3]={a/2,a*sqrt(3.0)/2,0},a3[3]={0,0,c};
    double bas[2][3]={{0,0,0},{1.0/3,2.0/3,0.5}};
    int n=0;
    for(int ix=0;ix<nc;ix++)for(int iy=0;iy<nc;iy++)for(int iz=0;iz<nc;iz++)
        for(int b=0;b<2;b++){double fx=ix+bas[b][0],fy=iy+bas[b][1],fz=iz+bas[b][2];
            pos[n*3]=fx*a1[0]+fy*a2[0]+fz*a3[0];pos[n*3+1]=fx*a1[1]+fy*a2[1]+fz*a3[1];
            pos[n*3+2]=fx*a1[2]+fy*a2[2]+fz*a3[2];n++;}
    for(int i=0;i<9;i++)h[i]=0; h[0]=nc*a1[0];h[3]=nc*a1[1];h[1]=nc*a2[0];h[4]=nc*a2[1];h[8]=nc*a3[2]; return n;}
static int make_bcc(double a,int nc,double*pos,double*h){
    double bas[2][3]={{0,0,0},{.5*a,.5*a,.5*a}};
    int n=0;
    for(int ix=0;ix<nc;ix++)for(int iy=0;iy<nc;iy++)for(int iz=0;iz<nc;iz++)
        for(int b=0;b<2;b++){pos[n*3]=a*ix+bas[b][0];pos[n*3+1]=a*iy+bas[b][1];pos[n*3+2]=a*iz+bas[b][2];n++;}
    for(int i=0;i<9;i++)h[i]=0; h[0]=h[4]=h[8]=nc*a; return n;}
static double default_a0(double dmax,const std::string&st,double s){
    double m=1.4,a0;
    if(st=="FCC")a0=dmax*sqrt(2.0)*m; else if(st=="HCP")a0=dmax*m; else a0=dmax*2.0/sqrt(3.0)*m;
    return a0*s;}

/* ═══════════ Symmetric neighbor list (host) ═══════════ */
static void nlist_build_sym(const double*pos,const double*h,const double*hi,
    int N,double rmcut,int*nl_count,int*nl_list){
    double rc2=(rmcut+3.0)*(rmcut+3.0);
    for(int i=0;i<N;i++) nl_count[i]=0;
    for(int i=0;i<N;i++) for(int j=i+1;j<N;j++){
        double dx=pos[j*3]-pos[i*3],dy=pos[j*3+1]-pos[i*3+1],dz=pos[j*3+2]-pos[i*3+2];
        mimg_flat(dx,dy,dz,hi,h);
        if(dx*dx+dy*dy+dz*dz<rc2){
            int ci=nl_count[i],cj=nl_count[j];
            if(ci<MAX_NEIGH){nl_list[i*MAX_NEIGH+ci]=j;nl_count[i]++;}
            if(cj<MAX_NEIGH){nl_list[j*MAX_NEIGH+cj]=i;nl_count[j]++;}
        }}}

/* ═══════════ Force kernel ═══════════ */
static double forces(double*Fv,double*Tv,double*Wm9,
    const double*pos,const double*qv,const double*body,
    const double*h,const double*hi,const int*nl_count,const int*nl_list,
    int N,int natom,double rmcut2,double*lab)
{
#ifdef _OPENACC
    #pragma acc parallel loop gang vector_length(VECTOR_LENGTH) present(qv,body,lab)
#elif defined(_OPENMP)
    #pragma omp parallel for schedule(static)
#endif
    for(int i=0;i<N;i++){
        double R[9]; q2R_flat(&qv[i*4],R);
#ifdef _OPENACC
        #pragma acc loop vector
#endif
        for(int a=0;a<natom;a++){
            double bx=body[a*3],by=body[a*3+1],bz=body[a*3+2];
            int idx=i*natom*3+a*3;
            lab[idx]=R[0]*bx+R[1]*by+R[2]*bz;
            lab[idx+1]=R[3]*bx+R[4]*by+R[5]*bz;
            lab[idx+2]=R[6]*bx+R[7]*by+R[8]*bz;}}
#ifdef _OPENACC
    #pragma acc parallel loop present(Fv,Tv)
#elif defined(_OPENMP)
    #pragma omp parallel for
#endif
    for(int i=0;i<N*3;i++){Fv[i]=0;Tv[i]=0;}
#ifdef _OPENACC
    #pragma acc parallel loop present(Wm9)
#elif defined(_OPENMP)
    #pragma omp parallel for
#endif
    for(int i=0;i<9;i++) Wm9[i]=0;

    double Ep=0;
#ifdef _OPENACC
    #pragma acc parallel loop gang vector_length(VECTOR_LENGTH) \
        present(pos,lab,Fv,Tv,Wm9,h,hi,nl_count,nl_list) reduction(+:Ep)
#elif defined(_OPENMP)
    #pragma omp parallel for schedule(dynamic,1) reduction(+:Ep)
#endif
    for(int i=0;i<N;i++){
        double fi0=0,fi1=0,fi2=0,ti0=0,ti1=0,ti2=0,my_Ep=0;
        double w00=0,w01=0,w02=0,w10=0,w11=0,w12=0,w20=0,w21=0,w22=0;
        int nni=nl_count[i];
        for(int k=0;k<nni;k++){
            int j=nl_list[i*MAX_NEIGH+k];
            double dmx=pos[j*3]-pos[i*3],dmy=pos[j*3+1]-pos[i*3+1],dmz=pos[j*3+2]-pos[i*3+2];
            mimg_flat(dmx,dmy,dmz,hi,h);
            if(dmx*dmx+dmy*dmy+dmz*dmz>rmcut2) continue;
#ifdef _OPENACC
            #pragma acc loop vector \
                reduction(+:fi0,fi1,fi2,ti0,ti1,ti2,my_Ep,w00,w01,w02,w10,w11,w12,w20,w21,w22)
#endif
            for(int ai=0;ai<natom;ai++){
                int ia=i*natom*3+ai*3;
                double rax=lab[ia],ray=lab[ia+1],raz=lab[ia+2];
                for(int bj=0;bj<natom;bj++){
                    int jb=j*natom*3+bj*3;
                    double ddx=dmx+lab[jb]-rax,ddy=dmy+lab[jb+1]-ray,ddz=dmz+lab[jb+2]-raz;
                    double r2=ddx*ddx+ddy*ddy+ddz*ddz;
                    if(r2<RCUT2){
                        if(r2<0.25)r2=0.25;
                        double ri2=1.0/r2,sr2=sig2_LJ*ri2,sr6=sr2*sr2*sr2,sr12=sr6*sr6;
                        double fm=24.0*eps_LJ*(2.0*sr12-sr6)*ri2;
                        double fx=fm*ddx,fy=fm*ddy,fz=fm*ddz;
                        fi0-=fx;fi1-=fy;fi2-=fz;
                        ti0-=(ray*fz-raz*fy); ti1-=(raz*fx-rax*fz); ti2-=(rax*fy-ray*fx);
                        my_Ep+=0.5*(4.0*eps_LJ*(sr12-sr6)-VSHFT);
                        w00+=0.5*ddx*fx;w01+=0.5*ddx*fy;w02+=0.5*ddx*fz;
                        w10+=0.5*ddy*fx;w11+=0.5*ddy*fy;w12+=0.5*ddy*fz;
                        w20+=0.5*ddz*fx;w21+=0.5*ddz*fy;w22+=0.5*ddz*fz;}}}}
        Fv[i*3]=fi0;Fv[i*3+1]=fi1;Fv[i*3+2]=fi2;
        Tv[i*3]=ti0;Tv[i*3+1]=ti1;Tv[i*3+2]=ti2;
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
        Ep+=my_Ep;}
    return Ep;}

/* ═══════════ KE (reduction) ═══════════ */
static double ke_trans(const double*vel,int N,double Mmol){
    double s=0;
#ifdef _OPENACC
    #pragma acc parallel loop present(vel) reduction(+:s)
#elif defined(_OPENMP)
    #pragma omp parallel for reduction(+:s)
#endif
    for(int i=0;i<N;i++) s+=vel[i*3]*vel[i*3]+vel[i*3+1]*vel[i*3+1]+vel[i*3+2]*vel[i*3+2];
    return 0.5*Mmol*s/CONV;}
static double ke_rot(const double*omg,int N,double I0){
    double s=0;
#ifdef _OPENACC
    #pragma acc parallel loop present(omg) reduction(+:s)
#elif defined(_OPENMP)
    #pragma omp parallel for reduction(+:s)
#endif
    for(int i=0;i<N;i++) s+=omg[i*3]*omg[i*3]+omg[i*3+1]*omg[i*3+1]+omg[i*3+2]*omg[i*3+2];
    return 0.5*I0*s/CONV;}
static inline double inst_T(double KE,int Nf){return 2*KE/(Nf*kB);}
static inline double inst_P(const double*W,double KEt,double V){return(2*KEt+W[0]+W[4]+W[8])/(3*V)*eV2GPa;}

/* ═══════════ NPT ═══════════ */
struct NPTState{double xi,Q,Vg[9],W,Pe,Tt; int Nf;};
static NPTState make_npt(double T,double Pe,int N){
    int Nf=6*N-3; NPTState s;
    s.xi=0; s.Q=std::max(Nf*kB*T*100.0*100.0,1e-20);
    for(int i=0;i<9;i++)s.Vg[i]=0;
    s.W=std::max((Nf+9)*kB*T*1000.0*1000.0,1e-20); s.Pe=Pe; s.Tt=T; s.Nf=Nf;
    return s;}

/* ═══════════ NPT step ═══════════ */
static std::pair<double,double>
step_npt(double*pos,double*vel,double*qv,double*omg,
    double*Fv,double*Tv,double*Wm9,double*h,double*hi,
    const double*body,double I0,double Mmol,
    int N,int natom,double rmcut2,double dt,NPTState&npt,
    const int*nl_count,const int*nl_list,double*lab)
{
    double hdt=0.5*dt;
    mat_inv9(h,hi);
#ifdef _OPENACC
    #pragma acc update device(hi[0:9])
#endif
    double V=fabs(mat_det9(h));
    double kt=ke_trans(vel,N,Mmol),kr=ke_rot(omg,N,I0),KE=kt+kr;
    npt.xi+=hdt*(2*KE-npt.Nf*kB*npt.Tt)/npt.Q;
    npt.xi=std::clamp(npt.xi,-0.1,0.1);
#ifdef _OPENACC
    #pragma acc update self(Wm9[0:9])
#endif
    double dP=inst_P(Wm9,kt,V)-npt.Pe;
    for(int a=0;a<3;a++) npt.Vg[a*4]+=hdt*V*dP/(npt.W*eV2GPa);
    for(int a=0;a<3;a++) npt.Vg[a*4]=std::clamp(npt.Vg[a*4],-0.01,0.01);
    double eps_tr=npt.Vg[0]*hi[0]+npt.Vg[4]*hi[4]+npt.Vg[8]*hi[8];
    double sc_nh=exp(-hdt*npt.xi),sc_pr=exp(-hdt*eps_tr/3.0),sc_v=sc_nh*sc_pr;
    double cF=CONV/Mmol,cT=CONV/I0;
#ifdef _OPENACC
    #pragma acc parallel loop present(vel,omg,Fv,Tv)
#elif defined(_OPENMP)
    #pragma omp parallel for
#endif
    for(int i=0;i<N;i++){
        vel[i*3  ]=vel[i*3  ]*sc_v+hdt*Fv[i*3  ]*cF;
        vel[i*3+1]=vel[i*3+1]*sc_v+hdt*Fv[i*3+1]*cF;
        vel[i*3+2]=vel[i*3+2]*sc_v+hdt*Fv[i*3+2]*cF;
        omg[i*3  ]=omg[i*3  ]*sc_nh+hdt*Tv[i*3  ]*cT;
        omg[i*3+1]=omg[i*3+1]*sc_nh+hdt*Tv[i*3+1]*cT;
        omg[i*3+2]=omg[i*3+2]*sc_nh+hdt*Tv[i*3+2]*cT;}
#ifdef _OPENACC
    #pragma acc parallel loop present(pos,vel,hi)
#elif defined(_OPENMP)
    #pragma omp parallel for
#endif
    for(int i=0;i<N;i++){
        double px=pos[i*3],py=pos[i*3+1],pz=pos[i*3+2];
        double vx=vel[i*3],vy=vel[i*3+1],vz=vel[i*3+2];
        double sx=hi[0]*px+hi[1]*py+hi[2]*pz,sy=hi[3]*px+hi[4]*py+hi[5]*pz,sz=hi[6]*px+hi[7]*py+hi[8]*pz;
        double vsx=hi[0]*vx+hi[1]*vy+hi[2]*vz,vsy=hi[3]*vx+hi[4]*vy+hi[5]*vz,vsz=hi[6]*vx+hi[7]*vy+hi[8]*vz;
        sx+=dt*vsx;sy+=dt*vsy;sz+=dt*vsz;
        sx-=floor(sx);sy-=floor(sy);sz-=floor(sz);
        pos[i*3]=sx;pos[i*3+1]=sy;pos[i*3+2]=sz;}
    for(int a=0;a<3;a++)for(int b=0;b<3;b++) h[a*3+b]+=dt*npt.Vg[a*3+b];
#ifdef _OPENACC
    #pragma acc update device(h[0:9])
#endif
#ifdef _OPENACC
    #pragma acc parallel loop present(pos,h)
#elif defined(_OPENMP)
    #pragma omp parallel for
#endif
    for(int i=0;i<N;i++){
        double sx=pos[i*3],sy=pos[i*3+1],sz=pos[i*3+2];
        pos[i*3]=h[0]*sx+h[1]*sy+h[2]*sz;
        pos[i*3+1]=h[3]*sx+h[4]*sy+h[5]*sz;
        pos[i*3+2]=h[6]*sx+h[7]*sy+h[8]*sz;}
#ifdef _OPENACC
    #pragma acc parallel loop present(qv,omg)
#elif defined(_OPENMP)
    #pragma omp parallel for
#endif
    for(int i=0;i<N;i++){
        double dq[4],tmp[4];
        omega2dq_flat(omg[i*3],omg[i*3+1],omg[i*3+2],dt,dq);
        qmul_flat(&qv[i*4],dq,tmp);
        qv[i*4]=tmp[0];qv[i*4+1]=tmp[1];qv[i*4+2]=tmp[2];qv[i*4+3]=tmp[3];
        qnorm_flat(&qv[i*4]);}
    mat_inv9(h,hi);
#ifdef _OPENACC
    #pragma acc update device(hi[0:9])
#endif
    double Ep=forces(Fv,Tv,Wm9,pos,qv,body,h,hi,nl_count,nl_list,N,natom,rmcut2,lab);
    double eps_tr2=npt.Vg[0]*hi[0]+npt.Vg[4]*hi[4]+npt.Vg[8]*hi[8];
    double sc_v2=sc_nh*exp(-hdt*eps_tr2/3.0);
#ifdef _OPENACC
    #pragma acc parallel loop present(vel,omg,Fv,Tv)
#elif defined(_OPENMP)
    #pragma omp parallel for
#endif
    for(int i=0;i<N;i++){
        vel[i*3  ]=(vel[i*3  ]+hdt*Fv[i*3  ]*cF)*sc_v2;
        vel[i*3+1]=(vel[i*3+1]+hdt*Fv[i*3+1]*cF)*sc_v2;
        vel[i*3+2]=(vel[i*3+2]+hdt*Fv[i*3+2]*cF)*sc_v2;
        omg[i*3  ]=(omg[i*3  ]+hdt*Tv[i*3  ]*cT)*sc_nh;
        omg[i*3+1]=(omg[i*3+1]+hdt*Tv[i*3+1]*cT)*sc_nh;
        omg[i*3+2]=(omg[i*3+2]+hdt*Tv[i*3+2]*cT)*sc_nh;}
    kt=ke_trans(vel,N,Mmol);kr=ke_rot(omg,N,I0);KE=kt+kr;
    npt.xi+=hdt*(2*KE-npt.Nf*kB*npt.Tt)/npt.Q;
    npt.xi=std::clamp(npt.xi,-0.1,0.1);
    double V2=fabs(mat_det9(h));
#ifdef _OPENACC
    #pragma acc update self(Wm9[0:9])
#endif
    dP=inst_P(Wm9,kt,V2)-npt.Pe;
    for(int a=0;a<3;a++) npt.Vg[a*4]+=hdt*V2*dP/(npt.W*eV2GPa);
    for(int a=0;a<3;a++) npt.Vg[a*4]=std::clamp(npt.Vg[a*4],-0.01,0.01);
    return{Ep,KE};}

/* ═══════════ OVITO output (host) ═══════════ */
static void write_ovito(FILE*fp,int istep,double dt,const double*pos,
    const double*vel,const double*qv,const double*body,const double*h,int N,int natom){
    fprintf(fp,"%d\n",N*natom);
    fprintf(fp,"Lattice=\"%.8f %.8f %.8f %.8f %.8f %.8f %.8f %.8f %.8f\" "
        "Properties=species:S:1:pos:R:3:c_mol:I:1:vx:R:1:vy:R:1:vz:R:1 Time=%.4f Step=%d pbc=\"T T T\"\n",
        h[0],h[3],h[6],h[1],h[4],h[7],h[2],h[5],h[8],istep*dt,istep);
    for(int i=0;i<N;i++){
        double R[9]; q2R_flat(&qv[i*4],R);
        for(int a=0;a<natom;a++){
            double bx=body[a*3],by=body[a*3+1],bz=body[a*3+2];
            double rx=pos[i*3]+R[0]*bx+R[1]*by+R[2]*bz;
            double ry=pos[i*3+1]+R[3]*bx+R[4]*by+R[5]*bz;
            double rz=pos[i*3+2]+R[6]*bx+R[7]*by+R[8]*bz;
            fprintf(fp,"C %14.8f %14.8f %14.8f %5d %14.8e %14.8e %14.8e\n",
                rx,ry,rz,i+1,vel[i*3],vel[i*3+1],vel[i*3+2]);}}}

/* ═══════════ CLI / filename / restart (host) ═══════════ */
std::map<std::string,std::string> parse_args(int argc,char**argv){
    std::map<std::string,std::string> o;
    for(int i=1;i<argc;i++){std::string a=argv[i];if(a.substr(0,2)!="--")continue;
        auto eq=a.find('=');std::string k=(eq!=std::string::npos)?a.substr(2,eq-2):a.substr(2);
        std::string v=(eq!=std::string::npos)?a.substr(eq+1):"";
        std::transform(k.begin(),k.end(),k.begin(),::tolower);o[k]=v;}
    return o;}
std::string get_opt(const std::map<std::string,std::string>&o,const std::string&k,const std::string&d){
    auto it=o.find(k);return it!=o.end()?it->second:d;}

static const struct{std::string key,defval;}OPT_DEFAULTS[]={
    {"fullerene","C60"},{"crystal","fcc"},{"cell","3"},{"temp","298.0"},
    {"pres","0.0"},{"step","10000"},{"dt","1.0"},{"init_scale","1.0"},{"seed","42"}};
std::string build_suffix(const std::map<std::string,std::string>&opts){
    std::string sfx;
    for(auto&d:OPT_DEFAULTS){if(opts.find(d.key)==opts.end())continue;
        std::string cv=opts.at(d.key);for(auto&c:cv)if(c==':')c='_';
        if(cv.size()>=2&&cv.substr(cv.size()-2)==".0")cv=cv.substr(0,cv.size()-2);
        sfx+="_"+d.key+"_"+cv;}
    return sfx;}
static bool file_exists(const std::string&p){struct stat st;return stat(p.c_str(),&st)==0;}
std::string unique_file(const std::string&base,const std::string&ext){
    std::string p=base+ext; if(!file_exists(p))return p;
    for(int n=1;n<=9999;n++){std::string c=base+"_"+std::to_string(n)+ext;if(!file_exists(c))return c;}
    return base+"_9999"+ext;}
std::string restart_filename(const std::string&oname,int istep,int nsteps){
    std::string b=oname; auto dt=b.rfind('.');if(dt!=std::string::npos)b=b.substr(0,dt);
    auto p=b.find("ovito_traj");if(p!=std::string::npos)b.replace(p,10,"restart");
    if(istep==nsteps)return b+".rst";
    int dg=1;for(int x=nsteps;x>=10;x/=10)dg++;
    char buf[64];snprintf(buf,64,"_%0*d",dg,istep);return b+buf+".rst";}

void write_restart_lj(const std::string&fname,int istep,
    const std::map<std::string,std::string>&opts,const std::string&st,int nc,
    double T,double Pe,int nsteps,double dt,int seed,const std::string&fspec,double init_scale,
    const double*h,const NPTState&npt,const double*pos,const double*qv,
    const double*vel,const double*omg,int N,int natom){
    FILE*f=fopen(fname.c_str(),"w");if(!f)return;
    fprintf(f,"# RESTART fuller_LJ_npt_md_serial_omp_acc\n# OPTIONS:");
    for(auto&[k,v]:opts)fprintf(f," --%s=%s",k.c_str(),v.c_str());
    fprintf(f,"\nSTEP %d\nNSTEPS %d\nDT %.15e\nTEMP %.15e\nPRES %.15e\n",istep,nsteps,dt,T,Pe);
    fprintf(f,"CRYSTAL %s\nNC %d\nFULLERENE %s\nINIT_SCALE %.15e\nSEED %d\n",st.c_str(),nc,fspec.c_str(),init_scale,seed);
    fprintf(f,"NMOL %d\nNATOM_MOL %d\n",N,natom);
    fprintf(f,"H");for(int i=0;i<9;i++)fprintf(f," %.15e",h[i]);
    fprintf(f,"\nNPT %.15e %.15e %.15e %.15e %.15e %d\n",npt.xi,npt.Q,npt.W,npt.Pe,npt.Tt,npt.Nf);
    fprintf(f,"VG");for(int i=0;i<9;i++)fprintf(f," %.15e",npt.Vg[i]);fprintf(f,"\n");
    for(int i=0;i<N;i++){
        fprintf(f,"MOL %d %.15e %.15e %.15e %.15e %.15e %.15e %.15e %.15e %.15e %.15e %.15e %.15e %.15e\n",
            i+1,pos[i*3],pos[i*3+1],pos[i*3+2],qv[i*4],qv[i*4+1],qv[i*4+2],qv[i*4+3],
            vel[i*3],vel[i*3+1],vel[i*3+2],omg[i*3],omg[i*3+1],omg[i*3+2]);}
    fprintf(f,"END\n");fclose(f);}

struct RestartDataLJ{int istep,N;double h[9];NPTState npt;
    std::vector<double>pos,qv,vel,omg;bool ok;};
RestartDataLJ read_restart_lj(const std::string&fname){
    RestartDataLJ rd;rd.ok=false;
    std::ifstream f(fname);if(!f)return rd;
    std::string line;
    for(int i=0;i<9;i++)rd.h[i]=0;
    rd.npt={};
    while(std::getline(f,line)){
        if(line.empty()||line[0]=='#')continue;
        std::istringstream ss(line);std::string tag;ss>>tag;
        if(tag=="STEP")ss>>rd.istep;
        else if(tag=="NMOL")ss>>rd.N;
        else if(tag=="H"){for(int i=0;i<9;i++)ss>>rd.h[i];}
        else if(tag=="NPT"){ss>>rd.npt.xi>>rd.npt.Q>>rd.npt.W>>rd.npt.Pe>>rd.npt.Tt>>rd.npt.Nf;}
        else if(tag=="VG"){for(int i=0;i<9;i++)ss>>rd.npt.Vg[i];}
        else if(tag=="MOL"){
            int idx;ss>>idx;double v[13];for(int i=0;i<13;i++)ss>>v[i];
            for(int i=0;i<3;i++)rd.pos.push_back(v[i]);
            for(int i=3;i<7;i++)rd.qv.push_back(v[i]);
            for(int i=7;i<10;i++)rd.vel.push_back(v[i]);
            for(int i=10;i<13;i++)rd.omg.push_back(v[i]);}
        else if(tag=="END")break;}
    rd.N=(int)rd.pos.size()/3; rd.ok=(rd.N>0);
    if(rd.ok)printf("  Restart loaded: %s (step %d, %d mols)\n",fname.c_str(),rd.istep,rd.N);
    return rd;}

/* ═══════════ MAIN ═══════════ */
int main(int argc,char**argv){
    auto opts=parse_args(argc,argv);
    if(opts.count("help")){
        printf(
            "fuller_LJ_npt_md_serial_omp_acc — LJ rigid-body fullerene NPT-MD\n\n"
            "Options:\n"
            "  --help                  Show this help\n"
            "  --fullerene=<name>      Fullerene species (default: C60)\n"
            "  --crystal=<fcc|hcp|bcc> Crystal structure (default: fcc)\n"
            "  --cell=<nc>             Unit cell repeats (default: 3)\n"
            "  --temp=<K>              Target temperature [K] (default: 298.0)\n"
            "  --pres=<GPa>            Target pressure [GPa] (default: 0.0)\n"
            "  --step=<N>              Production steps (default: 10000)\n"
            "  --dt=<fs>               Time step [fs] (default: 1.0)\n"
            "  --init_scale=<s>        Lattice scale factor (default: 1.0)\n"
            "  --seed=<n>              Random seed (default: 42)\n"
            "  --coldstart=<N>         Cold-start steps at 4K (default: 0)\n"
            "  --warmup=<N>            Warm-up ramp steps 4K->T (default: 0)\n"
            "  --from=<step>           Averaging start step (default: auto)\n"
            "  --to=<step>             Averaging end step (default: nsteps)\n"
            "  --mon=<N>               Monitor print interval (default: auto)\n"
            "  --warmup_mon=<mode>     Warmup monitor: norm|freq|some (default: norm)\n"
            "  --ovito=<N>             OVITO XYZ output interval, 0=off (default: 0)\n"
            "  --ofile=<filename>      OVITO output filename (default: auto)\n"
            "  --restart=<N>           Restart save interval, 0=off (default: 0)\n"
            "  --resfile=<path>        Resume from restart file\n"
            "  --libdir=<path>         Fullerene library dir (default: FullereneLib)\n\n"
            "Examples:\n"
            "  ./prog --temp=500 --pres=1.0 --step=50000\n"
            "  ./prog --coldstart=2000 --warmup=3000 --step=20000\n"
            "  ./prog --step=10000 --ovito=100\n"
            "  ./prog --step=50000 --restart=5000\n"
            "  ./prog --resfile=restart_LJ_serial_00010000.rst\n");
        return 0;}
    std::string crystal=get_opt(opts,"crystal","fcc");
    std::string st=crystal;std::transform(st.begin(),st.end(),st.begin(),::toupper);
    int nc=std::atoi(get_opt(opts,"cell","3").c_str());
    double T=std::atof(get_opt(opts,"temp","298.0").c_str());
    double Pe=std::atof(get_opt(opts,"pres","0.0").c_str());
    int nsteps=std::atoi(get_opt(opts,"step","10000").c_str());
    double dt=std::atof(get_opt(opts,"dt","1.0").c_str());
    int seed=std::atoi(get_opt(opts,"seed","42").c_str());
    double init_scale=std::atof(get_opt(opts,"init_scale","1.0").c_str());
    std::string fspec=get_opt(opts,"fullerene","C60");
    std::string libdir=get_opt(opts,"libdir","FullereneLib");
    int coldstart=std::atoi(get_opt(opts,"coldstart","0").c_str());
    int warmup=std::atoi(get_opt(opts,"warmup","0").c_str());
    int avg_from=std::atoi(get_opt(opts,"from","0").c_str());
    int avg_to=std::atoi(get_opt(opts,"to","0").c_str());
    int nrec_o=std::atoi(get_opt(opts,"ovito","0").c_str());
    int nrec_rst=std::atoi(get_opt(opts,"restart","0").c_str());
    std::string resfile=get_opt(opts,"resfile","");
    int start_step=0;
    int mon_interval=std::atoi(get_opt(opts,"mon","0").c_str());
    std::string warmup_mon_mode=get_opt(opts,"warmup_mon","norm");
    constexpr double T_cold=4.0;

    if(avg_to<=0) avg_to=nsteps;
    if(avg_from<=0) avg_from=std::max(1,nsteps-nsteps/4);
    int total_steps=coldstart+warmup+nsteps;
    int gavg_from=coldstart+warmup+avg_from,gavg_to=coldstart+warmup+avg_to;
    if(avg_from<1||avg_from>=avg_to||avg_to>nsteps){
        fprintf(stderr,"Error: invalid --from/--to range\n");return 1;}

    std::string sfx=build_suffix(opts);

#if defined(_OPENACC)
    std::string mode_tag="gpu";
#elif defined(_OPENMP)
    std::string mode_tag="omp";
#else
    std::string mode_tag="serial";
#endif
    std::string ovito_file=opts.count("ofile")?get_opt(opts,"ofile",""):unique_file("ovito_traj_LJ_"+mode_tag+sfx,".xyz");

    auto[fpath,label]=resolve_fullerene(fspec,libdir);
    MolData mol=load_cc1(fpath);
    int natom=mol.natom;
    double RMCUT=RCUT+2*mol.Rmol+1.0,RMCUT2=RMCUT*RMCUT;
    double a0=default_a0(mol.Dmol,st,init_scale);

    /* Compute Nmax for crystal type */
    int Nmax;
    if(st=="FCC") Nmax=4*nc*nc*nc;
    else if(st=="HCP") Nmax=2*nc*nc*nc;
    else Nmax=2*nc*nc*nc;

    /* GPU memory check */
#ifdef _OPENACC
    constexpr long BPM=(long)(3*5+4)*8+(long)MAX_NATOM*3*8+4+MAX_NEIGH*4;
    long mem_req=(long)Nmax*BPM+(long)natom*3*8+9*8*3;
    long gpu_mem=0; int nc_max_gpu=9999;
    int ndev=acc_get_num_devices(acc_device_nvidia);
    if(ndev>0){acc_set_device_num(0,acc_device_nvidia);
        gpu_mem=(long)acc_get_property(0,acc_device_nvidia,acc_property_memory);
        if(gpu_mem>0){long usable=(long)(gpu_mem*0.80);
            long max_n=(usable-(long)natom*3*8-9*8*3)/BPM;
            nc_max_gpu=(int)cbrt((double)max_n/(st=="FCC"?4.0:2.0));}}
    if(nc>nc_max_gpu){
        printf("Error: nc=%d exceeds GPU memory (max nc=%d for %.1f GB)\n",
            nc,nc_max_gpu,gpu_mem/1024.0/1024.0/1024.0);return 1;}
#endif

    /* Allocate flat arrays */
    double*pos=new double[Nmax*3]();
    double*vel=new double[Nmax*3]();
    double*omg=new double[Nmax*3]();
    double*qv =new double[Nmax*4]();
    double*Fv =new double[Nmax*3]();
    double*Tv =new double[Nmax*3]();
    double*lab=new double[Nmax*natom*3]();
    double*body=new double[natom*3];
    double h[9]={},hi[9]={},Wm9[9]={};
    int*nl_count=new int[Nmax]();
    int*nl_list =new int[Nmax*MAX_NEIGH]();

    for(int i=0;i<natom*3;i++) body[i]=mol.coords[i];

    /* Build crystal */
    int N;
    if(st=="FCC") N=make_fcc(a0,nc,pos,h);
    else if(st=="HCP") N=make_hcp(a0,nc,pos,h);
    else N=make_bcc(a0,nc,pos,h);
    mat_inv9(h,hi);

    printf("========================================================================\n");
#if defined(_OPENACC)
    printf("  Fullerene Crystal NPT-MD — LJ rigid-body (OpenACC GPU)\n");
#elif defined(_OPENMP)
    printf("  Fullerene Crystal NPT-MD — LJ rigid-body (OpenMP)\n");
#else
    printf("  Fullerene Crystal NPT-MD — LJ rigid-body (Serial)\n");
#endif
    printf("========================================================================\n");
#ifdef _OPENACC
    if(ndev>0){printf("  GPU device      : 0 (NVIDIA)\n");
        if(gpu_mem>0)printf("  GPU VRAM        : %.2f GB (usable 80%%: %.2f GB, max nc=%d)\n",
            gpu_mem/1024.0/1024.0/1024.0,gpu_mem*0.80/1024.0/1024.0/1024.0,nc_max_gpu);}
#endif
    printf("  Fullerene       : %s (%d atoms/mol)\n",label.c_str(),natom);
    printf("  Crystal         : %s %dx%dx%d  Nmol=%d\n",st.c_str(),nc,nc,nc,N);
    printf("  a0=%.3f A  T=%.1f K  P=%.4f GPa  dt=%.2f fs\n",a0,T,Pe,dt);
#ifdef _OPENACC
    printf("  GPU memory req  : %.2f MB\n",mem_req/1024.0/1024.0);
#endif
    if(coldstart>0) printf("  Coldstart       : %d steps at %.1f K\n",coldstart,T_cold);
    if(warmup>0)    printf("  Warmup          : %d steps (%.1fK->%.1fK)\n",warmup,T_cold,T);
    printf("  Production      : %d steps  avg=%d-%d\n",nsteps,avg_from,avg_to);
    printf("  Total           : %d steps\n",total_steps);
    printf("========================================================================\n\n");

    /* Initial velocities */
    double T_init=(coldstart>0||warmup>0)?T_cold:T;
    std::mt19937 rng(seed);std::normal_distribution<double> gauss(0,1);
    double sv=sqrt(kB*T_init*CONV/mol.Mmol),sw=sqrt(kB*T_init*CONV/mol.I0);
    for(int i=0;i<N;i++){
        for(int a=0;a<3;a++){vel[i*3+a]=sv*gauss(rng);omg[i*3+a]=sw*gauss(rng);}
        for(int a=0;a<4;a++)qv[i*4+a]=gauss(rng);
        double n=sqrt(qv[i*4]*qv[i*4]+qv[i*4+1]*qv[i*4+1]+qv[i*4+2]*qv[i*4+2]+qv[i*4+3]*qv[i*4+3]);
        for(int a=0;a<4;a++)qv[i*4+a]/=n;}
    double vcm[3]={0,0,0};
    for(int i=0;i<N;i++)for(int a=0;a<3;a++)vcm[a]+=vel[i*3+a];
    for(int a=0;a<3;a++)vcm[a]/=N;
    for(int i=0;i<N;i++)for(int a=0;a<3;a++)vel[i*3+a]-=vcm[a];

    NPTState npt=make_npt(T,Pe,N); npt.Tt=T_init;

    if(!resfile.empty()){
        auto rd=read_restart_lj(resfile);
        if(rd.ok){start_step=rd.istep;for(int i=0;i<9;i++)h[i]=rd.h[i];npt=rd.npt;
            int nn=std::min(N,rd.N);
            for(int i=0;i<nn*3;i++){pos[i]=rd.pos[i];vel[i]=rd.vel[i];omg[i]=rd.omg[i];}
            for(int i=0;i<nn*4;i++)qv[i]=rd.qv[i];
            printf("  Restarting from global step %d\n",start_step);}}

    mat_inv9(h,hi);
    nlist_build_sym(pos,h,hi,N,RMCUT,nl_count,nl_list);

    /* ═══ OpenACC data region ═══ */
#ifdef _OPENACC
    #pragma acc data \
        copy(pos[0:N*3],vel[0:N*3],omg[0:N*3],qv[0:N*4]) \
        copy(Fv[0:N*3],Tv[0:N*3]) \
        copyin(body[0:natom*3]) \
        copy(h[0:9],hi[0:9],Wm9[0:9]) \
        create(lab[0:N*natom*3]) \
        copyin(nl_count[0:N],nl_list[0:N*MAX_NEIGH])
    {
#endif
        apply_pbc(pos,h,hi,N);
        forces(Fv,Tv,Wm9,pos,qv,body,h,hi,nl_count,nl_list,N,natom,RMCUT2,lab);

        int prn=(mon_interval>0)?mon_interval:std::max(1,total_steps/50);
        int prn_pre=prn;
        if(coldstart+warmup>0){int div=100;
            if(warmup_mon_mode=="freq")div=10;else if(warmup_mon_mode=="some")div=1000;
            prn_pre=std::max(1,(coldstart+warmup)/div);}
        int nlup=25;
        double sT=0,sP=0,sa=0,sEp=0;int nav=0;
        auto t0=std::chrono::steady_clock::now();

        /* OVITO file must be opened outside acc data, but output needs host data */
        FILE*io_o=nullptr;
        if(nrec_o>0) io_o=fopen(ovito_file.c_str(),"w");

        printf("  %8s %5s %7s %9s %8s %10s %13s %7s\n",
            "step","phase","T[K]","P[GPa]","a[A]","Ecoh[eV]","Ecoh[kcal/m]","t[s]");

        bool stop_requested=false;
        for(int gstep=start_step+1;gstep<=total_steps;gstep++){
            const char*phase=gstep<=coldstart?"COLD":gstep<=coldstart+warmup?"WARM":"PROD";
            int cur_prn=(gstep<=coldstart+warmup)?prn_pre:prn;

            if(gstep<=coldstart) npt.Tt=T_cold;
            else if(gstep<=coldstart+warmup)
                npt.Tt=T_cold+(T-T_cold)*double(gstep-coldstart)/double(warmup);
            else npt.Tt=T;

            if(coldstart>0&&gstep==coldstart+1){npt.xi=0;for(int i=0;i<9;i++)npt.Vg[i]=0;}
            if(gstep<=coldstart){for(int i=0;i<9;i++)npt.Vg[i]=0;}

            if(gstep%nlup==0){
#ifdef _OPENACC
                #pragma acc update self(pos[0:N*3])
#endif
                mat_inv9(h,hi);
                nlist_build_sym(pos,h,hi,N,RMCUT,nl_count,nl_list);
#ifdef _OPENACC
                #pragma acc update device(nl_count[0:N],nl_list[0:N*MAX_NEIGH])
                #pragma acc update device(hi[0:9])
#endif
            }

            auto[Ep,KE]=step_npt(pos,vel,qv,omg,Fv,Tv,Wm9,h,hi,body,
                mol.I0,mol.Mmol,N,natom,RMCUT2,dt,npt,nl_count,nl_list,lab);

            double kt=ke_trans(vel,N,mol.Mmol);
            double V=fabs(mat_det9(h));
            double Tn=inst_T(KE,npt.Nf),Pn=inst_P(Wm9,kt,V);

            /* COLD/WARM velocity rescaling */
            if((gstep<=coldstart||gstep<=coldstart+warmup) && Tn>0.1){
                double tgt=(gstep<=coldstart)?T_cold:npt.Tt;
                double scale=sqrt(std::max(tgt,0.1)/Tn);
#ifdef _OPENACC
                #pragma acc parallel loop present(vel,omg)
#elif defined(_OPENMP)
                #pragma omp parallel for
#endif
                for(int i=0;i<N;i++){
                    vel[i*3]*=scale;vel[i*3+1]*=scale;vel[i*3+2]*=scale;
                    omg[i*3]*=scale;omg[i*3+1]*=scale;omg[i*3+2]*=scale;}
                kt=ke_trans(vel,N,mol.Mmol);KE=kt+ke_rot(omg,N,mol.I0);
                Tn=inst_T(KE,npt.Nf);
                npt.xi=0;
                if(gstep<=coldstart){for(int i=0;i<9;i++)npt.Vg[i]=0;}
            }

            double Ec=Ep/N,an=h[0]/nc;
            if(gstep>=gavg_from&&gstep<=gavg_to){sT+=Tn;sP+=Pn;sa+=an;sEp+=Ec;nav++;}

            if(io_o&&gstep%nrec_o==0){
#ifdef _OPENACC
                #pragma acc update self(pos[0:N*3],vel[0:N*3],qv[0:N*4])
#endif
                write_ovito(io_o,gstep,dt,pos,vel,qv,body,h,N,natom);fflush(io_o);}

            if(nrec_rst>0&&(gstep%nrec_rst==0||gstep==total_steps)){
#ifdef _OPENACC
                #pragma acc update self(pos[0:N*3],vel[0:N*3],qv[0:N*4],omg[0:N*3])
#endif
                auto rfn=restart_filename(ovito_file,gstep,total_steps);
                write_restart_lj(rfn,gstep,opts,st,nc,T,Pe,nsteps,dt,seed,fspec,init_scale,
                    h,npt,pos,qv,vel,omg,N,natom);
                if(stop_requested){
                    printf("\n  *** Stopped at restart checkpoint (step %d) ***\n",gstep);
                    break;}}

            if(gstep%cur_prn==0||gstep==total_steps){
                if(dir_exists("abort.md")){
                    printf("\n  *** abort.md detected at step %d ***\n",gstep);
                    if(nrec_rst>0){
#ifdef _OPENACC
                        #pragma acc update self(pos[0:N*3],vel[0:N*3],qv[0:N*4],omg[0:N*3])
#endif
                        auto rfn=restart_filename(ovito_file,gstep,total_steps);
                        write_restart_lj(rfn,gstep,opts,st,nc,T,Pe,nsteps,dt,seed,fspec,init_scale,
                            h,npt,pos,qv,vel,omg,N,natom);
                    }
                    break;
                }
                if(!stop_requested && dir_exists("stop.md")){
                    stop_requested=true;
                    printf("\n  *** stop.md detected at step %d — will stop at next checkpoint ***\n",gstep);
                }
                double el=std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count();
                printf("  %8d %5s %7.1f %9.3f %8.3f %10.5f %13.4f %7.0f\n",
                    gstep,phase,Tn,Pn,an,Ec,Ec*eV2kcalmol,el);}
        }
        if(io_o)fclose(io_o);

        if(nav>0){
            printf("\n========================================================================\n");
            printf("  Averages (%d samples): T=%.2f K  P=%.4f GPa  a=%.4f A  Ecoh=%.5f eV\n",
                nav,sT/nav,sP/nav,sa/nav,sEp/nav);
            printf("========================================================================\n");}
        printf("  Done (%.1f sec)\n",std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count());
#ifdef _OPENACC
    } /* end acc data */
#endif

    delete[]pos;delete[]vel;delete[]omg;delete[]qv;
    delete[]Fv;delete[]Tv;delete[]lab;delete[]body;
    delete[]nl_count;delete[]nl_list;
    return 0;
}
