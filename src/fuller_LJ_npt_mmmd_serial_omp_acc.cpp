// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2025, Takeshi Nishikawa
/*===========================================================================
  fuller_LJ_npt_mmmd_serial_omp_acc.cpp — Fullerene Crystal NPT-MD
  (Molecular Mechanics Force Field + LJ Intermolecular, Serial / OpenMP / OpenACC GPU)

  V_total = V_bond + V_angle + V_dihedral + V_improper + V_LJ + V_Coulomb

  Compile:
    Serial:  g++ -std=c++17 -O3 -Wno-unknown-pragmas \
               -o fuller_LJ_npt_mmmd_serial fuller_LJ_npt_mmmd_serial_omp_acc.cpp -lm
    OpenMP:  g++ -std=c++17 -O3 -fopenmp -Wno-unknown-pragmas \
               -o fuller_LJ_npt_mmmd_omp fuller_LJ_npt_mmmd_serial_omp_acc.cpp -lm
    OpenACC: nvc++ -std=c++17 -O3 -acc -gpu=cc80 -Minfo=accel \
               -o fuller_LJ_npt_mmmd_gpu fuller_LJ_npt_mmmd_serial_omp_acc.cpp -lm

  Runtime Options (all in --key=value format):
    --help                  Show help
    --fullerene=<name>      Fullerene species (default: C60)
    --crystal=<fcc|hcp|bcc> Crystal structure (default: fcc)
    --cell=<nc>             Unit cell repeat count (default: 3)
    --temp=<T_K>            Target temperature [K] (default: 298.0)
    --pres=<P_GPa>          Target pressure [GPa] (default: 0.0)
    --step=<N>              Number of production steps (default: 10000)
    --dt=<fs>               Time step [fs] (default: 0.1)
    --init_scale=<s>        Lattice constant scale factor (default: 1.0)
    --seed=<n>              Random seed (default: 42)
    --coldstart=<N>         Cold start (4K) steps (default: 0)
    --warmup=<N>            Warmup steps 4K->T (default: 0)
    --from=<step>           Averaging start step (default: 3/4 of production)
    --to=<step>             Averaging end step (default: nsteps)
    --mon=<N>               Monitoring output interval (default: auto)
    --warmup_mon=<mode>     Output frequency during warmup norm|freq|some (default: norm)
    --ovito=<N>             OVITO XYZ output interval (0=disabled, default: 0)
    --restart=<N>           Restart save interval (0=disabled, default: 0)
    --resfile=<path>        Resume from restart file
    --libdir=<path>         Fullerene library (default: FullereneLib)
    --ff_kb=<kcal/mol>      Bond stretching force constant (default: 469.0)
    --ff_kth=<kcal/mol>     Angle bending force constant (default: 63.0)
    --ff_v2=<kcal/mol>      Dihedral torsion force constant (default: 14.5)
    --ff_kimp=<kcal/mol>    Improper dihedral force constant (default: 15.0)

  Execution Examples:
    # Basic run (C60 FCC 3x3x3, 298K, 10000 steps)
    ./fuller_LJ_npt_mmmd_serial

    # Long run with specified temperature and pressure
    ./fuller_LJ_npt_mmmd_omp --temp=500 --pres=1.0 --step=100000

    # Cold start + Warmup + Production
    ./fuller_LJ_npt_mmmd_serial --coldstart=5000 --warmup=5000 --step=50000

    # OVITO output (write XYZ file every 200 steps)
    ./fuller_LJ_npt_mmmd_omp --step=20000 --ovito=200

    # Restart save (every 10000 steps + final step)
    ./fuller_LJ_npt_mmmd_serial --step=100000 --restart=10000

    # Resume from restart file
    ./fuller_LJ_npt_mmmd_serial --resfile=restart_mmmd_serial_00050000.rst

    # Customize force field parameters
    ./fuller_LJ_npt_mmmd_omp --ff_kb=500 --ff_kth=70 --step=20000

    # Use OVITO + Restart simultaneously
    ./fuller_LJ_npt_mmmd_gpu --step=200000 --ovito=1000 --restart=20000

  Stop Control:
    Create the following in the current directory during execution to control behavior:
    - abort.md: Stop immediately (saves restart if enabled before exiting)
    - stop.md:  Stop at next restart checkpoint

  Unit System: A, amu, eV, fs, K, GPa
===========================================================================*/
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <string>
#include <vector>
#include <array>
#include <set>
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

constexpr double CONV=9.64853321e-3, kB=8.617333262e-5;
constexpr double eV2GPa=160.21766208, eV2kcalmol=23.06054783;
constexpr double kcal2eV=1.0/eV2kcalmol, PI=3.14159265358979323846;
constexpr double mC=12.011;
constexpr double sigma_LJ=3.431, eps_LJ=2.635e-3;
constexpr double RCUT=3.0*sigma_LJ, RCUT2=RCUT*RCUT, sig2_LJ=sigma_LJ*sigma_LJ;
constexpr double _sr_v=1.0/3.0,_sr2_v=_sr_v*_sr_v,_sr6_v=_sr2_v*_sr2_v*_sr2_v;
constexpr double VSHFT=4.0*eps_LJ*(_sr6_v*_sr6_v-_sr6_v);
constexpr double COULOMB_K=14.3996;
constexpr double COUL_RCUT=RCUT, COUL_RCUT2=COUL_RCUT*COUL_RCUT;
constexpr int MAX_LJ_NEIGH=400;

/* ═══════════ Flat H-matrix ═══════════ */
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
    #pragma omp parallel for schedule(static)
#endif
    for(int i=0;i<N;i++){
        double px=pos[i*3],py=pos[i*3+1],pz=pos[i*3+2];
        double s0=hi[0]*px+hi[1]*py+hi[2]*pz,s1=hi[3]*px+hi[4]*py+hi[5]*pz,s2=hi[6]*px+hi[7]*py+hi[8]*pz;
        s0-=floor(s0);s1-=floor(s1);s2-=floor(s2);
        pos[i*3]=h[0]*s0+h[1]*s1+h[2]*s2;
        pos[i*3+1]=h[3]*s0+h[4]*s1+h[5]*s2;
        pos[i*3+2]=h[6]*s0+h[7]*s1+h[8]*s2;}}

/* ═══════════ cc1 reader (host) ═══════════ */
struct MolData{
    std::vector<std::array<double,3>> coords;
    std::vector<std::pair<int,int>> bonds;
    int natom; double Rmol,Dmol;};
MolData load_cc1(const std::string&path){
    MolData md;std::ifstream f(path);
    if(!f){fprintf(stderr,"Error: cannot open %s\n",path.c_str());exit(1);}
    f>>md.natom;md.coords.resize(md.natom);
    std::vector<std::vector<int>> adj(md.natom);
    for(int i=0;i<md.natom;i++){
        std::string line;std::getline(f,line);while(line.empty()&&f.good())std::getline(f,line);
        std::istringstream ss(line);std::string el;int idx;double x,y,z;int fl;
        ss>>el>>idx>>x>>y>>z>>fl;md.coords[i]={x,y,z};
        int b;while(ss>>b)adj[i].push_back(b-1);}
    Vec3 cm={0,0,0};for(auto&c:md.coords)for(int a=0;a<3;a++)cm[a]+=c[a];
    for(int a=0;a<3;a++)cm[a]/=md.natom;
    for(auto&c:md.coords)for(int a=0;a<3;a++)c[a]-=cm[a];
    md.Rmol=0;md.Dmol=0;
    for(int i=0;i<md.natom;i++){
        double r2=0;for(int a=0;a<3;a++)r2+=md.coords[i][a]*md.coords[i][a];
        double r=sqrt(r2);if(r>md.Rmol)md.Rmol=r;
        for(int j=i+1;j<md.natom;j++){
            double d2=0;for(int a=0;a<3;a++){double d=md.coords[i][a]-md.coords[j][a];d2+=d*d;}
            double d=sqrt(d2);if(d>md.Dmol)md.Dmol=d;}}
    std::set<std::pair<int,int>> bset;
    for(int i=0;i<md.natom;i++)for(int j:adj[i])bset.insert({std::min(i,j),std::max(i,j)});
    md.bonds.assign(bset.begin(),bset.end());
    return md;}

std::pair<std::string,std::string> resolve_fullerene(const std::string&spec,const std::string&lib="FullereneLib"){
    std::string sl=spec;std::transform(sl.begin(),sl.end(),sl.begin(),::tolower);
    if(sl=="buckyball"||sl=="c60"||sl=="c60:ih")return{lib+"/C60-76/C60-Ih.cc1","C60(Ih)"};
    if(sl=="c70"||sl=="c70:d5h")return{lib+"/C60-76/C70-D5h.cc1","C70(D5h)"};
    if(sl=="c72"||sl=="c72:d6d")return{lib+"/C60-76/C72-D6d.cc1","C72(D6d)"};
    if(sl=="c74"||sl=="c74:d3h")return{lib+"/C60-76/C74-D3h.cc1","C74(D3h)"};
    if(sl.substr(0,4)=="c76:"&&sl.size()>4){auto sym=spec.substr(4);return{lib+"/C60-76/C76-"+sym+".cc1","C76("+sym+")"};}
    if(sl.substr(0,4)=="c84:"){auto rest=spec.substr(4);auto c=rest.find(':');
        if(c!=std::string::npos){int n=std::atoi(rest.substr(0,c).c_str());auto sym=rest.substr(c+1);
            char buf[64];snprintf(buf,64,"C84-No.%02d-%s.cc1",n,sym.c_str());return{lib+"/C84/"+buf,"C84 No."+std::to_string(n)};}
        else{int n=std::atoi(rest.c_str());char pfx[32];snprintf(pfx,32,"C84-No.%02d-",n);
            std::string dpath=lib+"/C84";DIR*dp=opendir(dpath.c_str());
            if(dp){struct dirent*ep;while((ep=readdir(dp))){std::string fn=ep->d_name;
                if(fn.find(pfx)==0&&fn.size()>4&&fn.substr(fn.size()-4)==".cc1"){closedir(dp);return{dpath+"/"+fn,"C84 No."+std::to_string(n)};}}
                closedir(dp);}}}
    fprintf(stderr,"Unknown: %s\n",spec.c_str());exit(1);}

/* ═══════════ Crystal (host, flat) ═══════════ */
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
    double m=1.4,a0;if(st=="FCC")a0=dmax*sqrt(2.0)*m;else if(st=="HCP")a0=dmax*m;else a0=dmax*2.0/sqrt(3.0)*m;return a0*s;}

/* ═══════════ Topology generation (host) ═══════════ */
/* Flat GPU-friendly topology: separate arrays for indices and parameters */
struct FlatTopology {
    /* Bonds: Nb entries */
    int Nb; int*b_i0; int*b_i1; double*b_kb; double*b_r0;
    /* Angles: Nang entries */
    int Nang; int*ang_i0; int*ang_i1; int*ang_i2; double*ang_kth; double*ang_th0;
    /* Dihedrals: Ndih entries */
    int Ndih; int*dih_i0; int*dih_i1; int*dih_i2; int*dih_i3; double*dih_Vn; int*dih_mult; double*dih_gamma;
    /* Impropers: Nimp entries */
    int Nimp; int*imp_i0; int*imp_i1; int*imp_i2; int*imp_i3; double*imp_ki; double*imp_gamma;
    /* Molecule IDs */
    int*mol_id;
};

/* ═══════════ Compute dihedral angle from 4 flat positions ═══════════ */
static double compute_phi0(const double*c,int i,int j,int k,int l){
    double b1x=c[j*3]-c[i*3],b1y=c[j*3+1]-c[i*3+1],b1z=c[j*3+2]-c[i*3+2];
    double b2x=c[k*3]-c[j*3],b2y=c[k*3+1]-c[j*3+1],b2z=c[k*3+2]-c[j*3+2];
    double b3x=c[l*3]-c[k*3],b3y=c[l*3+1]-c[k*3+1],b3z=c[l*3+2]-c[k*3+2];
    double mx=b1y*b2z-b1z*b2y,my=b1z*b2x-b1x*b2z,mz=b1x*b2y-b1y*b2x;
    double nx=b2y*b3z-b2z*b3y,ny=b2z*b3x-b2x*b3z,nz=b2x*b3y-b2y*b3x;
    double mm=mx*mx+my*my+mz*mz,nn=nx*nx+ny*ny+nz*nz;
    if(mm<1e-20||nn<1e-20)return 0;
    double b2len=sqrt(b2x*b2x+b2y*b2y+b2z*b2z);
    double cosphi=(mx*nx+my*ny+mz*nz)/sqrt(mm*nn);
    if(cosphi>1.0)cosphi=1.0;if(cosphi<-1.0)cosphi=-1.0;
    double sinphi=(mx*b3x+my*b3y+mz*b3z)*b2len/sqrt(mm*nn);
    return atan2(sinphi,cosphi);}

FlatTopology build_flat_topology(const MolData&mol,int Nmol,double kb,double kth,double v2_dih,double k_imp){
    int na=mol.natom;
    std::vector<std::vector<int>> adj(na);
    for(auto&[bi,bj]:mol.bonds){adj[bi].push_back(bj);adj[bj].push_back(bi);}

    /* Flat coords for phi0 computation */
    double*cc=new double[na*3];
    for(int i=0;i<na;i++){cc[i*3]=mol.coords[i][0];cc[i*3+1]=mol.coords[i][1];cc[i*3+2]=mol.coords[i][2];}

    /* Count per-molecule topology */
    struct{int a0,a1;}bnd_tmp[4096]; int nb_mol=0;
    double bnd_r0[4096];
    for(auto&[bi,bj]:mol.bonds){
        double dx=mol.coords[bi][0]-mol.coords[bj][0],dy=mol.coords[bi][1]-mol.coords[bj][1],dz=mol.coords[bi][2]-mol.coords[bj][2];
        bnd_tmp[nb_mol]={bi,bj}; bnd_r0[nb_mol]=sqrt(dx*dx+dy*dy+dz*dz); nb_mol++;}

    struct{int a0,a1,a2;}ang_tmp[8192]; double ang_th0_tmp[8192]; int nang_mol=0;
    for(int j=0;j<na;j++){
        auto&nj=adj[j];
        for(int ii=0;ii<(int)nj.size();ii++)for(int kk=ii+1;kk<(int)nj.size();kk++){
            int i=nj[ii],k=nj[kk];
            double rji[3],rjk[3];
            for(int a=0;a<3;a++){rji[a]=mol.coords[i][a]-mol.coords[j][a];rjk[a]=mol.coords[k][a]-mol.coords[j][a];}
            double dji=sqrt(rji[0]*rji[0]+rji[1]*rji[1]+rji[2]*rji[2]);
            double djk=sqrt(rjk[0]*rjk[0]+rjk[1]*rjk[1]+rjk[2]*rjk[2]);
            double costh=(rji[0]*rjk[0]+rji[1]*rjk[1]+rji[2]*rjk[2])/(dji*djk);
            costh=std::clamp(costh,-1.0,1.0);
            ang_tmp[nang_mol]={i,j,k}; ang_th0_tmp[nang_mol]=acos(costh); nang_mol++;}}

    struct{int a0,a1,a2,a3;}dih_tmp[16384]; double dih_gamma_tmp[16384]; int ndih_mol=0;
    for(auto&[bj,bk]:mol.bonds){
        for(int i:adj[bj]){if(i==bk)continue;
            for(int l:adj[bk]){if(l==bj||l==i)continue;
                double phi0=compute_phi0(cc,i,bj,bk,l);
                dih_tmp[ndih_mol]={i,bj,bk,l};
                dih_gamma_tmp[ndih_mol]=2*phi0+PI;
                ndih_mol++;}}}

    struct{int a0,a1,a2,a3;}imp_tmp[4096]; double imp_gamma_tmp[4096]; int nimp_mol=0;
    for(int i=0;i<na;i++){
        if((int)adj[i].size()==3){
            int j=adj[i][0],k=adj[i][1],l=adj[i][2];
            double psi0=compute_phi0(cc,j,i,k,l);
            imp_tmp[nimp_mol]={i,j,k,l};
            imp_gamma_tmp[nimp_mol]=2*psi0+PI;
            nimp_mol++;}}
    delete[]cc;

    /* Allocate global arrays */
    FlatTopology ft;
    ft.Nb=nb_mol*Nmol; ft.Nang=nang_mol*Nmol; ft.Ndih=ndih_mol*Nmol; ft.Nimp=nimp_mol*Nmol;
    ft.b_i0=new int[ft.Nb]; ft.b_i1=new int[ft.Nb]; ft.b_kb=new double[ft.Nb]; ft.b_r0=new double[ft.Nb];
    ft.ang_i0=new int[ft.Nang]; ft.ang_i1=new int[ft.Nang]; ft.ang_i2=new int[ft.Nang];
    ft.ang_kth=new double[ft.Nang]; ft.ang_th0=new double[ft.Nang];
    ft.dih_i0=new int[ft.Ndih]; ft.dih_i1=new int[ft.Ndih]; ft.dih_i2=new int[ft.Ndih]; ft.dih_i3=new int[ft.Ndih];
    ft.dih_Vn=new double[ft.Ndih]; ft.dih_mult=new int[ft.Ndih]; ft.dih_gamma=new double[ft.Ndih];
    ft.imp_i0=new int[ft.Nimp]; ft.imp_i1=new int[ft.Nimp]; ft.imp_i2=new int[ft.Nimp]; ft.imp_i3=new int[ft.Nimp];
    ft.imp_ki=new double[ft.Nimp]; ft.imp_gamma=new double[ft.Nimp];
    ft.mol_id=new int[Nmol*na];

    /* Replicate */
    for(int m=0;m<Nmol;m++){
        int off=m*na, ob=m*nb_mol, oa=m*nang_mol, od=m*ndih_mol, oi=m*nimp_mol;
        for(int a=0;a<na;a++) ft.mol_id[off+a]=m;
        for(int b=0;b<nb_mol;b++){
            ft.b_i0[ob+b]=bnd_tmp[b].a0+off; ft.b_i1[ob+b]=bnd_tmp[b].a1+off;
            ft.b_kb[ob+b]=kb; ft.b_r0[ob+b]=bnd_r0[b];}
        for(int a=0;a<nang_mol;a++){
            ft.ang_i0[oa+a]=ang_tmp[a].a0+off; ft.ang_i1[oa+a]=ang_tmp[a].a1+off; ft.ang_i2[oa+a]=ang_tmp[a].a2+off;
            ft.ang_kth[oa+a]=kth; ft.ang_th0[oa+a]=ang_th0_tmp[a];}
        for(int d=0;d<ndih_mol;d++){
            ft.dih_i0[od+d]=dih_tmp[d].a0+off; ft.dih_i1[od+d]=dih_tmp[d].a1+off;
            ft.dih_i2[od+d]=dih_tmp[d].a2+off; ft.dih_i3[od+d]=dih_tmp[d].a3+off;
            ft.dih_Vn[od+d]=v2_dih; ft.dih_mult[od+d]=2; ft.dih_gamma[od+d]=dih_gamma_tmp[d];}
        for(int p=0;p<nimp_mol;p++){
            ft.imp_i0[oi+p]=imp_tmp[p].a0+off; ft.imp_i1[oi+p]=imp_tmp[p].a1+off;
            ft.imp_i2[oi+p]=imp_tmp[p].a2+off; ft.imp_i3[oi+p]=imp_tmp[p].a3+off;
            ft.imp_ki[oi+p]=k_imp; ft.imp_gamma[oi+p]=imp_gamma_tmp[p];}}
    printf("  Topology/mol: %d bonds, %d angles, %d dihedrals, %d impropers\n",nb_mol,nang_mol,ndih_mol,nimp_mol);
    printf("  Total:        %d bonds, %d angles, %d dihedrals, %d impropers\n",ft.Nb,ft.Nang,ft.Ndih,ft.Nimp);
    return ft;}

/* ═══════════ LJ neighbor list (host, half-list) ═══════════ */
static void build_nlist_lj(const double*pos,const double*h,const double*hi,
    int Na,const int*mol_id,int*nlc,int*nll){
    double rc2=(RCUT+2.0)*(RCUT+2.0);
    for(int i=0;i<Na;i++)nlc[i]=0;
    for(int i=0;i<Na-1;i++){int mi=mol_id[i];
        for(int j=i+1;j<Na;j++){if(mol_id[j]==mi)continue;
            double dx=pos[j*3]-pos[i*3],dy=pos[j*3+1]-pos[i*3+1],dz=pos[j*3+2]-pos[i*3+2];
            mimg9(dx,dy,dz,hi,h);
            if(dx*dx+dy*dy+dz*dz<rc2){
                if(nlc[i]<MAX_LJ_NEIGH){nll[i*MAX_LJ_NEIGH+nlc[i]]=j;nlc[i]++;}}}}}

/* ═══════════ Force kernels ═══════════ */
struct ForceResult{double Eb,Ea,Ed,Ei,Elj,Ecoul,Etot;};

static ForceResult compute_forces(double*F,double*vir9,
    const double*pos,const double*h,const double*hi,
    /* bond */ int Nb,const int*b_i0,const int*b_i1,const double*b_kb,const double*b_r0,
    /* angle */ int Nang,const int*ang_i0,const int*ang_i1,const int*ang_i2,const double*ang_kth,const double*ang_th0,
    /* dihedral */ int Ndih,const int*dih_i0,const int*dih_i1,const int*dih_i2,const int*dih_i3,
                   const double*dih_Vn,const int*dih_mult,const double*dih_gamma,
    /* improper */ int Nimp,const int*imp_i0,const int*imp_i1,const int*imp_i2,const int*imp_i3,const double*imp_ki,const double*imp_gamma,
    /* LJ NL */ const int*nlc,const int*nll,
    /* Coulomb */ const double*charge,const int*mol_id,bool has_charge,
    int Na)
{
    /* Zero forces and virial */
#ifdef _OPENACC
    #pragma acc parallel loop present(F)
#elif defined(_OPENMP)
    #pragma omp parallel for schedule(static)
#endif
    for(int i=0;i<Na*3;i++) F[i]=0;
#ifdef _OPENACC
    #pragma acc parallel loop present(vir9)
#elif defined(_OPENMP)
    #pragma omp parallel for schedule(static)
#endif
    for(int i=0;i<9;i++) vir9[i]=0;

    double Eb=0,Ea=0,Ed=0,Ei=0,Elj=0;

    /* ── 1. Bond stretching ── */
#ifdef _OPENACC
    #pragma acc parallel loop present(pos,F,vir9,b_i0,b_i1,b_kb,b_r0,h,hi) reduction(+:Eb)
#elif defined(_OPENMP)
    #pragma omp parallel for reduction(+:Eb)
#endif
    for(int b=0;b<Nb;b++){
        int i=b_i0[b],j=b_i1[b];
        double dx=pos[j*3]-pos[i*3],dy=pos[j*3+1]-pos[i*3+1],dz=pos[j*3+2]-pos[i*3+2];
        mimg9(dx,dy,dz,hi,h);
        double r=sqrt(dx*dx+dy*dy+dz*dz); if(r<1e-10)continue;
        double dr=r-b_r0[b];
        Eb+=0.5*b_kb[b]*dr*dr;
        double fm=-b_kb[b]*dr/r;
        double fx=fm*dx,fy=fm*dy,fz=fm*dz;
#if defined(_OPENACC)
        #pragma acc atomic update
#elif defined(_OPENMP)
        #pragma omp atomic update
#endif
        F[i*3]+=fx;
#if defined(_OPENACC)
        #pragma acc atomic update
#elif defined(_OPENMP)
        #pragma omp atomic update
#endif
        F[i*3+1]+=fy;
#if defined(_OPENACC)
        #pragma acc atomic update
#elif defined(_OPENMP)
        #pragma omp atomic update
#endif
        F[i*3+2]+=fz;
#if defined(_OPENACC)
        #pragma acc atomic update
#elif defined(_OPENMP)
        #pragma omp atomic update
#endif
        F[j*3]-=fx;
#if defined(_OPENACC)
        #pragma acc atomic update
#elif defined(_OPENMP)
        #pragma omp atomic update
#endif
        F[j*3+1]-=fy;
#if defined(_OPENACC)
        #pragma acc atomic update
#elif defined(_OPENMP)
        #pragma omp atomic update
#endif
        F[j*3+2]-=fz;
#if defined(_OPENACC)
        #pragma acc atomic update
#elif defined(_OPENMP)
        #pragma omp atomic update
#endif
        vir9[0]+=dx*fx;
#if defined(_OPENACC)
        #pragma acc atomic update
#elif defined(_OPENMP)
        #pragma omp atomic update
#endif
        vir9[4]+=dy*fy;
#if defined(_OPENACC)
        #pragma acc atomic update
#elif defined(_OPENMP)
        #pragma omp atomic update
#endif
        vir9[8]+=dz*fz;
    }

    /* ── 2. Angle bending ── */
#ifdef _OPENACC
    #pragma acc parallel loop present(pos,F,ang_i0,ang_i1,ang_i2,ang_kth,ang_th0,h,hi) reduction(+:Ea)
#elif defined(_OPENMP)
    #pragma omp parallel for reduction(+:Ea)
#endif
    for(int a=0;a<Nang;a++){
        int i=ang_i0[a],j=ang_i1[a],k=ang_i2[a];
        double rji0=pos[i*3]-pos[j*3],rji1=pos[i*3+1]-pos[j*3+1],rji2=pos[i*3+2]-pos[j*3+2];
        double rjk0=pos[k*3]-pos[j*3],rjk1=pos[k*3+1]-pos[j*3+1],rjk2=pos[k*3+2]-pos[j*3+2];
        mimg9(rji0,rji1,rji2,hi,h); mimg9(rjk0,rjk1,rjk2,hi,h);
        double dji=sqrt(rji0*rji0+rji1*rji1+rji2*rji2);
        double djk=sqrt(rjk0*rjk0+rjk1*rjk1+rjk2*rjk2);
        if(dji<1e-10||djk<1e-10)continue;
        double costh=(rji0*rjk0+rji1*rjk1+rji2*rjk2)/(dji*djk);
        if(costh>0.999999)costh=0.999999; if(costh<-0.999999)costh=-0.999999;
        double th=acos(costh),dth=th-ang_th0[a];
        Ea+=0.5*ang_kth[a]*dth*dth;
        double sinth=sqrt(1.0-costh*costh+1e-30);
        double dV=-ang_kth[a]*dth/sinth;
        /* Forces on i,k; j gets reaction */
        double rji_arr[3]={rji0,rji1,rji2},rjk_arr[3]={rjk0,rjk1,rjk2};
        for(int c=0;c<3;c++){
            double fi=dV*(rjk_arr[c]/(dji*djk)-costh*rji_arr[c]/(dji*dji));
            double fk=dV*(rji_arr[c]/(dji*djk)-costh*rjk_arr[c]/(djk*djk));
#if defined(_OPENACC)
            #pragma acc atomic update
#elif defined(_OPENMP)
            #pragma omp atomic update
#endif
            F[i*3+c]+=fi;
#if defined(_OPENACC)
            #pragma acc atomic update
#elif defined(_OPENMP)
            #pragma omp atomic update
#endif
            F[k*3+c]+=fk;
#if defined(_OPENACC)
            #pragma acc atomic update
#elif defined(_OPENMP)
            #pragma omp atomic update
#endif
            F[j*3+c]-=fi+fk;}
    }

    /* ── 3. Dihedral torsion (analytical forces) ── */
#ifdef _OPENACC
    #pragma acc parallel loop present(pos,F,dih_i0,dih_i1,dih_i2,dih_i3,dih_Vn,dih_mult,dih_gamma,h,hi) reduction(+:Ed)
#elif defined(_OPENMP)
    #pragma omp parallel for schedule(dynamic) reduction(+:Ed)
#endif
    for(int d=0;d<Ndih;d++){
        int i0=dih_i0[d],i1=dih_i1[d],i2=dih_i2[d],i3=dih_i3[d];
        /* b1=r12, b2=r23, b3=r34 */
        double b1x=pos[i1*3]-pos[i0*3],b1y=pos[i1*3+1]-pos[i0*3+1],b1z=pos[i1*3+2]-pos[i0*3+2];
        double b2x=pos[i2*3]-pos[i1*3],b2y=pos[i2*3+1]-pos[i1*3+1],b2z=pos[i2*3+2]-pos[i1*3+2];
        double b3x=pos[i3*3]-pos[i2*3],b3y=pos[i3*3+1]-pos[i2*3+1],b3z=pos[i3*3+2]-pos[i2*3+2];
        mimg9(b1x,b1y,b1z,hi,h); mimg9(b2x,b2y,b2z,hi,h); mimg9(b3x,b3y,b3z,hi,h);
        /* m = b1 x b2, n = b2 x b3 */
        double mx=b1y*b2z-b1z*b2y,my=b1z*b2x-b1x*b2z,mz=b1x*b2y-b1y*b2x;
        double nx=b2y*b3z-b2z*b3y,ny=b2z*b3x-b2x*b3z,nz=b2x*b3y-b2y*b3x;
        double mm=mx*mx+my*my+mz*mz,nn=nx*nx+ny*ny+nz*nz;
        if(mm<1e-20||nn<1e-20)continue;
        double imm=1.0/mm,inn=1.0/nn;
        double b2len=sqrt(b2x*b2x+b2y*b2y+b2z*b2z);
        /* cos(phi), sin(phi) */
        double cosphi=(mx*nx+my*ny+mz*nz)/sqrt(mm*nn);
        if(cosphi>1.0)cosphi=1.0;if(cosphi<-1.0)cosphi=-1.0;
        double b1dotb2=b1x*b2x+b1y*b2y+b1z*b2z;
        double b2dotb3=b2x*b3x+b2y*b3y+b2z*b3z;
        double sinphi=(mx*b3x+my*b3y+mz*b3z)*b2len/sqrt(mm*nn);
        double phi=atan2(sinphi,cosphi);
        int mult=dih_mult[d]; double gamma=dih_gamma[d],Vn=dih_Vn[d];
        Ed+=0.5*Vn*(1.0+cos(mult*phi-gamma));
        double dphi=-0.5*Vn*mult*sin(mult*phi-gamma);
        /* Analytical: dV/dr using m,n vectors (Bekker et al.) */
        double f1x= dphi*b2len*imm*mx,         f1y= dphi*b2len*imm*my,         f1z= dphi*b2len*imm*mz;
        double f4x=-dphi*b2len*inn*nx,          f4y=-dphi*b2len*inn*ny,          f4z=-dphi*b2len*inn*nz;
        double coef2i=b1dotb2/(b2len*b2len),coef2k=b2dotb3/(b2len*b2len);
        double f2x=-f1x+coef2i*f1x-coef2k*f4x, f2y=-f1y+coef2i*f1y-coef2k*f4y, f2z=-f1z+coef2i*f1z-coef2k*f4z;
        double f3x=-f4x-coef2i*f1x+coef2k*f4x, f3y=-f4y-coef2i*f1y+coef2k*f4y, f3z=-f4z-coef2i*f1z+coef2k*f4z;
#if defined(_OPENACC)
        #pragma acc atomic update
#elif defined(_OPENMP)
        #pragma omp atomic update
#endif
        F[i0*3]+=f1x;
#if defined(_OPENACC)
        #pragma acc atomic update
#elif defined(_OPENMP)
        #pragma omp atomic update
#endif
        F[i0*3+1]+=f1y;
#if defined(_OPENACC)
        #pragma acc atomic update
#elif defined(_OPENMP)
        #pragma omp atomic update
#endif
        F[i0*3+2]+=f1z;
#if defined(_OPENACC)
        #pragma acc atomic update
#elif defined(_OPENMP)
        #pragma omp atomic update
#endif
        F[i1*3]+=f2x;
#if defined(_OPENACC)
        #pragma acc atomic update
#elif defined(_OPENMP)
        #pragma omp atomic update
#endif
        F[i1*3+1]+=f2y;
#if defined(_OPENACC)
        #pragma acc atomic update
#elif defined(_OPENMP)
        #pragma omp atomic update
#endif
        F[i1*3+2]+=f2z;
#if defined(_OPENACC)
        #pragma acc atomic update
#elif defined(_OPENMP)
        #pragma omp atomic update
#endif
        F[i2*3]+=f3x;
#if defined(_OPENACC)
        #pragma acc atomic update
#elif defined(_OPENMP)
        #pragma omp atomic update
#endif
        F[i2*3+1]+=f3y;
#if defined(_OPENACC)
        #pragma acc atomic update
#elif defined(_OPENMP)
        #pragma omp atomic update
#endif
        F[i2*3+2]+=f3z;
#if defined(_OPENACC)
        #pragma acc atomic update
#elif defined(_OPENMP)
        #pragma omp atomic update
#endif
        F[i3*3]+=f4x;
#if defined(_OPENACC)
        #pragma acc atomic update
#elif defined(_OPENMP)
        #pragma omp atomic update
#endif
        F[i3*3+1]+=f4y;
#if defined(_OPENACC)
        #pragma acc atomic update
#elif defined(_OPENMP)
        #pragma omp atomic update
#endif
        F[i3*3+2]+=f4z;
    }

    /* ── 4. Improper dihedral (same analytical formula, center=i1) ── */
#ifdef _OPENACC
    #pragma acc parallel loop present(pos,F,imp_i0,imp_i1,imp_i2,imp_i3,imp_ki,imp_gamma,h,hi) reduction(+:Ei)
#elif defined(_OPENMP)
    #pragma omp parallel for reduction(+:Ei)
#endif
    for(int p=0;p<Nimp;p++){
        /* center=i0, neighbors=i1,i2,i3  ->  dihedral order: i1-i0-i2-i3 */
        int c0=imp_i0[p],c1=imp_i1[p],c2=imp_i2[p],c3=imp_i3[p];
        double b1x=pos[c0*3]-pos[c1*3],b1y=pos[c0*3+1]-pos[c1*3+1],b1z=pos[c0*3+2]-pos[c1*3+2];
        double b2x=pos[c2*3]-pos[c0*3],b2y=pos[c2*3+1]-pos[c0*3+1],b2z=pos[c2*3+2]-pos[c0*3+2];
        double b3x=pos[c3*3]-pos[c2*3],b3y=pos[c3*3+1]-pos[c2*3+1],b3z=pos[c3*3+2]-pos[c2*3+2];
        mimg9(b1x,b1y,b1z,hi,h);mimg9(b2x,b2y,b2z,hi,h);mimg9(b3x,b3y,b3z,hi,h);
        double mx=b1y*b2z-b1z*b2y,my=b1z*b2x-b1x*b2z,mz=b1x*b2y-b1y*b2x;
        double nx=b2y*b3z-b2z*b3y,ny=b2z*b3x-b2x*b3z,nz=b2x*b3y-b2y*b3x;
        double mm=mx*mx+my*my+mz*mz,nn=nx*nx+ny*ny+nz*nz;
        if(mm<1e-20||nn<1e-20)continue;
        double imm=1.0/mm,inn=1.0/nn;
        double b2len=sqrt(b2x*b2x+b2y*b2y+b2z*b2z);
        double sinphi=(mx*b3x+my*b3y+mz*b3z)*b2len/sqrt(mm*nn);
        double cosphi=(mx*nx+my*ny+mz*nz)/sqrt(mm*nn);
        if(cosphi>1.0)cosphi=1.0;if(cosphi<-1.0)cosphi=-1.0;
        double phi=atan2(sinphi,cosphi);
        double ki=imp_ki[p],gm=imp_gamma[p];
        Ei+=0.5*ki*(1.0+cos(2*phi-gm));
        double dphi=-0.5*ki*2*sin(2*phi-gm);
        double b1db2=b1x*b2x+b1y*b2y+b1z*b2z,b2db3=b2x*b3x+b2y*b3y+b2z*b3z;
        double f1x=dphi*b2len*imm*mx,f1y=dphi*b2len*imm*my,f1z=dphi*b2len*imm*mz;
        double f4x=-dphi*b2len*inn*nx,f4y=-dphi*b2len*inn*ny,f4z=-dphi*b2len*inn*nz;
        double c2i=b1db2/(b2len*b2len),c2k=b2db3/(b2len*b2len);
        /* Map: atom1->c1, atom2->c0, atom3->c2, atom4->c3 */
#if defined(_OPENACC)
        #pragma acc atomic update
#elif defined(_OPENMP)
        #pragma omp atomic update
#endif
        F[c1*3]+=f1x;
#if defined(_OPENACC)
        #pragma acc atomic update
#elif defined(_OPENMP)
        #pragma omp atomic update
#endif
        F[c1*3+1]+=f1y;
#if defined(_OPENACC)
        #pragma acc atomic update
#elif defined(_OPENMP)
        #pragma omp atomic update
#endif
        F[c1*3+2]+=f1z;
        double f2x=-f1x+c2i*f1x-c2k*f4x,f2y=-f1y+c2i*f1y-c2k*f4y,f2z=-f1z+c2i*f1z-c2k*f4z;
#if defined(_OPENACC)
        #pragma acc atomic update
#elif defined(_OPENMP)
        #pragma omp atomic update
#endif
        F[c0*3]+=f2x;
#if defined(_OPENACC)
        #pragma acc atomic update
#elif defined(_OPENMP)
        #pragma omp atomic update
#endif
        F[c0*3+1]+=f2y;
#if defined(_OPENACC)
        #pragma acc atomic update
#elif defined(_OPENMP)
        #pragma omp atomic update
#endif
        F[c0*3+2]+=f2z;
        double f3x=-f4x-c2i*f1x+c2k*f4x,f3y=-f4y-c2i*f1y+c2k*f4y,f3z=-f4z-c2i*f1z+c2k*f4z;
#if defined(_OPENACC)
        #pragma acc atomic update
#elif defined(_OPENMP)
        #pragma omp atomic update
#endif
        F[c2*3]+=f3x;
#if defined(_OPENACC)
        #pragma acc atomic update
#elif defined(_OPENMP)
        #pragma omp atomic update
#endif
        F[c2*3+1]+=f3y;
#if defined(_OPENACC)
        #pragma acc atomic update
#elif defined(_OPENMP)
        #pragma omp atomic update
#endif
        F[c2*3+2]+=f3z;
#if defined(_OPENACC)
        #pragma acc atomic update
#elif defined(_OPENMP)
        #pragma omp atomic update
#endif
        F[c3*3]+=f4x;
#if defined(_OPENACC)
        #pragma acc atomic update
#elif defined(_OPENMP)
        #pragma omp atomic update
#endif
        F[c3*3+1]+=f4y;
#if defined(_OPENACC)
        #pragma acc atomic update
#elif defined(_OPENMP)
        #pragma omp atomic update
#endif
        F[c3*3+2]+=f4z;
    }

    /* ── 5. LJ intermolecular (half-list + atomic) ── */
#ifdef _OPENACC
    #pragma acc parallel loop gang present(pos,F,vir9,nlc,nll,h,hi) reduction(+:Elj)
#elif defined(_OPENMP)
    #pragma omp parallel for schedule(dynamic,4) reduction(+:Elj)
#endif
    for(int i=0;i<Na;i++){
        int nni=nlc[i];
        for(int jn=0;jn<nni;jn++){
            int j=nll[i*MAX_LJ_NEIGH+jn];
            double dx=pos[j*3]-pos[i*3],dy=pos[j*3+1]-pos[i*3+1],dz=pos[j*3+2]-pos[i*3+2];
            mimg9(dx,dy,dz,hi,h);
            double r2=dx*dx+dy*dy+dz*dz;if(r2>RCUT2)continue;if(r2<0.25)r2=0.25;
            double ri2=1.0/r2,sr2=sig2_LJ*ri2,sr6=sr2*sr2*sr2,sr12=sr6*sr6;
            double fm=24*eps_LJ*(2*sr12-sr6)*ri2;
            Elj+=4*eps_LJ*(sr12-sr6)-VSHFT;
            F[i*3]-=fm*dx;F[i*3+1]-=fm*dy;F[i*3+2]-=fm*dz;
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
#if defined(_OPENACC)
            #pragma acc atomic update
#elif defined(_OPENMP)
            #pragma omp atomic update
#endif
            vir9[0]+=dx*fm*dx;
#if defined(_OPENACC)
            #pragma acc atomic update
#elif defined(_OPENMP)
            #pragma omp atomic update
#endif
            vir9[4]+=dy*fm*dy;
#if defined(_OPENACC)
            #pragma acc atomic update
#elif defined(_OPENMP)
            #pragma omp atomic update
#endif
            vir9[8]+=dz*fm*dz;
        }}

    /* ── 6. Coulomb (Serial only -- skipped when all charges are zero) ── */
    double Ecoul=0;
    if(has_charge){
        for(int i=0;i<Na-1;i++){
            for(int j=i+1;j<Na;j++){
                if(mol_id[i]==mol_id[j])continue;
                double dx=pos[j*3]-pos[i*3],dy=pos[j*3+1]-pos[i*3+1],dz=pos[j*3+2]-pos[i*3+2];
                mimg9(dx,dy,dz,hi,h);
                double r2=dx*dx+dy*dy+dz*dz;
                if(r2>COUL_RCUT2)continue;
                double r=sqrt(r2);
                double qi=charge[i],qj=charge[j];
                double Vc=COULOMB_K*qi*qj/r;
                double Vc_cut=COULOMB_K*qi*qj/COUL_RCUT;
                Ecoul+=Vc-Vc_cut;
                double fm=COULOMB_K*qi*qj/(r2*r);
                F[i*3]-=fm*dx; F[i*3+1]-=fm*dy; F[i*3+2]-=fm*dz;
                F[j*3]+=fm*dx; F[j*3+1]+=fm*dy; F[j*3+2]+=fm*dz;
            }
        }
    }

    return{Eb,Ea,Ed,Ei,Elj,Ecoul,Eb+Ea+Ed+Ei+Elj+Ecoul};}

/* ═══════════ KE ═══════════ */
static double ke_total(const double*vel,int Na){
    double s=0;
#ifdef _OPENACC
    #pragma acc parallel loop present(vel) reduction(+:s)
#elif defined(_OPENMP)
    #pragma omp parallel for reduction(+:s)
#endif
    for(int i=0;i<Na;i++) s+=mC*(vel[i*3]*vel[i*3]+vel[i*3+1]*vel[i*3+1]+vel[i*3+2]*vel[i*3+2]);
    return 0.5*s/CONV;}
static double inst_T(double KE,int Nf){return 2*KE/(Nf*kB);}
static double inst_P(const double*W,double KE,double V){return(2*KE+W[0]+W[4]+W[8])/(3*V)*eV2GPa;}

/* ═══════════ NPT state ═══════════ */
struct NPTState{double xi,Q,Vg[9],W_,Pe,Tt;int Nf;};
static NPTState make_npt(double T,double Pe,int Na){
    int Nf=3*Na-3;NPTState s;s.xi=0;s.Q=std::max(Nf*kB*T*1e4,1e-20);
    for(int i=0;i<9;i++)s.Vg[i]=0;
    s.W_=std::max((Nf+9)*kB*T*1e6,1e-20);s.Pe=Pe;s.Tt=T;s.Nf=Nf;return s;}

/* ═══════════ NPT step ═══════════ */
static std::pair<ForceResult,double>
step_npt(double*pos,double*vel,double*F,double*vir9,double*h,double*hi,
    int Na,double dt,NPTState&npt,FlatTopology&ft,const int*nlc,const int*nll,
    const double*charge,const int*mol_id,bool has_charge)
{
    double hdt=0.5*dt,V=fabs(mat_det9(h));
    double KE=ke_total(vel,Na);
    npt.xi+=hdt*(2*KE-npt.Nf*kB*npt.Tt)/npt.Q; npt.xi=std::clamp(npt.xi,-0.05,0.05);
#ifdef _OPENACC
    #pragma acc update self(vir9[0:9])
#endif
    double dP=inst_P(vir9,KE,V)-npt.Pe;
    for(int a=0;a<3;a++)npt.Vg[a*4]+=hdt*V*dP/(npt.W_*eV2GPa);
    for(int a=0;a<3;a++)npt.Vg[a*4]=std::clamp(npt.Vg[a*4],-0.005,0.005);
    double eps=npt.Vg[0]*hi[0]+npt.Vg[4]*hi[4]+npt.Vg[8]*hi[8];
    double sc_nh=exp(-hdt*npt.xi),sc_pr=exp(-hdt*eps/3.0),sc_v=sc_nh*sc_pr;
    double mi_inv=CONV/mC;
#ifdef _OPENACC
    #pragma acc parallel loop present(vel,F)
#elif defined(_OPENMP)
    #pragma omp parallel for schedule(static)
#endif
    for(int i=0;i<Na;i++){
        vel[i*3  ]=vel[i*3  ]*sc_v+hdt*F[i*3  ]*mi_inv;
        vel[i*3+1]=vel[i*3+1]*sc_v+hdt*F[i*3+1]*mi_inv;
        vel[i*3+2]=vel[i*3+2]*sc_v+hdt*F[i*3+2]*mi_inv;}
#ifdef _OPENACC
    #pragma acc parallel loop present(pos,vel,hi)
#elif defined(_OPENMP)
    #pragma omp parallel for schedule(static)
#endif
    for(int i=0;i<Na;i++){
        double px=pos[i*3],py=pos[i*3+1],pz=pos[i*3+2];
        double vx=vel[i*3],vy=vel[i*3+1],vz=vel[i*3+2];
        double sx=hi[0]*px+hi[1]*py+hi[2]*pz,sy=hi[3]*px+hi[4]*py+hi[5]*pz,sz=hi[6]*px+hi[7]*py+hi[8]*pz;
        double vsx=hi[0]*vx+hi[1]*vy+hi[2]*vz,vsy=hi[3]*vx+hi[4]*vy+hi[5]*vz,vsz=hi[6]*vx+hi[7]*vy+hi[8]*vz;
        sx+=dt*vsx;sy+=dt*vsy;sz+=dt*vsz;sx-=floor(sx);sy-=floor(sy);sz-=floor(sz);
        pos[i*3]=sx;pos[i*3+1]=sy;pos[i*3+2]=sz;}
    for(int a=0;a<3;a++)for(int b=0;b<3;b++) h[a*3+b]+=dt*npt.Vg[a*3+b];
    mat_inv9(h,hi);
#ifdef _OPENACC
    #pragma acc update device(h[0:9],hi[0:9])
#endif
#ifdef _OPENACC
    #pragma acc parallel loop present(pos,h)
#elif defined(_OPENMP)
    #pragma omp parallel for schedule(static)
#endif
    for(int i=0;i<Na;i++){
        double sx=pos[i*3],sy=pos[i*3+1],sz=pos[i*3+2];
        pos[i*3]=h[0]*sx+h[1]*sy+h[2]*sz;
        pos[i*3+1]=h[3]*sx+h[4]*sy+h[5]*sz;
        pos[i*3+2]=h[6]*sx+h[7]*sy+h[8]*sz;}
    auto fr=compute_forces(F,vir9,pos,h,hi,
        ft.Nb,ft.b_i0,ft.b_i1,ft.b_kb,ft.b_r0,
        ft.Nang,ft.ang_i0,ft.ang_i1,ft.ang_i2,ft.ang_kth,ft.ang_th0,
        ft.Ndih,ft.dih_i0,ft.dih_i1,ft.dih_i2,ft.dih_i3,ft.dih_Vn,ft.dih_mult,ft.dih_gamma,
        ft.Nimp,ft.imp_i0,ft.imp_i1,ft.imp_i2,ft.imp_i3,ft.imp_ki,ft.imp_gamma,
        nlc,nll,charge,mol_id,has_charge,Na);
    double eps2=npt.Vg[0]*hi[0]+npt.Vg[4]*hi[4]+npt.Vg[8]*hi[8];
    double sc_v2=sc_nh*exp(-hdt*eps2/3.0);
#ifdef _OPENACC
    #pragma acc parallel loop present(vel,F)
#elif defined(_OPENMP)
    #pragma omp parallel for schedule(static)
#endif
    for(int i=0;i<Na;i++){
        vel[i*3  ]=(vel[i*3  ]+hdt*F[i*3  ]*mi_inv)*sc_v2;
        vel[i*3+1]=(vel[i*3+1]+hdt*F[i*3+1]*mi_inv)*sc_v2;
        vel[i*3+2]=(vel[i*3+2]+hdt*F[i*3+2]*mi_inv)*sc_v2;}
    KE=ke_total(vel,Na);
    npt.xi+=hdt*(2*KE-npt.Nf*kB*npt.Tt)/npt.Q; npt.xi=std::clamp(npt.xi,-0.05,0.05);
    double V2=fabs(mat_det9(h));
#ifdef _OPENACC
    #pragma acc update self(vir9[0:9])
#endif
    dP=inst_P(vir9,KE,V2)-npt.Pe;
    for(int a=0;a<3;a++)npt.Vg[a*4]+=hdt*V2*dP/(npt.W_*eV2GPa);
    for(int a=0;a<3;a++)npt.Vg[a*4]=std::clamp(npt.Vg[a*4],-0.005,0.005);
    return{fr,KE};}

/* ═══════════ OVITO (host) ═══════════ */
static void write_ovito(FILE*fp,int step,double dt,const double*pos,const double*vel,
    const int*mol_id,const double*h,int Na){
    fprintf(fp,"%d\n",Na);
    fprintf(fp,"Lattice=\"%.8f %.8f %.8f %.8f %.8f %.8f %.8f %.8f %.8f\" "
        "Properties=species:S:1:pos:R:3:c_mol:I:1:vx:R:1:vy:R:1:vz:R:1 Time=%.4f Step=%d pbc=\"T T T\"\n",
        h[0],h[3],h[6],h[1],h[4],h[7],h[2],h[5],h[8],step*dt,step);
    for(int i=0;i<Na;i++)
        fprintf(fp,"C %14.8f %14.8f %14.8f %5d %14.8e %14.8e %14.8e\n",
            pos[i*3],pos[i*3+1],pos[i*3+2],mol_id[i]+1,vel[i*3],vel[i*3+1],vel[i*3+2]);}

/* ═══════════ CLI ═══════════ */
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

/* ═══════════ RESTART ═══════════ */
std::string restart_filename(const std::string&oname,int istep,int nsteps){
    std::string b=oname;auto dt=b.rfind('.');if(dt!=std::string::npos)b=b.substr(0,dt);
    auto p=b.find("ovito_traj");if(p!=std::string::npos)b.replace(p,10,"restart");
    if(istep==nsteps)return b+".rst";
    int dg=1;for(int x=nsteps;x>=10;x/=10)dg++;
    char buf[64];snprintf(buf,64,"_%0*d",dg,istep);return b+buf+".rst";}

struct RestartDataMMD{
    int istep,Na; double h[9]; NPTState npt;
    std::vector<double> pos,vel; bool ok;};

void write_restart_mmmd(const std::string&fname,int istep,
    const std::map<std::string,std::string>&opts,
    const std::string&st,int nc,double T,double Pe,int nsteps,double dt_val,int seed,
    const std::string&fspec,double init_scale,
    const double*h,const NPTState&npt,
    const double*pos,const double*vel,
    int Na,int Nmol,int natom){
    FILE*f=fopen(fname.c_str(),"w");
    if(!f){fprintf(stderr,"Cannot write restart: %s\n",fname.c_str());return;}
    fprintf(f,"# RESTART fuller_LJ_npt_mmmd_serial_omp_acc\n# OPTIONS:");
    for(auto&[k,v]:opts)fprintf(f," --%s=%s",k.c_str(),v.c_str());
    fprintf(f,"\n");
    fprintf(f,"STEP %d\nNSTEPS %d\nDT %.15e\nTEMP %.15e\nPRES %.15e\n",istep,nsteps,dt_val,T,Pe);
    fprintf(f,"CRYSTAL %s\nNC %d\nFULLERENE %s\nINIT_SCALE %.15e\nSEED %d\n",
            st.c_str(),nc,fspec.c_str(),init_scale,seed);
    fprintf(f,"NMOL %d\nNATOM_MOL %d\nNATOM %d\n",Nmol,natom,Na);
    fprintf(f,"H");for(int i=0;i<9;i++)fprintf(f," %.15e",h[i]);
    fprintf(f,"\nNPT %.15e %.15e %.15e %.15e %.15e %d\n",npt.xi,npt.Q,npt.W_,npt.Pe,npt.Tt,npt.Nf);
    fprintf(f,"VG");for(int i=0;i<9;i++)fprintf(f," %.15e",npt.Vg[i]);
    fprintf(f,"\n");
    for(int i=0;i<Na;i++){
        fprintf(f,"ATOM %d %.15e %.15e %.15e %.15e %.15e %.15e\n",
            i+1,pos[i*3],pos[i*3+1],pos[i*3+2],vel[i*3],vel[i*3+1],vel[i*3+2]);}
    fprintf(f,"END\n");fclose(f);}

RestartDataMMD read_restart_mmmd(const std::string&fname){
    RestartDataMMD rd;rd.ok=false;
    std::ifstream f(fname);if(!f){fprintf(stderr,"Cannot read: %s\n",fname.c_str());return rd;}
    std::string line;
    for(int i=0;i<9;i++)rd.h[i]=0;
    rd.npt={0,0,{0,0,0,0,0,0,0,0,0},0,0,0,0};
    while(std::getline(f,line)){
        if(line.empty()||line[0]=='#')continue;
        std::istringstream ss(line);std::string tag;ss>>tag;
        if(tag=="STEP")ss>>rd.istep;
        else if(tag=="NATOM")ss>>rd.Na;
        else if(tag=="H"){for(int i=0;i<9;i++)ss>>rd.h[i];}
        else if(tag=="NPT"){ss>>rd.npt.xi>>rd.npt.Q>>rd.npt.W_>>rd.npt.Pe>>rd.npt.Tt>>rd.npt.Nf;}
        else if(tag=="VG"){for(int i=0;i<9;i++)ss>>rd.npt.Vg[i];}
        else if(tag=="ATOM"){
            int idx;double px,py,pz,vx,vy,vz;ss>>idx>>px>>py>>pz>>vx>>vy>>vz;
            rd.pos.push_back(px);rd.pos.push_back(py);rd.pos.push_back(pz);
            rd.vel.push_back(vx);rd.vel.push_back(vy);rd.vel.push_back(vz);}
        else if(tag=="END")break;}
    rd.Na=(int)rd.pos.size()/3;rd.ok=(rd.Na>0);
    if(rd.ok)printf("  Restart loaded: %s (step %d, %d atoms)\n",fname.c_str(),rd.istep,rd.Na);
    return rd;}

/* ═══════════ MAIN ═══════════ */
int main(int argc,char**argv){
    auto opts=parse_args(argc,argv);
    if(opts.count("help")){
        printf(
            "fuller_LJ_npt_mmmd_serial_omp_acc — MM+LJ fullerene NPT-MD\n\n"
            "Options:\n"
            "  --help                  Show this help\n"
            "  --fullerene=<name>      Fullerene species (default: C60)\n"
            "  --crystal=<fcc|hcp|bcc> Crystal structure (default: fcc)\n"
            "  --cell=<nc>             Unit cell repeats (default: 3)\n"
            "  --temp=<K>              Target temperature [K] (default: 298.0)\n"
            "  --pres=<GPa>            Target pressure [GPa] (default: 0.0)\n"
            "  --step=<N>              Production steps (default: 10000)\n"
            "  --dt=<fs>               Time step [fs] (default: 0.1)\n"
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
            "  --libdir=<path>         Fullerene library dir (default: FullereneLib)\n"
            "  --ff_kb=<kcal/mol>      Bond stretch constant (default: 469.0)\n"
            "  --ff_kth=<kcal/mol>     Angle bend constant (default: 63.0)\n"
            "  --ff_v2=<kcal/mol>      Dihedral constant (default: 14.5)\n"
            "  --ff_kimp=<kcal/mol>    Improper constant (default: 15.0)\n\n"
            "Examples:\n"
            "  ./prog --temp=500 --pres=1.0 --step=100000\n"
            "  ./prog --coldstart=5000 --warmup=5000 --step=50000\n"
            "  ./prog --step=20000 --ovito=200\n"
            "  ./prog --step=100000 --restart=10000\n"
            "  ./prog --resfile=restart_mmmd_serial_00050000.rst\n"
            "  ./prog --ff_kb=500 --ff_kth=70 --step=20000\n");
        return 0;}
    std::string crystal=gopt(opts,"crystal","fcc");
    std::string st=crystal;std::transform(st.begin(),st.end(),st.begin(),::toupper);
    int nc=std::atoi(gopt(opts,"cell","3").c_str());
    double T=std::atof(gopt(opts,"temp","298.0").c_str());
    double Pe=std::atof(gopt(opts,"pres","0.0").c_str());
    int nsteps=std::atoi(gopt(opts,"step","10000").c_str());
    double dt=std::atof(gopt(opts,"dt","0.1").c_str());
    int seed=std::atoi(gopt(opts,"seed","42").c_str());
    double init_scale=std::atof(gopt(opts,"init_scale","1.0").c_str());
    std::string fspec=gopt(opts,"fullerene","C60");
    std::string libdir=gopt(opts,"libdir","FullereneLib");
    int coldstart=std::atoi(gopt(opts,"coldstart","0").c_str());
    int warmup=std::atoi(gopt(opts,"warmup","0").c_str());
    int avg_from=std::atoi(gopt(opts,"from","0").c_str());
    int avg_to=std::atoi(gopt(opts,"to","0").c_str());
    int nrec_o=std::atoi(gopt(opts,"ovito","0").c_str());
    int mon_interval=std::atoi(gopt(opts,"mon","0").c_str());
    int nrec_rst=std::atoi(gopt(opts,"restart","0").c_str());
    std::string resfile=gopt(opts,"resfile","");
    int start_step=0;
    std::string warmup_mon_mode=gopt(opts,"warmup_mon","norm");
    constexpr double T_cold=4.0;
    double ff_kb=std::atof(gopt(opts,"ff_kb","469.0").c_str())*kcal2eV;
    double ff_kth=std::atof(gopt(opts,"ff_kth","63.0").c_str())*kcal2eV;
    double ff_v2=std::atof(gopt(opts,"ff_v2","14.5").c_str())*kcal2eV;
    double ff_kimp=std::atof(gopt(opts,"ff_kimp","15.0").c_str())*kcal2eV;

    if(avg_to<=0)avg_to=nsteps;if(avg_from<=0)avg_from=std::max(1,nsteps-nsteps/4);
    int total_steps=coldstart+warmup+nsteps;
    int gavg_from=coldstart+warmup+avg_from,gavg_to=coldstart+warmup+avg_to;

    auto[fpath,label]=resolve_fullerene(fspec,libdir);
    MolData mol=load_cc1(fpath);
    double a0=default_a0(mol.Dmol,st,init_scale);
    int Nmol_max=(st=="FCC")?4*nc*nc*nc:(st=="HCP")?2*nc*nc*nc:2*nc*nc*nc;
    int Na_max=Nmol_max*mol.natom;

    /* Mode tag */
    const char* mode_tag =
#ifdef _OPENACC
        "OpenACC GPU";
#elif defined(_OPENMP)
        "OpenMP";
#else
        "Serial";
#endif

    /* Mode suffix for filenames */
    const char* mode_suffix =
#ifdef _OPENACC
        "_gpu";
#elif defined(_OPENMP)
        "_omp";
#else
        "_serial";
#endif

    /* GPU memory check */
#ifdef _OPENACC
    int ndev=acc_get_num_devices(acc_device_nvidia);
    if(ndev>0){acc_set_device_num(0,acc_device_nvidia);
        long gpu_mem=(long)acc_get_property(0,acc_device_nvidia,acc_property_memory);
        long need=(long)Na_max*(3*8*3+MAX_LJ_NEIGH*4+4);
        if(gpu_mem>0&&need>(long)(gpu_mem*0.80)){
            printf("Error: GPU memory insufficient (need %.1f GB, have %.1f GB)\n",
                need/1024.0/1024.0/1024.0,gpu_mem/1024.0/1024.0/1024.0);return 1;}}
#endif

    /* Build crystal */
    double*mol_centers=new double[Nmol_max*3];
    double h[9]={},hi[9]={},vir9[9]={};
    int Nmol;
    if(st=="FCC")Nmol=make_fcc(a0,nc,mol_centers,h);
    else if(st=="HCP")Nmol=make_hcp(a0,nc,mol_centers,h);
    else Nmol=make_bcc(a0,nc,mol_centers,h);
    int Na=Nmol*mol.natom; mat_inv9(h,hi);

    /* Allocate flat arrays */
    double*pos=new double[Na*3]();
    double*vel=new double[Na*3]();
    double*F  =new double[Na*3]();
    double*charge=new double[Na]();  /* All zero for fullerenes */
    int*nlc=new int[Na]();
    int*nll=new int[Na*MAX_LJ_NEIGH]();

    for(int m=0;m<Nmol;m++)for(int a=0;a<mol.natom;a++){
        int idx=m*mol.natom+a;
        pos[idx*3]=mol_centers[m*3]+mol.coords[a][0];
        pos[idx*3+1]=mol_centers[m*3+1]+mol.coords[a][1];
        pos[idx*3+2]=mol_centers[m*3+2]+mol.coords[a][2];}
    delete[]mol_centers;

    /* Check if any charge is nonzero */
    bool has_charge=false;
    for(int i=0;i<Na&&!has_charge;i++) if(fabs(charge[i])>1e-10) has_charge=true;

    /* Build topology */
    FlatTopology ft=build_flat_topology(mol,Nmol,ff_kb,ff_kth,ff_v2,ff_kimp);

    printf("========================================================================\n");
    printf("  Fullerene Crystal NPT-MD — MM Force Field (%s)\n",mode_tag);
    printf("========================================================================\n");
#ifdef _OPENACC
    printf("  GPU device      : 0 (NVIDIA)\n");
#elif defined(_OPENMP)
    printf("  OpenMP threads  : %d\n",omp_get_max_threads());
#endif
    printf("  Fullerene       : %s (%d atoms/mol)\n",label.c_str(),mol.natom);
    printf("  Crystal         : %s %dx%dx%d  Nmol=%d  Natom=%d\n",st.c_str(),nc,nc,nc,Nmol,Na);
    printf("  a0=%.3f  T=%.1f K  P=%.4f GPa  dt=%.3f fs\n",a0,T,Pe,dt);
    printf("  Production      : %d steps  Total=%d\n",nsteps,total_steps);
    printf("========================================================================\n\n");

    /* Initial velocities */
    double T_init=(coldstart>0||warmup>0)?T_cold:T;
    std::mt19937 rng(seed);std::normal_distribution<double> gauss(0,1);
    double sv=sqrt(kB*T_init*CONV/mC);
    for(int i=0;i<Na;i++)for(int a=0;a<3;a++)vel[i*3+a]=sv*gauss(rng);
    double vcm[3]={0,0,0};
    for(int i=0;i<Na;i++)for(int a=0;a<3;a++)vcm[a]+=vel[i*3+a];
    for(int a=0;a<3;a++)vcm[a]/=Na;
    for(int i=0;i<Na;i++)for(int a=0;a<3;a++)vel[i*3+a]-=vcm[a];

    NPTState npt=make_npt(T,Pe,Na);npt.Tt=T_init;

    if(!resfile.empty()){
        auto rd=read_restart_mmmd(resfile);
        if(rd.ok){
            start_step=rd.istep;
            for(int i=0;i<9;i++)h[i]=rd.h[i];
            mat_inv9(h,hi);
            npt=rd.npt;
            for(int i=0;i<Na*3&&i<rd.Na*3;i++){pos[i]=rd.pos[i];vel[i]=rd.vel[i];}
            printf("  Restarting from global step %d\n",start_step);}}

    build_nlist_lj(pos,h,hi,Na,ft.mol_id,nlc,nll);

    /* ═══ OpenACC data region ═══ */
#ifdef _OPENACC
    #pragma acc data \
        copy(pos[0:Na*3],vel[0:Na*3],F[0:Na*3]) \
        copy(h[0:9],hi[0:9],vir9[0:9]) \
        copyin(ft.b_i0[0:ft.Nb],ft.b_i1[0:ft.Nb],ft.b_kb[0:ft.Nb],ft.b_r0[0:ft.Nb]) \
        copyin(ft.ang_i0[0:ft.Nang],ft.ang_i1[0:ft.Nang],ft.ang_i2[0:ft.Nang],ft.ang_kth[0:ft.Nang],ft.ang_th0[0:ft.Nang]) \
        copyin(ft.dih_i0[0:ft.Ndih],ft.dih_i1[0:ft.Ndih],ft.dih_i2[0:ft.Ndih],ft.dih_i3[0:ft.Ndih],ft.dih_Vn[0:ft.Ndih],ft.dih_mult[0:ft.Ndih],ft.dih_gamma[0:ft.Ndih]) \
        copyin(ft.imp_i0[0:ft.Nimp],ft.imp_i1[0:ft.Nimp],ft.imp_i2[0:ft.Nimp],ft.imp_i3[0:ft.Nimp],ft.imp_ki[0:ft.Nimp],ft.imp_gamma[0:ft.Nimp]) \
        copyin(nlc[0:Na],nll[0:Na*MAX_LJ_NEIGH])
    {
#endif
        apply_pbc(pos,h,hi,Na);
        compute_forces(F,vir9,pos,h,hi,
            ft.Nb,ft.b_i0,ft.b_i1,ft.b_kb,ft.b_r0,
            ft.Nang,ft.ang_i0,ft.ang_i1,ft.ang_i2,ft.ang_kth,ft.ang_th0,
            ft.Ndih,ft.dih_i0,ft.dih_i1,ft.dih_i2,ft.dih_i3,ft.dih_Vn,ft.dih_mult,ft.dih_gamma,
            ft.Nimp,ft.imp_i0,ft.imp_i1,ft.imp_i2,ft.imp_i3,ft.imp_ki,ft.imp_gamma,
            nlc,nll,charge,ft.mol_id,has_charge,Na);

        int prn=(mon_interval>0)?mon_interval:std::max(1,total_steps/50);
        int prn_pre=prn;
        if(coldstart+warmup>0){int div=100;
            if(warmup_mon_mode=="freq")div=10;else if(warmup_mon_mode=="some")div=1000;
            prn_pre=std::max(1,(coldstart+warmup)/div);}
        int nlup=20;
        double sT=0,sP=0,sa=0,sEb=0,sEa=0,sEd=0,sElj=0,sEt=0;int nav=0;
        auto t0=std::chrono::steady_clock::now();
        char ovito_tag[64];snprintf(ovito_tag,64,"ovito_traj_mmmd%s",mode_suffix);
        std::string ovito_file=unique_file(ovito_tag,".xyz");
        FILE*io_o=nrec_o>0?fopen(ovito_file.c_str(),"w"):nullptr;

        printf("  %8s %5s %7s %9s %8s %9s %9s %9s %9s %9s %7s\n",
            "step","phase","T[K]","P[GPa]","a[A]","E_bond","E_angle","E_dih","E_LJ","E_total","t[s]");

        bool stop_requested=false;
        char rst_tag[64];snprintf(rst_tag,64,"restart_mmmd%s",mode_suffix);
        std::string rst_base=nrec_o>0?ovito_file:std::string(rst_tag);
        for(int gstep=start_step+1;gstep<=total_steps;gstep++){
            const char*phase=gstep<=coldstart?"COLD":gstep<=coldstart+warmup?"WARM":"PROD";
            int cur_prn=(gstep<=coldstart+warmup)?prn_pre:prn;

            if(gstep<=coldstart)npt.Tt=T_cold;
            else if(gstep<=coldstart+warmup)npt.Tt=T_cold+(T-T_cold)*double(gstep-coldstart)/double(warmup);
            else npt.Tt=T;
            if(coldstart>0&&gstep==coldstart+1){npt.xi=0;for(int i=0;i<9;i++)npt.Vg[i]=0;}
            if(gstep<=coldstart)for(int i=0;i<9;i++)npt.Vg[i]=0;

            if(gstep%nlup==0){
#ifdef _OPENACC
                #pragma acc update self(pos[0:Na*3])
#endif
                mat_inv9(h,hi);
                build_nlist_lj(pos,h,hi,Na,ft.mol_id,nlc,nll);
#ifdef _OPENACC
                #pragma acc update device(nlc[0:Na],nll[0:Na*MAX_LJ_NEIGH],hi[0:9])
#endif
            }

            auto[fr,KE]=step_npt(pos,vel,F,vir9,h,hi,Na,dt,npt,ft,nlc,nll,
                charge,ft.mol_id,has_charge);
            double V=fabs(mat_det9(h)),Tn=inst_T(KE,npt.Nf);
#ifdef _OPENACC
            #pragma acc update self(vir9[0:9])
#endif
            double Pn=inst_P(vir9,KE,V);

            if((gstep<=coldstart||gstep<=coldstart+warmup)&&Tn>0.1){
                double tgt=(gstep<=coldstart)?T_cold:npt.Tt;
                double scale=sqrt(std::max(tgt,0.1)/Tn);
#ifdef _OPENACC
                #pragma acc parallel loop present(vel)
#elif defined(_OPENMP)
                #pragma omp parallel for schedule(static)
#endif
                for(int i=0;i<Na;i++){vel[i*3]*=scale;vel[i*3+1]*=scale;vel[i*3+2]*=scale;}
                KE=ke_total(vel,Na);Tn=inst_T(KE,npt.Nf);
                npt.xi=0;if(gstep<=coldstart)for(int i=0;i<9;i++)npt.Vg[i]=0;}

            double an=h[0]/nc;
            if(gstep>=gavg_from&&gstep<=gavg_to){
                sT+=Tn;sP+=Pn;sa+=an;sEb+=fr.Eb/Nmol;sEa+=fr.Ea/Nmol;
                sEd+=(fr.Ed+fr.Ei)/Nmol;sElj+=fr.Elj/Nmol;sEt+=fr.Etot/Nmol;nav++;}

            if(io_o&&gstep%nrec_o==0){
#ifdef _OPENACC
                #pragma acc update self(pos[0:Na*3],vel[0:Na*3])
#endif
                write_ovito(io_o,gstep,dt,pos,vel,ft.mol_id,h,Na);fflush(io_o);}

            if(nrec_rst>0&&(gstep%nrec_rst==0||gstep==total_steps)){
#ifdef _OPENACC
                #pragma acc update self(pos[0:Na*3],vel[0:Na*3])
#endif
                write_restart_mmmd(restart_filename(rst_base,gstep,total_steps),
                    gstep,opts,st,nc,T,Pe,nsteps,dt,seed,fspec,init_scale,
                    h,npt,pos,vel,Na,Nmol,mol.natom);
                if(stop_requested){
                    printf("\n  *** Stopped at restart checkpoint (step %d) ***\n",gstep);
                    break;}}

            if(gstep%cur_prn==0||gstep==total_steps){
                if(dir_exists("abort.md")){
                    printf("\n  *** abort.md detected at step %d ***\n",gstep);
                    if(nrec_rst>0){
#ifdef _OPENACC
                        #pragma acc update self(pos[0:Na*3],vel[0:Na*3])
#endif
                        write_restart_mmmd(restart_filename(rst_base,gstep,total_steps),
                            gstep,opts,st,nc,T,Pe,nsteps,dt,seed,fspec,init_scale,
                            h,npt,pos,vel,Na,Nmol,mol.natom);}
                    break;
                }
                if(!stop_requested&&dir_exists("stop.md")){
                    stop_requested=true;
                    printf("\n  *** stop.md detected at step %d — will stop at next checkpoint ***\n",gstep);
                    if(nrec_rst==0)break;}
                double el=std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count();
                printf("  %8d %5s %7.1f %9.3f %8.3f %9.4f %9.4f %9.4f %9.4f %9.4f %7.0f\n",
                    gstep,phase,Tn,Pn,an,fr.Eb/Nmol,fr.Ea/Nmol,(fr.Ed+fr.Ei)/Nmol,fr.Elj/Nmol,fr.Etot/Nmol,el);}
        }
        if(io_o)fclose(io_o);
        if(nav>0){
            printf("\n  Averages (%d): T=%.2f P=%.4f a=%.4f  bond=%.4f ang=%.4f dih=%.4f LJ=%.4f tot=%.4f\n",
                nav,sT/nav,sP/nav,sa/nav,sEb/nav,sEa/nav,sEd/nav,sElj/nav,sEt/nav);}
        printf("  Done (%.1f sec)\n",std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count());
#ifdef _OPENACC
    } /* end acc data */
#endif

    delete[]pos;delete[]vel;delete[]F;delete[]charge;delete[]nlc;delete[]nll;
    delete[]ft.b_i0;delete[]ft.b_i1;delete[]ft.b_kb;delete[]ft.b_r0;
    delete[]ft.ang_i0;delete[]ft.ang_i1;delete[]ft.ang_i2;delete[]ft.ang_kth;delete[]ft.ang_th0;
    delete[]ft.dih_i0;delete[]ft.dih_i1;delete[]ft.dih_i2;delete[]ft.dih_i3;delete[]ft.dih_Vn;delete[]ft.dih_mult;delete[]ft.dih_gamma;
    delete[]ft.imp_i0;delete[]ft.imp_i1;delete[]ft.imp_i2;delete[]ft.imp_i3;delete[]ft.imp_ki;delete[]ft.imp_gamma;
    delete[]ft.mol_id;
    return 0;
}
