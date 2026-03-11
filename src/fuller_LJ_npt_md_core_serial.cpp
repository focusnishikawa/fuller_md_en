/*===========================================================================
  fuller_LJ_npt_md_core_serial.cpp
  C60 Fullerene Crystal NPT Molecular Dynamics Simulation
  (LJ Rigid-Body Model, Core Version -- Single-Thread Serial)

  Compilation:
    g++ -std=c++17 -O3 -o fuller_LJ_core_serial fuller_LJ_npt_md_core_serial.cpp -lm

  Runtime Options:
    ./fuller_LJ_core_serial [nc]

    nc (integer, default: 3, max: 8)
      Number of FCC unit cell repetitions. Number of molecules N = 4*nc^3.
      nc=3 -> N=108, nc=4 -> N=256, nc=5 -> N=500

    How to specify:
      ./fuller_LJ_core_serial 3         # positional argument
      ./fuller_LJ_core_serial --cell=5  # keyword argument

  Examples:
    # Default (3x3x3, N=108 molecules, 1000 steps)
    ./fuller_LJ_core_serial

    # Large system (5x5x5, N=500 molecules)
    ./fuller_LJ_core_serial 5

  Fixed Parameters (modify in source code):
    Temperature T  = 300 K
    Pressure Pe    = 0.0 GPa
    Time step dt   = 1.0 fs
    Number of steps = 1000
    Output interval = 100 steps
    Neighbor list update = 25 steps

  Features:
    - NPT-MD simulation with LJ intermolecular interactions of C60 rigid molecules
    - Nose-Hoover thermostat + Parrinello-Rahman pressure control
    - Rigid-body rotation via quaternions
    - Velocity-Verlet time integration
    - Automatic generation of FCC crystal initial configuration
    - No restart capability, no OVITO output (core version)

  Parallelization Notes:
    This code runs in single-thread mode.
    Each computational loop is independent per molecule, enabling the following parallelization:

    * OpenMP:
      Add #pragma omp parallel for before each loop.
      Protect virial Wm9 accumulation with #pragma omp atomic update.
      Compile: g++ -std=c++17 -O3 -fopenmp ...

    * OpenACC GPU:
      Place arrays on GPU: #pragma acc data copy(pos[0:N*3], ...)
      Each loop: #pragma acc parallel loop present(...)
      Device functions: #pragma acc routine seq
      Virial: #pragma acc atomic update
      Compile: nvc++ -std=c++17 -O3 -acc -gpu=cc80 ...

    * GPU Optimization Strategy:
      forces(): gang-parallel (molecules) x vector-parallel (intra-molecular atoms ai)
      - Symmetric full list: no Newton's 3rd law -> no write conflicts
      - Vectorize ai atom loop: up to 128 threads for warp parallelism
      - Expand virial 9 components into scalar reduction variables
      Data residency: keep arrays on GPU with #pragma acc data

  Unit System: A (distance), amu (mass), eV (energy), fs (time), K (temperature), GPa (pressure)
===========================================================================*/

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <random>
#include <chrono>
#include <string>


/* ═══════════════ Physical Constants and Unit Conversion ═══════════════ */
/*  Conversion factor for using Newton's equation of motion F = m * a
    in the A/amu/eV/fs unit system.
    a [A/fs^2] = (F [eV/A]) / (m [amu]) * CONV                              */
constexpr double CONV       = 9.64853321e-3;   // eV*fs^2/(amu*A^2)
constexpr double kB         = 8.617333262e-5;   // Boltzmann constant [eV/K]
constexpr double eV2GPa     = 160.21766208;      // eV/A^3 → GPa
constexpr double eV2kcalmol = 23.06054783;       // eV → kcal/mol

/* ═══════════════ LJ Potential Parameters ═══════════════ */
/*  V(r) = 4 eps [ (sigma/r)^12 - (sigma/r)^6 ]
    RCUT: Cutoff distance (3*sigma ~ 10.29 A). Atom pairs with r > RCUT are skipped.
    VSHFT: Subtract V(RCUT) to avoid energy discontinuity at the cutoff.     */
constexpr double sigma_LJ = 3.431;              // C-C LJ sigma [A]
constexpr double eps_LJ   = 2.635e-3;           // C-C LJ epsilon [eV]
constexpr double RCUT      = 3.0*sigma_LJ;      // Cutoff distance ~10.29 A
constexpr double RCUT2     = RCUT*RCUT;
constexpr double sig2_LJ   = sigma_LJ*sigma_LJ;
constexpr double mC        = 12.011;             // Carbon atomic mass [amu]

/* VSHFT: constexpr -- sigma/RCUT = 1/3, (1/3)^6 = 1/729 */
constexpr double _sr_v  = 1.0/3.0;
constexpr double _sr2_v = _sr_v * _sr_v;
constexpr double _sr6_v = _sr2_v * _sr2_v * _sr2_v;
constexpr double VSHFT  = 4.0*eps_LJ*(_sr6_v*_sr6_v - _sr6_v);

/* ═══════════════ Molecular Parameters ═══════════════ */
constexpr int C60_NATOM = 60;                    // Number of atoms per molecule
constexpr int MAX_NATOM = 84;                    // Maximum atom count (for C84 support)
constexpr double MC60   = C60_NATOM * mC;        // C60 molecular mass [amu]
constexpr double RC60   = 3.55;                  // C60 radius [A]
constexpr double RMCUT   = RCUT + 2*RC60 + 1.0;  // Molecular pair cutoff ~18.4 A
constexpr double RMCUT2  = RMCUT*RMCUT;

/*  MAX_NEIGH: Maximum neighbor count for neighbor list
    Neighbor counts for FCC crystal (a0=14.17A):
      1st shell: 12 at a0/sqrt(2) = 10.0 A
      2nd shell:  6 at a0         = 14.2 A
      3rd shell: 24 at a0*sqrt(3/2) = 17.4 A
      Total: ~42 (same count since symmetric list)
    With 3.0A skin margin, 80 is safely sufficient.                          */
constexpr int MAX_NEIGH = 80;

/*  VECTOR_LENGTH: Number of vector threads per gang for OpenACC GPU version (reference)
    128 = 4 warps (NVIDIA GPU warp = 32 threads)
    C60 (natom=60): ceil(60/128)=1 iteration, C84 (natom=84): ceil(84/128)=1 iteration */
constexpr int VECTOR_LENGTH = 128;


/* ═══════════════ Flat 9-Component H-Matrix Operations ═══════════════ */
/*  h[0..8] = { H00,H01,H02, H10,H11,H12, H20,H21,H22 }
    row i, column j -> h[3*i+j]
    3x3 matrix representing the simulation cell shape.
    Volume V = |det(H)|, inverse Hi converts real coords -> fractional coords. */
#define H_(h,i,j) ((h)[3*(i)+(j)])

/* OpenACC: Add #pragma acc routine seq before functions to make them device functions */
static inline double mat_det9(const double* h){
    return H_(h,0,0)*(H_(h,1,1)*H_(h,2,2)-H_(h,1,2)*H_(h,2,1))
          -H_(h,0,1)*(H_(h,1,0)*H_(h,2,2)-H_(h,1,2)*H_(h,2,0))
          +H_(h,0,2)*(H_(h,1,0)*H_(h,2,1)-H_(h,1,1)*H_(h,2,0));
}

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


/* ═══════════════ Minimum Image Convention ═══════════════ */
/*  Compute the "shortest" distance vector between two particles (periodic boundary conditions).
    Round to [-0.5, 0.5) in fractional coordinates, then convert back to real coordinates.
    Hot function called in the innermost loop of forces().
    OpenACC: make it a device function with #pragma acc routine seq    */
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


/* ═══════════════ Quaternion Operations ═══════════════ */
/*  Represent 3D rotations with quaternion q = [w,x,y,z]. No gimbal lock; only normalization needed.
    - q2R_flat:      quaternion -> rotation matrix (used to compute real-space coords of intra-molecular atoms)
    - qmul_flat:     quaternion product (compose two rotations)
    - qnorm_flat:    normalize |q|=1 (correct accumulated numerical error)
    - omega2dq_flat: angular velocity omega * time dt -> small-rotation quaternion
    OpenACC: add #pragma acc routine seq before each function           */

static inline void q2R_flat(const double* q, double* R){
    double w=q[0],x=q[1],y=q[2],z=q[3];
    R[0]=1-2*(y*y+z*z); R[1]=2*(x*y-w*z);   R[2]=2*(x*z+w*y);
    R[3]=2*(x*y+w*z);   R[4]=1-2*(x*x+z*z); R[5]=2*(y*z-w*x);
    R[6]=2*(x*z-w*y);   R[7]=2*(y*z+w*x);   R[8]=1-2*(x*x+y*y);
}

static inline void qmul_flat(const double* a, const double* b, double* out){
    out[0]=a[0]*b[0]-a[1]*b[1]-a[2]*b[2]-a[3]*b[3];
    out[1]=a[0]*b[1]+a[1]*b[0]+a[2]*b[3]-a[3]*b[2];
    out[2]=a[0]*b[2]-a[1]*b[3]+a[2]*b[0]+a[3]*b[1];
    out[3]=a[0]*b[3]+a[1]*b[2]-a[2]*b[1]+a[3]*b[0];
}

static inline void qnorm_flat(double* q){
    double n=sqrt(q[0]*q[0]+q[1]*q[1]+q[2]*q[2]+q[3]*q[3]);
    double inv=1.0/n;
    q[0]*=inv; q[1]*=inv; q[2]*=inv; q[3]*=inv;
}

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


/* ═══════════════ C60 Coordinate Generation ═══════════════ */
/*  Generate 60 vertices in 3 groups using the golden ratio phi = (1+sqrt(5))/2:
      Group 1: cyclic permutations of (0, +/-1, +/-3*phi)   -> 12 vertices
      Group 2: cyclic permutations of (+/-2, +/-(1+2*phi), +/-phi) -> 24 vertices
      Group 3: cyclic permutations of (+/-1, +/-(2+phi), +/-2*phi) -> 24 vertices
    Scale by 0.72 to match actual C60 size (radius ~ 3.55 A).
    I0 = isotropic moment of inertia: used in rigid-body rotation equation tau = I * alpha. */
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


/* ═══════════════ FCC Crystal Generation ═══════════════ */
/*  FCC (face-centered cubic) is the stable structure of C60 crystals at room temperature.
    4 molecules per unit cell: (0,0,0), (a/2,a/2,0), (a/2,0,a/2), (0,a/2,a/2)
    nc x nc x nc lattice -> 4 * nc^3 molecules (nc=3 -> 108 molecules)       */
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


/* ═══════════════ Neighbor List Construction (Symmetric Full List) ═══════════════ */
/*  Symmetric list: stores both i->j and j->i.
    In forces(), each molecule accumulates only its own forces -> no write conflicts (ideal for GPU parallelization).
    nl_count[i]: number of neighbors of molecule i
    nl_list[i*MAX_NEIGH+k]: index of k-th neighbor of molecule i             */
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


/* ═══════════════ Apply PBC ═══════════════ */
/*  Wrap all molecules back into the cell (fractional coordinate s in [0,1)).
    Each molecule can be processed independently.
    OpenMP:  #pragma omp parallel for schedule(static)
    OpenACC: #pragma acc parallel loop present(pos,h,hi)                 */
static void apply_pbc(double* pos, const double* h, const double* hi, int N){
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


/* ═══════════════ Force / Torque / Virial Calculation (Main Kernel) ═══════════════ */
/*  * Heaviest computation accounting for >90% of total time -- top priority for parallelization

    Processing flow:
      1. Lab coordinate computation: quaternion -> rotation matrix -> real-space coords of intra-molecular atoms (relative to COM)
      2. Main force loop: compute LJ forces for 60x60 atom pairs for all neighbor pairs (i,j)

    Symmetric full list approach:
      Each molecule i accumulates only its own forces -> no write conflicts (ideal for GPU parallelization)
      No Newton's 3rd law -> 2x computation but fully parallelizable
      Multiply energy/virial by 0.5 to correct for double counting

    Parallelization guide:
      OpenMP:
        Lab computation loop:  #pragma omp parallel for schedule(static)
        Main force loop:       #pragma omp parallel for schedule(dynamic,1) reduction(+:Ep)
        Wm9 accumulation:      #pragma omp atomic update
      OpenACC:
        Lab computation loop:  #pragma acc parallel loop gang vector_length(128) present(...)
        Intra-atom loop:       #pragma acc loop vector
        Main force loop:       #pragma acc parallel loop gang vector_length(128) present(...) reduction(+:Ep)
        Atom pair loop:        #pragma acc loop vector reduction(+:fi0,...,w22)
        Wm9 accumulation:      #pragma acc atomic update                  */
static double forces(double* Fv, double* Tv, double* Wm9,
                     const double* pos, const double* qv,
                     const double* body, const double* h, const double* hi,
                     const int* nl_count, const int* nl_list,
                     int N, int natom, double rmcut2,
                     double* lab)
{
    /* --- Step 1: Lab coordinate computation (quaternion -> rotation matrix -> real-space coords of intra-molecular atoms) --- */
    for(int i=0;i<N;i++){
        double R[9];
        q2R_flat(&qv[i*4], R);
        for(int a=0;a<natom;a++){
            double bx=body[a*3], by=body[a*3+1], bz=body[a*3+2];
            int idx=i*natom*3+a*3;
            lab[idx  ]=R[0]*bx+R[1]*by+R[2]*bz;
            lab[idx+1]=R[3]*bx+R[4]*by+R[5]*bz;
            lab[idx+2]=R[6]*bx+R[7]*by+R[8]*bz;
        }
    }

    /* --- Zero initialization --- */
    for(int i=0;i<N*3;i++){ Fv[i]=0.0; Tv[i]=0.0; }
    for(int i=0;i<9;i++) Wm9[i]=0.0;

    /* --- Step 2: LJ force calculation main kernel --- */
    double Ep=0.0;

    for(int i=0;i<N;i++){
        double fi0=0, fi1=0, fi2=0;   /* Local force accumulation for molecule i */
        double ti0=0, ti1=0, ti2=0;   /* Local torque accumulation for molecule i */
        double my_Ep=0;
        /* Expand virial 9 components into scalars (no arrays, for GPU vector reduction compatibility) */
        double w00=0,w01=0,w02=0, w10=0,w11=0,w12=0, w20=0,w21=0,w22=0;

        int nni=nl_count[i];
        for(int k=0;k<nni;k++){
            int j=nl_list[i*MAX_NEIGH+k];
            /* Minimum image distance between molecules i and j */
            double dmx=pos[j*3]-pos[i*3];
            double dmy=pos[j*3+1]-pos[i*3+1];
            double dmz=pos[j*3+2]-pos[i*3+2];
            mimg_flat(dmx,dmy,dmz,hi,h);
            if(dmx*dmx+dmy*dmy+dmz*dmz > rmcut2) continue;

            /* 60x60 atom pair loop (sequential execution in single-thread mode) */
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
                        if(r2<0.25) r2=0.25;   /* Clamp to prevent NaN */
                        double ri2=1.0/r2;
                        double sr2=sig2_LJ*ri2;
                        double sr6=sr2*sr2*sr2;
                        double sr12=sr6*sr6;
                        double fm=24.0*eps_LJ*(2.0*sr12-sr6)*ri2;
                        double fx=fm*ddx, fy=fm*ddy, fz=fm*ddz;

                        /* Force on molecule i (j side is computed in j's loop) */
                        fi0-=fx; fi1-=fy; fi2-=fz;
                        /* Torque = -ra x F */
                        ti0-=(ray*fz-raz*fy);
                        ti1-=(raz*fx-rax*fz);
                        ti2-=(rax*fy-ray*fx);
                        /* Energy (half: symmetric list computes both (i,j) and (j,i)) */
                        my_Ep+=0.5*(4.0*eps_LJ*(sr12-sr6)-VSHFT);
                        /* Virial (half) */
                        w00+=0.5*ddx*fx; w01+=0.5*ddx*fy; w02+=0.5*ddx*fz;
                        w10+=0.5*ddy*fx; w11+=0.5*ddy*fy; w12+=0.5*ddy*fz;
                        w20+=0.5*ddz*fx; w21+=0.5*ddz*fy; w22+=0.5*ddz*fz;
                    }
                }
            }
        }

        /* Write force/torque for molecule i to global arrays (no conflicts: each i writes only to itself) */
        Fv[i*3]=fi0; Fv[i*3+1]=fi1; Fv[i*3+2]=fi2;
        Tv[i*3]=ti0; Tv[i*3+1]=ti1; Tv[i*3+2]=ti2;

        /* Aggregate virial contributions from all molecules.
           Protect with atomic update when parallelized:
             OpenMP:  #pragma omp atomic update
             OpenACC: #pragma acc atomic update                          */
        Wm9[0]+=w00; Wm9[1]+=w01; Wm9[2]+=w02;
        Wm9[3]+=w10; Wm9[4]+=w11; Wm9[5]+=w12;
        Wm9[6]+=w20; Wm9[7]+=w21; Wm9[8]+=w22;

        Ep+=my_Ep;
    }
    return Ep;
}


/* ═══════════════ Kinetic Energy ═══════════════ */
/*  Parallelization guide:
      OpenMP:  #pragma omp parallel for reduction(+:s)
      OpenACC: #pragma acc parallel loop present(vel) reduction(+:s)     */
static double ke_trans(const double* vel, int N, double Mmol){
    double s=0;
    for(int i=0;i<N;i++)
        s+=vel[i*3]*vel[i*3]+vel[i*3+1]*vel[i*3+1]+vel[i*3+2]*vel[i*3+2];
    return 0.5*Mmol*s/CONV;
}

static double ke_rot(const double* omg, int N, double I0){
    double s=0;
    for(int i=0;i<N;i++)
        s+=omg[i*3]*omg[i*3]+omg[i*3+1]*omg[i*3+1]+omg[i*3+2]*omg[i*3+2];
    return 0.5*I0*s/CONV;
}

static inline double inst_T(double KE, int Nf){return 2*KE/(Nf*kB);}
static inline double inst_P(const double* W, double KEt, double V){
    return (2*KEt+W[0]+W[4]+W[8])/(3*V)*eV2GPa;
}


/* ═══════════════ NPT State Variables ═══════════════ */
/*  Nose-Hoover thermostat + Parrinello-Rahman barostat:
      xi:    thermostat variable (friction coefficient for temperature control)
      Q:     thermostat mass (response speed for temperature control)
      Vg[9]: barostat velocity (rate of cell deformation)
      W:     barostat mass (response speed for pressure control)
      Pe:    target pressure [GPa], Tt: target temperature [K], Nf: degrees of freedom */
struct NPTState { double xi,Q,Vg[9],W,Pe,Tt; int Nf; };

static NPTState make_npt(double T, double Pe, int N){
    int Nf=6*N-3;  /* 3 translational + 3 rotational DOF - 3 COM constraints */
    NPTState s;
    s.xi=0; s.Q=std::max(Nf*kB*T*100.0*100.0,1e-20);
    for(int i=0;i<9;i++) s.Vg[i]=0;
    s.W=std::max((Nf+9)*kB*T*1000.0*1000.0,1e-20);
    s.Pe=Pe; s.Tt=T; s.Nf=Nf;
    return s;
}


/* ═══════════════ NPT Velocity-Verlet One Step ═══════════════ */
/*  Time evolution in the NPT (constant temperature, constant pressure) ensemble.
    Velocity-Verlet method: requires only one force evaluation per step.
      (A) Thermostat first half -> (B) Barostat first half -> (C) Velocity first half update
      -> (D) Position update -> (E) Cell update -> (F) Fractional -> real coords -> (G) Quaternion update
      -> (H) Force recomputation -> (I) Velocity second half update -> (J)(K) Thermo/baro second half

    Parallelization guide:
      Loops (C)(D)(F)(G)(I) are independent across all molecules:
        OpenMP:  #pragma omp parallel for schedule(static)
        OpenACC: #pragma acc parallel loop present(...)
      (E) H-matrix update is a host-side scalar operation (no parallelization needed)
      For OpenACC, #pragma acc update device(h[0:9]) is required after H-matrix update */
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

    /* (C) Velocity first half update */
    for(int i=0;i<N;i++){
        for(int a=0;a<3;a++){
            vel[i*3+a]=vel[i*3+a]*sc_v+hdt*Fv[i*3+a]*cF;
            omg[i*3+a]=omg[i*3+a]*sc_nh+hdt*Tv[i*3+a]*cT;
        }
    }

    /* (D) Position update (integrate in fractional coords + PBC) */
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

    /* (E) Cell H-matrix update */
    for(int a=0;a<3;a++) for(int b=0;b<3;b++) h[a*3+b]+=dt*npt.Vg[a*3+b];

    /* (F) Fractional coords -> real coords */
    for(int i=0;i<N;i++){
        double sx=pos[i*3],sy=pos[i*3+1],sz=pos[i*3+2];
        pos[i*3  ]=h[0]*sx+h[1]*sy+h[2]*sz;
        pos[i*3+1]=h[3]*sx+h[4]*sy+h[5]*sz;
        pos[i*3+2]=h[6]*sx+h[7]*sy+h[8]*sz;
    }

    /* (G) Quaternion update */
    for(int i=0;i<N;i++){
        double dq[4],tmp[4];
        omega2dq_flat(omg[i*3],omg[i*3+1],omg[i*3+2],dt,dq);
        qmul_flat(&qv[i*4],dq,tmp);
        qv[i*4]=tmp[0]; qv[i*4+1]=tmp[1]; qv[i*4+2]=tmp[2]; qv[i*4+3]=tmp[3];
        qnorm_flat(&qv[i*4]);
    }

    /* (H) Force recomputation */
    mat_inv9(h,hi);
    double Ep=forces(Fv,Tv,Wm9,pos,qv,body,h,hi,nl_count,nl_list,
                     N,natom,rmcut2,lab);

    /* (I) Velocity second half update */
    double eps_tr2=npt.Vg[0]*hi[0]+npt.Vg[4]*hi[4]+npt.Vg[8]*hi[8];
    double sc_v2=sc_nh*exp(-hdt*eps_tr2/3.0);
    for(int i=0;i<N;i++){
        for(int a=0;a<3;a++){
            vel[i*3+a]=(vel[i*3+a]+hdt*Fv[i*3+a]*cF)*sc_v2;
            omg[i*3+a]=(omg[i*3+a]+hdt*Tv[i*3+a]*cT)*sc_nh;
        }
    }

    /* (J)(K) Thermo/baro second half update */
    kt=ke_trans(vel,N,Mmol); kr=ke_rot(omg,N,I0); KE=kt+kr;
    npt.xi+=hdt*(2*KE-npt.Nf*kB*npt.Tt)/npt.Q;
    npt.xi=std::clamp(npt.xi,-0.1,0.1);
    double V2=fabs(mat_det9(h));
    dP=inst_P(Wm9,kt,V2)-npt.Pe;
    for(int a=0;a<3;a++) npt.Vg[a*4]+=hdt*V2*dP/(npt.W*eV2GPa);
    for(int a=0;a<3;a++) npt.Vg[a*4]=std::clamp(npt.Vg[a*4],-0.01,0.01);

    return {Ep,KE};
}


/* ═══════════════ Main Program ═══════════════ */
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

    /* --- Array allocation (raw arrays: compatible with OpenACC data regions) --- */
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
    printf("  C60 LJ NPT-MD Core (Serial)\n");
    printf("================================================================\n");
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
    /* Remove center-of-mass velocity */
    double vcm[3]={0,0,0};
    for(int i=0;i<N;i++) for(int a=0;a<3;a++) vcm[a]+=vel[i*3+a];
    for(int a=0;a<3;a++) vcm[a]/=N;
    for(int i=0;i<N;i++) for(int a=0;a<3;a++) vel[i*3+a]-=vcm[a];

    NPTState npt=make_npt(T,Pe,N);

    /* --- Initial neighbor list construction --- */
    nlist_build_sym(pos,h,hi,N,RMCUT,nl_count,nl_list);

    /* Apply PBC + initial force calculation */
    /*  OpenACC: before this, place arrays on GPU memory with
        #pragma acc data copy(...) copyin(...) create(...)
        and run the entire MD loop within the data region               */
    apply_pbc(pos,h,hi,N);
    forces(Fv,Tv,Wm9,pos,qv,body,h,hi,nl_count,nl_list,
           N,natom,RMCUT2,lab);

    double sT=0,sP=0,sa=0,sEp=0; int nav=0;
    auto t0=std::chrono::steady_clock::now();
    printf("%8s %7s %9s %8s %10s %7s\n",
           "step","T[K]","P[GPa]","a[A]","Ecoh[eV]","t[s]");

    /* === MD Main Loop === */
    for(int g=1;g<=nsteps;g++){

        /* Rebuild neighbor list */
        /*  OpenACC: before rebuild, transfer coordinates to host with
            #pragma acc update self(pos[0:N*3]), then after construction,
            transfer to device with
            #pragma acc update device(nl_count[0:N], nl_list[0:N*MAX_NEIGH]) */
        if(g%nlup==0){
            mat_inv9(h,hi);
            nlist_build_sym(pos,h,hi,N,RMCUT,nl_count,nl_list);
        }

        /* One step of time evolution */
        auto [Ep,KE]=step_npt(pos,vel,qv,omg,Fv,Tv,Wm9,
                               h,hi,body,c60.I0,c60.Mmol,
                               N,natom,RMCUT2,dt,npt,
                               nl_count,nl_list,lab);

        /* Instantaneous observables */
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

    /* --- Cleanup --- */
    delete[] pos; delete[] vel; delete[] omg; delete[] qv;
    delete[] Fv;  delete[] Tv;  delete[] lab; delete[] body;
    delete[] nl_count; delete[] nl_list;
    return 0;
}
