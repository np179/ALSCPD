# PJT2 for H2O
# If the atoms are labelled O, H1,H2, 
# then the following are the internuclear distances:
# O-H1=r1, O-H2=r2, H1-H2=r4
#

from numpy import *


def surfh2o(r1,r4,r2):
   
# bond angle using cosine theorem

      ctheta=(r1*r1 + r2*r2 - r4*r4)/(2.e0*r1*r2)
      ctheta=min(ctheta,1.e0)
      ctheta=max(ctheta,-1.e0)
      theta=arccos(ctheta)
      pot = PJT2(r1,r2,theta)
      return pot



############################################################################ 

def PJT2_2D(grd1, grd2, grd3):
    # make the function below compatible with the ALSCPD-MC Algorithm
    inp = [grd1, grd2, grd3]
    grdl = []
    grdind = []
    fix = 0
    for i, elem in enumerate(inp):
        if type(elem) == ndarray:
            grdl.append(elem)
            grdind.append(i)
        elif type(elem) == float:
            fix = i
    
    G1, G2 = meshgrid(grdl[0], grdl[1], indexing='ij')
    arg = [0,0,0]
    arg[grdind[0]] = G1
    arg[grdind[1]] = G2
    arg[fix] = inp[fix]
    return PJT2(*arg)


def PJT2(Q1,Q2,THETA):
#     Potential PJT2 due Polyansky, Jensen and Tennyson,
#     J. Chem. Phys., 105, 6490-6497 (1996)
#     Update of Polyansky, Jensen and Tennyson, J Chem Phys 101, 7651 (1994))
#     Units: Hartree and Bohr
#     RZ = OH equilibrium value
#     RHO = equilibrium value of pi - bond angle(THETA)

      TOANG = 0.5291772
      CMTOAU = 219474.624
      X1 = 1.0
      RHO1    =    75.50035308
      FA1     =    .00000000
      FA2     = 18902.44193433
      FA3     = 1893.99788146
      FA4     = 4096.73443772
      FA5     = -1959.60113289
      FA6     = 4484.15893388
      FA7     = 4044.55388819
      FA8     = -4771.45043545
      FA9     =    0.00000000
      FA10    =    0.00000000
      RZ      =    .95792059
      A       =   2.22600000
      F1A1    = -6152.40141181
      F2A1    = -2902.13912267
      F3A1    = -5732.68460689
      F4A1    = 953.88760833
      F11     = 42909.88869093
      F1A11   = -2767.19197173
      F2A11   = -3394.24705517
      F3A11   =    .00000000
      F13     = -1031.93055205
      F1A13   = 6023.83435258
      F2A13   =    .00000000
      F3A13   =    .00000000
      F111    =    .00000000
      F1A111  =  124.23529382
      F2A111  = -1282.50661226
      F113    = -1146.49109522
      F1A113  = 9884.41685141
      F2A113  = 3040.34021836
      F1111   = 2040.96745268
      FA1111  = .00000000
      F1113   = -422.03394198
      FA1113  = -7238.09979404
      F1133   =    .00000000
      FA1133  =    .00000000
      F11111  = -4969.24544932
      F111111 =  8108.49652354
      F71     = 90.00000000


      c1     = 50.0
      c2     = 10.0
      beta1  = 22.0
      beta2  = 13.5
      gammas = 0.05
      gammaa = 0.10
      delta  = 0.85
      rhh0   = 1.40

      RHO=RHO1*3.141592654/180.000000000
      RHO=RHO1*3.141592654/180.000000000

      FA11=0.0
      F1A3=F1A1
      F2A3=F2A1
      F3A3=F3A1
      F4A3=F4A1
      F33=F11
      F1A33=F1A11
      F2A33=F2A11
      F333=F111
      F1A333=F1A111
      F2A333=F2A111
      F133=F113
      F1A133=F1A113
      F2A133=F2A113
      F3333=F1111
      FA3333=FA1111
      F1333=F1113
      FA1333=FA1113
      F33333=F11111
      F333333 =F111111
      F73     =F71


#     Find value for DR and DS
      DR = TOANG*Q1 - RZ
      DS = TOANG*Q2 - RZ


#     Transform to Morse coordinates
      Y1 = X1 - exp(-A * DR)
      Y3 = X1 - exp(-A * DS)


#     transform to Jensens angular coordinate
      CORO = cos(THETA) + cos(RHO)

#     Now for the potential
      V0=(FA2+FA3*CORO+FA4*CORO**2+FA6*CORO**4+FA7*CORO**5)*CORO**2
      V0=V0+(FA8*CORO**6+FA5*CORO**3+FA9*CORO**7+FA10*CORO**8 )*CORO**2
      V0=V0+(                                    FA11*CORO**9 )*CORO**2
      FE1= F1A1*CORO+F2A1*CORO**2+F3A1*CORO**3+F4A1*CORO**4
      FE3= F1A3*CORO+F2A3*CORO**2+F3A3*CORO**3+F4A3*CORO**4
      FE11= F11+F1A11*CORO+F2A11*CORO**2
      FE33= F33+F1A33*CORO+F2A33*CORO**2
      FE13= F13+F1A13*CORO
      FE111= F111+F1A111*CORO+F2A111*CORO**2
      FE333= F333+F1A333*CORO+F2A333*CORO**2
      FE113= F113+F1A113*CORO+F2A113*CORO**2
      FE133= F133+F1A133*CORO+F2A133*CORO**2
      FE1111= F1111+FA1111*CORO
      FE3333= F3333+FA3333*CORO
      FE1113= F1113+FA1113*CORO
      FE1333= F1333+FA1333*CORO
      FE1133=       FA1133*CORO
      FE11111=F11111
      FE33333=F33333
      FE111111=F111111
      FE333333=F333333
      FE71    =F71
      FE73    =F73
      V   = V0 +  FE1*Y1+FE3*Y3 \
            +FE11*Y1**2+FE33*Y3**2+FE13*Y1*Y3 \
            +FE111*Y1**3+FE333*Y3**3+FE113*Y1**2*Y3 \
            +FE133*Y1*Y3**2 \
            +FE1111*Y1**4+FE3333*Y3**4+FE1113*Y1**3*Y3 \
            +FE1333*Y1*Y3**3+FE1133*Y1**2*Y3**2 \
            +FE11111*Y1**5+FE33333*Y3**5 \
            +FE111111*Y1**6+FE333333*Y3**6 \
            +FE71    *Y1**7+FE73    *Y3**7 


#     modification by Choi & Light, J. Chem. Phys., 97, 7031 (1992).
#      sqrt2=sqrt(2.0)
#      xmup1=sqrt2/3.0+0.5
#      xmum1=xmup1-x1
#      term=2.0*xmum1*xmup1*q1*q2*cos(theta)
#      r1=toang*sqrt((xmup1*q1)**2+(xmum1*q2)**2-term)
#      r2=toang*sqrt((xmum1*q1)**2+(xmup1*q2)**2-term)
#      rhh=sqrt(q1**2+q2**2-2.0*q1*q2*cos(theta))
#      rbig=(r1+r2)/sqrt2
#      rlit=(r1-r2)/sqrt2
#
#      alpha=(x1-tanh(gammas*rbig**2))*(x1-tanh(gammaa*rlit**2))
#      alpha1=beta1*alpha
#      alpha2=beta2*alpha
#      drhh=toang*(rhh-delta*rhh0)
#      DOLEG=     (1.4500-THETA)
#     IF (THETA.LE.0.64  ) V=0.1E17
#     IF((DR.LE.-0.4).AND.(THETA.LE.1.1)) V=0.1E17
#     IF((DS.LE.-0.4).AND.(THETA.LE.1.1)) V=0.1E17
#     IF (DS.LE. 0.0  ) V=0.1E17
#      v = v + c1*exp(-alpha1*drhh) + c2*exp(-alpha2*drhh)
 
#     Convert to Hartree
      V=V/CMTOAU
#      print(Q1,Q2,THETA,V)
      return V




#-----------------------------------------------------------------------
#  Morse function for OH potential curve. Parameters are taken from
#  JCP 113, 3150 (2000).
#-----------------------------------------------------------------------
def vho(r):
    
    De = 0.1628e0
    beta = 1.22e0
    roh = 1.85e0

    v = De * ( (1-exp(-beta*(r-roh)))**2 - 1.e0 )

    return v



