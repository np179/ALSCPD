#!/usr/bin/env python3
"""
   module dvr

   defines classes:

   DVR      : base class 
   hoDVR    : harmonic oscillator DVR
   sinDVR   : sine DVR

"""

import numpy 
from scipy.linalg import eigh_tridiagonal
from numpy.linalg import multi_dot
from scipy.special import factorial




def sqrt_factorial(n):
    # use a sqrt-version of factorial
    if n == 0 or n == 1:
        return 1.0
    
    sqrtf = 1.0
    for i in range(2,n+1):
        sqrtf = sqrtf*numpy.sqrt(i)
    return sqrtf
    





# base class that does nothing
# just make all derived classes
# subclasses of DVR 
class DVR:
    pass
    




class hoDVR(DVR):
    """
    Harmonic oscillotor DVR.

    hoDVR(N,xi=None,xf=None,mass=1.e0, freq=1.e0, eq=0.e0)

    Initialize either with a number of points and a grid range as 

        hoDVR(N,xi,xf)

    or with mass, frequency and equilibrium position as

        hoDVR(N,mass=m, freq=omega, eq=xeq)

    Default:

        hoDVR(N)

    will take mass=1, freq=1, eq=0

    *************************************************************
    *** tested to 150 points, or x approximately in [-16, 16] ***
    *************************************************************


    Members:

       hoDVR.grid:     The primitive basis points
       hoDVR.trafo:    Unitary transformation matrix between 
                       HO basis and primitve grid
       hoDVR.dif1:     d/dx matric in grid basis
       hoDVR.dif2:     d^2/dx^2 matric in grid basis
       hoDVR.weights:  sqrt(DVR weights)  

       hoDVR.hoxeq:    Equilibrium position
       hoDVR.homass:   Mass
       hoDVR.hofreq:   Frequency (omega)
       hoDVR.range:    Range of the grid 
       hoDVR.xi:       First grid point
       hoDVR.xf:       Last grid point
    """

    def __init__(self, N,xi=None,xf=None,mass=1.e0, freq=1.e0, eq=0.e0):

        inittype = None
        
        if all((xi is not None ,xf is not None,)): 
            inittype = 1 # xi, xf
        elif all((xi is None ,xf is None,)):
            inittype = 2 # mass, freq, eq. 
        else:
            message = "class hoDVR: Initialize either with mass, freguency, "
            message += "and optionally equilibrium position"
            message += "or with first and last grid point."
            raise RuntimeError(message)

        if N < 2:
            message = "class hoDVR: init with N < 2 makes no sense."
            raise RuntimeError(message)

        self.N = N

        # setup local variables
        if inittype == 1:
            self.hoxeq  = (numpy.double(xi)+numpy.double(xf))*0.5e0 # equilibrium position
            self.homass = 1.e0
            self.hofreq = 1.e0
            self.range=numpy.double(xf)-numpy.double(xi)
            self.xi = numpy.double(xi)
            self.xf = numpy.double(xf)
        
        else:
            self.hoxeq  = eq
            self.hofreq = freq
            self.homass = mass
            self.range  = 0.0e0

       
        # in the following:
        #   index j: HO-eigenstates
        #   index g: grid-points
  
        # setup the x matrix in HO basis 
        diagonal = numpy.zeros(N, dtype=numpy.double)
        subdiagonal =  numpy.zeros(N-1, dtype=numpy.double)

        hofm =  self.homass*self.hofreq
        
        # subdiagonal: x = sqrt(\hbar/ 2*m*omega) * ( a^+ + a)
        for j in range(1,N):
            subdiagonal[j-1] = numpy.sqrt(numpy.double(j)/hofm/2)
            
        self.grid, self.trafo = eigh_tridiagonal(diagonal, subdiagonal)
          
        

        # calculate mass-frequency scaling from grid
        # if xi-xf is given
        if inittype == 1:
            hofm = self.range/(self.grid[-1] - self.grid[0])
            # we only know the product, so just set arbitrarily:
            self.grid *= hofm
            hofm = 1.e0/hofm**2
            self.homass = numpy.sqrt(hofm)
            self.hofreq = numpy.sqrt(hofm)
        else:
            self.xi = self.grid[0] + self.hoxeq
            self.xf = self.grid[-1] + self.hoxeq
            self.range =  self.xf -  self.xi
            # mass and frequency scaling
            hofm =  self.homass*self.hofreq

        

        # DVR eigenvectors in the columns of self.trafo
        # are only defined up to a sign. The rows
        # contain the basis functions on the grid. So to make
        # sure they don't have a random sign, we test the sign
        # against the analytical solution. This is a
        # bit of a quick and dirty check as the
        # following is not very stable, but will
        # work for about up to 150 points. 

        # create the first 0 to N-1 Hermite polynomials with recursion
        # using mass and frequendy scaled coordinates
        X = (self.grid[:])*numpy.sqrt(hofm)
        
        hermite = numpy.zeros((N,N),dtype=numpy.double)
        hermite[0,:] = 1.0
        hermite[1,:] = X*2
        for n in range(2,self.N):
            hermite[n,:] = 2.0*X*hermite[n-1,:] - 2.0*(n-1)*hermite[n-2,:]

        # the recursion is not too stable, but we only do this for 
        # the weights and to detect the sign.
        # For this it is good enough. The instabilities
        # occor on the edges for large abs(x) because of high powers.
        # but we will multiply with exp(-x**2/2) in the next step
        # which makes the error small again
     

        # normalization and augmenting with gaussian
        # this will fail for some large N because the factorial gets huge

        
        for n in range(self.N):
            #hermite[n,:] /= numpy.sqrt( 2**n * factorial(n) * numpy.sqrt(numpy.pi/hofm))
            hermite[n,:] /= numpy.sqrt( 2**n * numpy.sqrt(numpy.pi/hofm))*sqrt_factorial(n)
            hermite[n,:] *= numpy.exp(-X**2/2.0)



        
        for g in range(N):
            if numpy.sign(self.trafo[self.N-1,g]) != numpy.sign(hermite[self.N-1,g]):
                self.trafo[:,g] *= -1.0



        # create matrices containing derivatives:
        # start with second derivative. we use a
        # trick here: we know the matrix emenents of
        # the Hamiltonian H in its eigenbasis, it is
        # just harmonic oscillator energies on the diagonal.
        # and the energies we know analytically as
        # E_j = \hbar omega (j+1/2)
        # we also know H = p^2/2m + V and
        # p^2 = - d^2/dq^2. (setting \hbar to one) 
        # multiplying H with -2m we get
        # -2m H =  d^2/dq^2 - 2mV
        # We can set up -2m H in the eigenbasis,
        # then transform to the grid basis where
        # we know V and subtract -2mV

        H = numpy.zeros((N,N),dtype=numpy.double)
        for j in range(N):
            H[j,j] = -(j+0.5e0)*hofm*2.e0 # = -2 m omega (j+1/2) = -2m E_j 


        # transform to grid basis
        self.dif2 = self.trafo.T @ H @ self.trafo
        #---->matmul------->-----^


        # subtract potential
        for g in range(N):
            # V = 1/2 m omega^2 q^2 --> -2 m V =  -m^2 omega^2 q^2 
            self.dif2[g,g] = self.dif2[g,g] + (hofm*self.grid[g])**2

         
        # The first derivative we can construct from -i times
        # the momentum operator expressed in creation
        # and anihilation operators
        # -i*p = sqrt(m omega/2)( a^+ - a) = d/dx
        # which is an antisymmetric tridiagonal matrix
        # in the eigen-basis
        
        momentum = numpy.zeros((N,N),dtype=numpy.double)
        for j in range(N-1):
            momentum[j,j+1] =  numpy.sqrt(hofm*(j+1)/2.e0)
            momentum[j+1,j] = -numpy.sqrt(hofm*(j+1)/2.e0)

        # transform to grid representation
        self.dif1 = self.trafo.T @ momentum @ self.trafo
        

        # calculate the DVR weights
        # (not to be confused with the quadrature weights)

        self.weights = numpy.zeros(N,dtype=numpy.double)

        # usually one would be able to do
        # 1/w_alpha = U[j,alpha]/phi(x_alpha)
        # but this is instable because phi(x_alpha)
        # might be very small or zero
        # we constract the chi_alpha(x_alpha)
        # from the hermite polynomial used above
        # The weights may be a bit off at the edges of
        # the grid but if the system is modeled
        # properly the wavefunction should be very small there

        # compute the weights. The value of the
        # eigenfunction chi at x_g
        # (one could of course just use the scipy Hermite-weights 
        # but the purpose is also to show how it is done manually)
        for g in range(self.N):
            # the nth eigenvector os in nth column 
            w = numpy.dot(self.trafo[:,g],hermite[:,g])
            if w < 0:
                # make sure it is positive
                w = -w
                
            # not squared as in legendre quadrature because
            # we will equally distribute the weights on the bra and
            # ket vectors
            self.weights[g] = 1/w  # this is really sqrt(w_alpha)

            # this is a debug check with U[j,alpha]/phi_j(x_alpha), j=0 for small grids:
            #print(self.weights[g]- self.trafo[0,g]*numpy.exp(X[g]**2/2.0)*(hofm/numpy.pi)**(-0.25))


        # finally we shift to the equilibrium position
        self.grid += self.hoxeq
        



    def applyweights(self, psi):
        "multipy the dvr weight into a wavefunction"

        try: 
            assert (len(psi) == len(self.weights))
        except:
            message = "hoDVR: psi seems to belond to a different DVR:"
            message += " self.N = {}, len(psi) = {}".format(self.N, len(psi))
            raise RuntimeError(message)

        return psi[:]*self.weights[:]



    def removeweights(self,psi):
        "divide psi by the dvr weights"

        try: 
            assert (len(psi) == len(self.weights))
        except:
            message = "hoDVR: psi seems to belond to a different DVR:"
            message += " self.N = {}, len(psi) = {}".format(self.N, len(psi))
            raise RuntimeError(message)

        return psi[:]/self.weights[:]
        

        

    def __str__(self):

        message = """
        Harmonic oscillator DVR
          N       = {: d}
          mass    = {: .8f}
          omega   = {: .8f}
          center  = {: .8f}
          xi      = {: .8f}
          xf      = {: .8f}
          """.format(self.N, self.homass, self.hofreq, self.hoxeq, self.xi, self.xf)

        return message












class sinDVR(DVR):
    """
    Sine DVR (Particle in a box DVR)
    
    sinDVR(N,xi=None,xf=None,L=pi,eq=pi/2):

    Initialize with a number of points and a grid range as 

        sinDVR(N,xi,xf)

    where N number of points, xi, xf first and last point or

        sinDVR(N,L=L,eq=ew)

    with N number of points, L length of the interval and eq.
    center of the interval.


    Note that xi and xf do not coincide with the beginning of
    and end of the interval (the wavefunction is zero there
    enyway). 


    Members:

       sinDVR.grid:     The primitive basis points
       sinDVR.trafo:    Unitary transformation matrix between 
                        HO basis and primitve grid
       sinDVR.dif1:     d/dx matric in grid basis
       sinDVR.dif2:     d^2/dx^2 matrix in grid basis
       sinDVR.weights:  sqrt(DVR weights)

       sinDVR.L:        Interval length
       sinDVR.xi:       First grid point
       sinDVR.xf:       Last grid point

    """


    def __init__(self,N,xi=None,xf=None,L=numpy.pi,eq=numpy.pi/2):
    
        
        inittype = None
        
        if all((xi is not None ,xf is not None,)): 
            inittype = 1 # xi, xf
        elif all((xi is None ,xf is None,)):
            inittype = 2 # L, eq
        else:
            message = "class sinDVR: Initialize either with "
            message += "length of interval and center"
            message += "or with first and last grid point."
            raise RuntimeError(message)

        if N < 2:
            message = "class sinDVR: init with N < 2 makes no sense."
            raise RuntimeError(message)
        
        self.N = N
    

        self.grid = numpy.zeros(N)
        
        if inittype == 1:
            dx=(xf-xi)/(N-1)
            self.xi = xi
            self.xf = xf
            self.L = dx*(N+1)
            
        else:
            dx = L/(N+1)
            self.xi = eq - L/2 + dx
            self.xf = eq + L/2 - dx
            self.L = L
            
        for g in range(N):
            self.grid[g] = self.xi + g*dx 

            
        # some quantities we need a few times
        x = N+1
        fac1 = numpy.pi/x
        deltax = (self.xf - self.xi)/(N-1)


        
        # --- DVR/FBR transformation matrix is analytically given
        self.trafo = numpy.zeros((N,N),dtype=numpy.double)
        
        fac1 = numpy.pi/(N+1)
        for g1 in range(N):
            for g2 in range(N):
                self.trafo[g2,g1]= numpy.sqrt(2.e0/x)*numpy.sin((g2+1)*(g1+1)*fac1)

        
        # --- Second derivative in Sine-DVR basis (given analytically)
        self.dif2 = numpy.zeros((N,N),dtype=numpy.double)
           
        for g1 in range(N):
            for g2 in range(g1):
                self.dif2[g2,g1]=-(numpy.pi/deltax)**2 \
                                  *2.e0*(-1.e0)**(g2-g1)/x**2 \
                                  *numpy.sin((g2+1)*fac1)*numpy.sin((g1+1)*fac1) \
                                  /(numpy.cos((g2+1)*fac1)-numpy.cos((g1+1)*fac1))**2 
                self.dif2[g1,g2]=self.dif2[g2,g1]

    
            self.dif2[g1,g1]=-(numpy.pi/deltax)**2 \
                              *(1.e0/3.e0+1.e0/(6.e0*x**2) \
                                -1.e0/(2.e0*x**2*numpy.sin((g1+1)*fac1)**2))


        #-----------------------------------------------------------------------
        # Derivative in FBR basis is analytically given for SIN-DVR
        # but is not tridiagonal:
        #   i)  all diagonal elements and all elements for which the row
        #       and the column index are simultaneously odd or even
        #       are zero
        #   ii) for all other elements it holds:
        #
        #           d_ab = 4ab/(a**2-b**2) 1/(N+1) 1/deltax
        #
        #   iii) from ii) it follows that the matrix is still antisymmetric
        #-----------------------------------------------------------------------

        tmp = numpy.zeros((N,N),dtype=numpy.double)
        deltax = (self.xf-self.xi)/(self.N-1)
        fac1 = 4.e0/((N+1)*deltax)
        for g1 in range(1,N+1):
            for g2 in range(1,g1):
                if (g2+g1) % 2 == 0:   ## % modulo
                    tmp[g2-1,g1-1] = 0.e0
                else:
                    tmp[g2-1,g1-1] = fac1*g2*g1/(g2**2-g1**2)
               
                tmp[g1-1,g2-1] = -tmp[g2-1,g1-1]
            
            tmp[g1-1,g1-1] = 0.e0

            
        self.dif1 = self.trafo.T @ tmp @ self.trafo


        self.weights = numpy.ones(self.N)*numpy.sqrt(self.grid[1]-self.grid[0])


    def __str__(self):
        
        message = """
        sine DVR
          N       = {: d}
          xi      = {: .8f}
          xf      = {: .8f}
          L       = {: .8f}
        """.format(self.N,  self.xi, self.xf, self.L)

        return message
