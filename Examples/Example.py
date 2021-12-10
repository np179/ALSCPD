import ALS.ALSclass as ALS
import ALS.dvr as dvr
import ALS.h2o as h2o
import numpy as np



if __name__ == "__main__":

    # set up a DVR-Basis to calculate a 3D H2O-Potential as
    # example tensor

    # set up the three coordinates
    N1h2 = 20
    xi1h2 = 1.0
    xf1h2 = 3.475
    r1h2 = dvr.sinDVR(N1h2, xi=xi1h2, xf=xf1h2)

    N2h2 = 20
    xi2h2 = 1.0
    xf2h2 = 3.475
    r2h2 = dvr.sinDVR(N2h2, xi=xi2h2, xf=xf2h2)

    Nuh2 = 20
    xiuh2 = -0.95
    xfuh2 = 0.6
    uh2 = dvr.sinDVR(Nuh2, xi=xiuh2, xf=xfuh2)
    # angular transformation required by this spescific potential
    thetah2 = np.arccos(uh2.grid)

    R1h2, R2h2, Thetah2 = np.meshgrid(r1h2.grid, r2h2.grid, thetah2, indexing="ij")
    # create the example tensor
    V = h2o.PJT2(R1h2, R2h2, Thetah2)
    # set the CPD expansion rank
    rank = 5
    
    # create an object
    ALSobject = ALS.ALSCPD("Testobject", V, rank=rank)

    # maximum amount of iterations:
    max_it = 1000
    # threshhold to signal convergence given in [cm-1]
    thresh = 100

    # iterate with 1D Method
    ALSobject.run(max_it, thresh)

    # iterate further with 2D Method
    ALSobject.run2D(max_it, thresh)

    # iterate further with 2D Method using singular value decomposition
    # maybe faster than regular 2D for large tensors
    ALSobject.run2D(max_it, thresh, YSVD=True)

    # get the weights and single particle potentials after iterations
    weights, SPP = ALSobject.get_current()

    # plot the error for the iterations
    ALSobject.plot_error()


    # set up Monte-Carlo
    # list of DVR grids
    gridlist = [r1h2.grid, r2h2.grid, thetah2]

    # function for 1D cuts
    func1 = h2o.PJT2
    # function for 2D cuts
    func2 = h2o.PJT2_2D

    # amount of sampling points, uniform sampling over coordinate grids
    nsmpl = 1000
    
    # create Object ready for Monte-Carlo runs
    ALSMCobject = ALS.ALSCPD("TestobjectMC", V, rank=rank, func1D=func1,\
         func2D=func2, grids=gridlist, nsmpl=nsmpl)

    # iterate with 1D Monte-Carlo Method
    # this method is poorly suited for the example tensor and will
    # therefore behave unstable
    ALSMCobject.runMC(max_it, thresh)

    # iterate with 2D Monte-Carlo
    ALSMCobject.run2DMC(max_it, thresh)

    # plot the error for the iterations
    ALSMCobject.plot_error()

