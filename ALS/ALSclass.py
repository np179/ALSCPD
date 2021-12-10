from ALS.ALS1D import *
from ALS.ALS2D import *
from ALS.twoDsub import *
from ALS.MonteC import *
from ALS.tracker import *
import matplotlib.pyplot as plt
import copy as cp
import numpy as np

# define the class object

# factor to go from au to cm-1
au2ic = 219474.63137

class ALSCPD:
    '''
    Define an object to perform ALS CPD. Initialize with filename, given exact tensor and initial rank.
    
    ******************************************************************************************************
    Either initialize with:
    [ALSCPD] = __init__(self.filename, self.v_ex, self.rank)
    
    OR if MonteCarlo will be used initialize with:
    [ALSCPD] = __init__(self.filename, self.v_ex, self.rank, self.func1D, self.grids, self.nsmpl, self.presmpl)
    ******************************************************************************************************
   
    ******************************************************************************************************
    [Attributes]:
    
            self.filename[str]: String which acts as a filename for the created output-file.     
            
            self.rank[int]: Current rank of the CPD expansion.
            self.iter[int]: Current amount of iterations passed.
            
            self.v_ex[array]: The exact tensor to be decomposed.
            self.weights[array]: Array containing the CPD weights.
            
            self.Nlist[list]: List containing the shape of the exact tensor.
            self.nu_list_init[list]: List containing the initial, random SPP.
            self.dyn_nu[list]: List containing the SPP which are updated during iterations.
            self.sigmas[list]: List of the sigmas needed to build the single-hole overlap matrix.
            self.errorl[list]: List of the RMSE for the ALS functional over the iterations. [cm-1]
            
       !For MonteCarlo:
            
            self.presmpl[str]: In case the sampling points are supposed to be read from an existing file
                        the filename can be specified here.
            
            self.nsmpl[int]: Amount of sampling points. Currently only uniform sampling. Changing of 
                        amount of sampling points currently not implemented.
                       
            self.func1D1D[function]: Callable to generate the 1D cuts through the potential. Should operate
                        on the individual grids to avoid errors.
            self.func1D2D[function]: Callable to generate the 2D cuts through the potential. Should operate
                        on the individual grids to avoid errors.
                        
            self.smpl_idx[array]: Array containing the sampling points in index representation.
            self.nu_smpl[array]: Array containing the sampled SPP.
            
            self.grids[list]: List containing the individual grids for the cuts.
            
       !Get initialized when running 1D or 2D MCALSCPD for the first time on the object:
       
            self.cuts1D[list]: List containing the 1D cuts for all DOF.
            self.cuts2D[list]: List containing the 2D cuts for all combinations.
       
   *******************************************************************************************************
   
   *******************************************************************************************************
   [Build-In's]:
           
           ---------------------------------------------------------------------------------------------- 
           self.copy_reset(other):
           
               Function to copy a ALSCPD object in its initial state to run from same initial guess.
               
               [Args]:
                       other[ALSCPD]: Object to get initial conditions from.
                       
               [Changes]:
                   
                   All attributes are changed to mimick the initial other ALSCPD object.
           ----------------------------------------------------------------------------------------------                
               
           ----------------------------------------------------------------------------------------------            
           self.copy(other):
           
               Function to copy another ALSCPD object.
               
               [Args]:
                       other[ALSCPD]: Object to be copied.
                       
               [Changes]:
                   
                   All attributes (except filename) are changed to those of the other object.
           ----------------------------------------------------------------------------------------------           
           
           ----------------------------------------------------------------------------------------------             
           self.run(max_iter, thresh, prec=None, dyn=False, tracker=True)
               
               Function to iterate through the algorithm while the RMSE is above threshold.
               
               [Args]:
                       max_iter[int]: Maximum amount of iterations to run if convergence isn't reached.
                       thresh[float]: Threshold to reach for convergence.
                       prec[float]: Gives the epsilon for the regularization, standard is root of machine 
                                      precision for float (~1E-8).
                       dyn[bool]: If True the rank of expansion will be increased by one if the error changes
                                      less then 1E-2 in two consecutive iterations. Default = True.
                       tracker[bool]: Set if the progress tracker should be displayed, default is True.
                       
               [Changes]:
                   
                   self.iter
                   self.dyn_nu
                   self.weights
                   self.sigmas
                   self.errorl
           ---------------------------------------------------------------------------------------------- 
                      
           ----------------------------------------------------------------------------------------------             
            self.run2D(max_iter, thresh, prec=None, dyn=False, YSVD=False, BSVD=False, tracker=True)
                
                Function to iterate through the 2D algorithm while the RMSE is above threshold.
                
                [Args]:
                       max_iter[int]: Maximum amount of iterations to run if convergence isn't reached.
                       thresh[float]: Threshold to reach for convergence.
                       prec[float]: Gives the epsilon for the regularization, standard is root of machine 
                                      precision for float (~1E-8).
                       dyn[bool]: If True the rank of expansion will be increased by one if the error changes
                                      less then 1E-2 in two consecutive iterations. Default = True.
                               
                               !!! SVD leads to unstable behaviour if dynamic rank expansion is enabled.
                                   Even though the program should automatically determine how many singular
                                   values and vectors are necessary keep in mind that the error might
                                   increase in some iterations. In this case SVD might be disabled.!!!
                                  
                       YSVD[bool]: Build the Y for the subALS LES from the SVD of x_ij. Default False. 

                       BSVD[bool]: Build the B for the subALS LES from the SVD of x_ij. Default False.
                       tracker[bool]: Set if the progress tracker should be displayed, default is True.                                   
                                   
               [Changes]:
                   
                   self.iter
                   self.dyn_nu
                   self.weights
                   self.sigmas
                   self.errorl
           ---------------------------------------------------------------------------------------------- 
           
           ----------------------------------------------------------------------------------------------                    
            self.runMC(max_iter, thresh, prec=None, tracker=True):
    
                Function to run the 1D ALSCPD-MC Algorithm.
    
                [Args]:
                       max_iter[int]: Maximum amount of iterations to run.
                       thresh[float]: Maximum error to signal convergence.
                       prec[float]: Gives the epsilon for the regularization, standard is root of machine 
                                      precision for float (~1E-8).  
                       tracker[bool]: Set if the progress tracker should be displayed, default is True.                                      
                [Changes]:
                       
                    self.iter
                    self.dyn_nu
                    self.weights
                    self.nu_smpl
                    self.errorl
                    
                If not initialized already, initializes:
                    
                    self.cuts2D
                    self.smpl_idx
                    self.nu_smpl                    
           ---------------------------------------------------------------------------------------------- 
           
           ----------------------------------------------------------------------------------------------
           self.run2DMC(max_iter, thresh, prec=None, tracker=True)
              
              Run the 2DMC-ALSCPD Algorithm.
    
              [Args]:
                      max_iter[int]: Maximum amount of iterations.
                      thresh[float]: Threshhold to signal convergence.
                      prec[float]: Value for the regularization, default is ~1E-8.
                      tracker[bool]: Set if the progress tracker should be displayed, default is True.
                       
              [Changes]:
                       
                    self.iter
                    self.dyn_nu
                    self.weights
                    self.nu_smpl
                    self.errorl

              If not initialized already, initializes:
                    
                    self.cuts2D
                    self.smpl_idx
                    self.nu_smpl
           ----------------------------------------------------------------------------------------------         
                    
           ----------------------------------------------------------------------------------------------                    
            self.plot_error(marker='')
            
                Function to plot the RMSE of the ALS functional over the iterations, first few elements
                are neglected as they are usually needlessly large and essentially meaningless.
                
                [Args]:
                        marker[str]: Set a marker for the indivual point along the curve. Default is
                                    no markers.
                        show[bool]:  Sets if the plot is to be shown, default is True.
           ---------------------------------------------------------------------------------------------- 
            
           ----------------------------------------------------------------------------------------------           
            self.get_current()
               
               [Returns]:
                       [array]: self.weights
                       [list]: self.dyn_nu
           ---------------------------------------------------------------------------------------------- 
           
           ----------------------------------------------------------------------------------------------                        
            self.change_rank(new_rank):
            
                Function to increase the CPD rank.
                
                [Args]:
                        new_rank[int]: New rank of the CPD.
                        
                [Changes]:
                    
                    self.rank
                    self.weights
                    self.dyn_nu
                    self.sigmas
           ---------------------------------------------------------------------------------------------- 
           
           ----------------------------------------------------------------------------------------------             
            self.get_perc()
            
                Function to get the current percentage of points used in the expansion compared to the 
                full tensor.
                
                [Returns]:
                        [float]: Current percentage of points used in expansion compared to full tensor.
           ----------------------------------------------------------------------------------------------                         
   *******************************************************************************************************
            '''
    
    
    
    def __init__(self, filename, v_ex, rank, func1D=None, func2D=None, grids=None, nsmpl=None, presmpl=None):
        # the init looks like a mess atm maybe clean this up later
        
        # set filename for current job
        self.filename = filename+".als"
        # store the exact tensor of the object
        self.v_ex = v_ex
        # store the rank of the expansion, maybe update later?
        self.rank = rank
        # get the number of grid points for all dimensions
        self.Nlist = []
        for dim in range(np.ndim(v_ex)):
            self.Nlist.append(v_ex.shape[dim])
        
        # initialized the ALS with the given rank and the list of point numbers
        self.weights, self.nu_list_init, self.sigmas = initALS(self.rank, self.Nlist)
        
        # get copys of the initial SPP, the initial weigths can always be created
        # via np.zeros(self.rank)
        self.dyn_nu = cp.deepcopy(self.nu_list_init)
        
        # store the number of iterations
        self.iter = 0
        # store the error in a list
        error1 = geterrorleft(self.v_ex, self.weights, self.dyn_nu)
        error2 = geterrorright(self.v_ex, self.weights, prec=None)
        totalerror = get_rmse(error1,error2)
        
        with open('{}'.format(self.filename), 'w') as file:
            file.write('! Iteration RMSEleft RMSEright RMSEtot \n')
            file.write('{} {} {} {} \n'\
                       .format(self.iter, np.sqrt(error1)*au2ic, np.sqrt(error2)*au2ic, totalerror))
        
        self.errorl = [totalerror]
        #create the save file
        
        # initialize the Object for Monte-Carlo, dont decide on 1D or 2D here, just initialize the
        # cuts and everything once when the 'run'-Routines are called
        if func1D != None or func2D != None and grids != None and nsmpl != None:
            if func1D != None:
                self.func1D = func1D
            if func2D != None:
                self.func2D = func2D
            self.grids = grids
            
            
            if presmpl == None:
                self.nsmpl = nsmpl     
                #self.smpl_idx, self.cuts1D, self.nu_smpl = setup_MC(self.grids, self.nsmpl, self.dyn_nu, self.func1D)
                self.smpl_idx = None
                self.cuts1D = None
                self.cuts2D = None
                self.nu_smpl = None
                self.combl = None
                
            elif presmpl != None:
                self.smpl_idx = grab_smpl(presmpl)
                self.nsmpl = self.smpl_idx.shape[0]
                #truesmpl = get_true_points(self.grids, self.smpl_idx)
                #self.cuts1D = get_all_cuts(self.func1D, self.grids, truesmpl)
                self.cuts1D = None
                self.cuts2D = None
                self.nu_smpl = get_all_nu_smpl(self.dyn_nu, self.smpl_idx)
                
        elif func1D == None and func2D == None and grids == None and nsmpl == None:
            self.func1D = None
            self.func2D = None
            self.grids = None
            self.nsmpl = None
                 
        else:
            raise RuntimeError('Object could not be initialized properly.')
            
        
    def __str__(self):
        '''Get properties of the object'''
        return """
        Object filename: {}.
        Current loaded tensor has shape: {}.
        Current rank of expansion: {}.
        Current amount of iterations: {}.
        Current error: {:.2f}cm-1.
        Current expansion uses {:.2f}% of storage compared to full.
        """.format(self.filename, self.v_ex.shape, self.rank, self.iter, self.errorl[self.iter], self.get_perc())
    
    
    def copy_reset(self, other):
        '''self.copy_reset(other):
        
               Function to copy a ALSCPD object in its initial state to run from same initial guess.
               
               [Args]:
                       other[ALSCPD]: Object to be get initial conditions from.
                       
               [Changes]:
                   
                   All attributes (except filename) are changed to mimick the initial other ALSCPD object.'''
        
        #self.filename = other.filename
        self.v_ex = other.v_ex
        self.Nlist = other.Nlist
        self.rank = other.nu_list_init[0].shape[0]
        self.weights = np.zeros(self.rank)
        self.nu_list_init = other.nu_list_init
        self.dyn_nu = cp.deepcopy(self.nu_list_init)
        self.sigmas = get_sigmas(self.dyn_nu)
        self.iter = 0
        
        if type(other.nsmpl) != int and type(other.smpl_idx) != np.array:
            self.func1D = other.func1D
            self.func2D = other.func2D
            self.grids = other.grids
            self.nsmpl = other.nsmpl
            self.smpl_idx = other.smpl_idx
            self.cuts1D = other.cuts1D
            self.cuts2D = other.cuts2D
            self.nu_smpl = get_all_nu_smpl(self.nu_list_init, self.smpl_idx)
        
        
        error1 = geterrorleft(self.v_ex, self.weights, self.dyn_nu)
        error2 = geterrorright(self.v_ex, self.weights, prec=None)
        totalerror = get_rmse(error1,error2)
        
        with open('{}'.format(self.filename), 'w') as file:
            file.write('! Iteration RMSEleft RMSEright RMSEtot \n')
            file.write('{} {} {} {} \n'\
                       .format(self.iter, np.sqrt(error1)*au2ic, np.sqrt(error2)*au2ic, totalerror))
        
        self.errorl = [totalerror]
        
        
    def copy(self, other):
        '''self.copy(other):
        
               Function to copy another ALSCPD object.
               
               [Args]:
                       other[ALSCPD]: Object to be copied.
                       
               [Changes]:
                   
                   All attributes are changed to those of the other object.'''
        
        #self.filename = other.filename+'CP'
        self.v_ex = cp.deepcopy(other.v_ex)
        self.rank = cp.deepcopy(other.rank)
        self.Nlist = cp.deepcopy(other.Nlist)
        self.weights = cp.deepcopy(other.weights)
        self.nu_list_init = cp.deepcopy(other.nu_list_init)
        self.sigmas = cp.deepcopy(other.sigmas)
        self.dyn_nu = cp.deepcopy(other.dyn_nu)
        self.iter = cp.deepcopy(other.iter)
        self.errorl = cp.deepcopy(other.errorl)

        if other.nsmpl != None:
            self.func1D = other.func1D
            self.func2D = other.func2D
            self.grids = other.grids
            self.nsmpl = other.nsmpl
            self.smpl_idx = other.smpl_idx
            self.cuts1D = other.cuts1D
            self.cuts2D = other.cuts2D
            self.nu_smpl = get_all_nu_smpl(self.nu_list_init, self.smpl_idx)
        
        
        error1 = geterrorleft(self.v_ex, self.weights, self.dyn_nu)
        error2 = geterrorright(self.v_ex, self.weights, prec=None)
        totalerror = get_rmse(error1,error2)
        
        with open('{}'.format(self.filename), 'w') as file:
            file.write('! Iteration RMSEleft RMSEright RMSEtot \n')
            file.write('{} {} {} {} \n'\
                       .format(self.iter, np.sqrt(error1)*au2ic, np.sqrt(error2)*au2ic, totalerror))
        
        
    def run(self, max_iter, thresh, prec=None, dyn=False, tracker=True):
        '''self.run(max_iter, thresh, prec=None)
               
               Function to iterate through the algorithm while the RMSE is above threshold.
               
               [Args]:
                       max_iter[int]: Maximum amount of iterations to run if convergence isn't reached.
                       thresh[float]: Threshold to reach for convergence.
                       prec[float]: Gives the epsilon for the regularization, standard is root of machine 
                                      precision for float (~1E-8).
                       dyn[bool]: If True the rank of expansion will be increased by one if the error changes
                                      less then 1E-2 in two consecutive iterations. Default = False.
                       tracker[bool]: Set if the progress tracker should be displayed, default is True.
                                      
               [Changes]:
                   
                   self.iter
                   self.dyn_nu
                   self.weights
                   self.sigmas
                   self.errorl'''
        
        # keep track of the progress, this is implemented in all the 'run'-Routines
        # beware of this: IF YOU PLAN TO RUN A LOT OF SINGLE ITERATIONS, TURN THE TRACKER OFF!
        if tracker == True:
            perc_iter, perc_cur, track, cur_perc = init_tracker(max_iter)
            track_progress('Iterating 1D....', perc_cur)
        
        it = 0
        
        
        totalerror = self.errorl[self.iter]
        with open('{}'.format(self.filename), 'a') as file:
            
            file.write('! Running 1DALSCPD. \n')
            
            while totalerror > thresh and it < max_iter:
                
                if it < max_iter:

                    for k in range(np.ndim(self.v_ex)):
                        self.weights, self.dyn_nu[k] = get_update(self.v_ex, self.dyn_nu, self.sigmas, k, prec=prec)
                        self.sigmas = update_sigma(self.dyn_nu, self.sigmas, k)
                    
                    error1 = geterrorleft(self.v_ex, self.weights, self.dyn_nu)
                    error2 = geterrorright(self.v_ex, self.weights, prec)
                    totalerror = get_rmse(error1, error2)
                                

                    file.write('{} {} {} {} \n'\
                               .format(self.iter, np.sqrt(error1)*au2ic, np.sqrt(error2)*au2ic, totalerror))
                
                    self.errorl.append(totalerror)
                    
                    self.iter += 1
                    
                    if self.errorl[self.iter-1] - totalerror < -1:
                        raise RuntimeError('Error increased in iteration {}.'.format(self.iter))
                
                    elif self.errorl[self.iter-1]-totalerror < 1E-2:
                        if dyn == True:
                            self.change_rank(self.rank+5)
                
                    it += 1
                    
                    try:
                        if it == cur_perc:                        
                            perc_iter, perc_cur, track, cur_perc =\
                            keep_track('Iterating 1D....', 'Err: {:.2f}cm-1 '.format(totalerror),\
                                       perc_iter, perc_cur, track, cur_perc)
                        if it == max_iter:
                            track_progress('Iterating 1D....', 1, 'Err: {:.2f}cm-1 '.format(totalerror))
                            print('')                            
                    except:
                        # if this fails there is eiter some unforeseen error or the above initialization didnt
                        # take place, either way we can just keep iterating without the tracker
                        pass
                        
        
    
    def run2D(self, max_iter, thresh, prec=None, dyn=False, YSVD=False, BSVD=False, tracker=True):
        '''self.run2D(max_iter, thresh, SV_prec=.5, prec=None)
                
                Function to iterate through the 2D algorithm while the RMSE is above threshold.
                
                [Args]:
                       max_iter[int]: Maximum amount of iterations to run if convergence isn't reached.
                       thresh[float]: Threshold to reach for convergence. Default 100cm-1.
                       prec[float]: Gives the epsilon for the regularization, standard is root of machine 
                                      precision for float (~1E-8).
                       dyn[bool]: If True the rank of expansion will be increased by one if the error changes
                                      less then 1E-2 in two consecutive iterations. Default = False.
                       
                               !!! SVD may lead to unstable behaviour if dynamic rank expansion is enabled.
                                   Even though the program should automatically determine how many singular
                                   values and vectors are necessary keep in mind that the error might
                                   increase in some iterations.!!!
                                  
                       YSVD[bool]: Build the Y for the subALS LES from the SVD of x_ij. Default False. 

                       BSVD[bool]: Build the B for the subALS LES from the SVD of x_ij. Default False.
                       tracker[bool]: Set if the progress tracker should be displayed, default is True.
                       
               [Changes]:
                   
                   self.iter
                   self.dyn_nu
                   self.weights
                   self.sigmas
                   self.errorl'''
        
        # start tracker if requested
        if tracker == True:
            perc_iter, perc_cur, track, cur_perc = init_tracker(max_iter)     
            track_progress('Iterating 2D....', perc_cur)        
        
        it = 0

        totalerror = self.errorl[self.iter]
        
        # if there is another method to get the mode combinations one can pass them to this routine
        # in shape [[i,j],[i,k],...] instead of generating the comblist here
        comblist = create_comblist(np.ndim(self.v_ex))
        counter = 0
        
        with open('{}'.format(self.filename), 'a') as file:
            
            if YSVD == False:
                file.write('! Running 2DALSCPD. \n')
            elif YSVD == True:
                file.write('! Running 2DALSCPD/SVD. \n')
                
            while totalerror > thresh and it < max_iter:
                
                if counter == 0:
                    # skip every second subiteration, alternating every other outer iteration
                    for n in np.arange(len(comblist))[::2]:                    
                        S_kl = assemble_S2D(self.sigmas, comblist[n][0], comblist[n][1])
                        b_kl = get_b_ein2D(self.v_ex, self.dyn_nu, comblist[n][0], comblist[n][1])
                        x_kl = solve_linear2D(S_kl, b_kl)
                            
                        self.weights, self.dyn_nu, self.sigmas = \
                        runsub(x_kl, S_kl, self.weights, self.dyn_nu, self.sigmas,\
                                   comblist[n][0], comblist[n][1], 20, YSVD=YSVD, BSVD=BSVD)
                                
                        counter = 1
                
                elif counter == 1:
                    for n in np.arange(1,len(comblist))[::2]:
                        S_kl = assemble_S2D(self.sigmas, comblist[n][0], comblist[n][1])
                        b_kl = get_b_ein2D(self.v_ex, self.dyn_nu, comblist[n][0], comblist[n][1])
                        x_kl = solve_linear2D(S_kl, b_kl)
                            
                        self.weights, self.dyn_nu, self.sigmas = \
                        runsub(x_kl, S_kl, self.weights, self.dyn_nu, self.sigmas,\
                                   comblist[n][0], comblist[n][1], 20, YSVD=YSVD, BSVD=BSVD)
                                
                        counter = 0
                
                error1 = geterrorleft(self.v_ex, self.weights, self.dyn_nu)
                error2 = geterrorright(self.v_ex, self.weights, prec)
                totalerror = get_rmse(error1, error2)
                

                file.write('{} {} {} {} \n'\
                    .format(self.iter, np.sqrt(error1)*au2ic, np.sqrt(error2)*au2ic, totalerror))
                
                self.errorl.append(totalerror)
                
                self.iter += 1       
                
                if self.errorl[self.iter-1] - totalerror < -1:
                    if YSVD==False and BSVD==False:    
                        raise RuntimeError('Error increased in iteration {}.'.format(self.iter))
                    elif YSVD==True or BSVD==True:         
                        pass # it is ok
                        #YSVD=False
                        #BSVD=False
                        #print('Switching off SVD in iteration {} due to increasing error.'.format(self.iter))
                        
                elif self.errorl[self.iter-1]-totalerror < 1E-2:
                    if dyn == True:
                        self.change_rank(self.rank+5)
                
                it += 1
                try:
                    if it == cur_perc:
                        perc_iter, perc_cur, track, cur_perc =\
                        keep_track('Iterating 2D....', 'Err: {:.2f}cm-1 '.format(totalerror),\
                                   perc_iter, perc_cur, track, cur_perc)
                    if it == max_iter:
                        track_progress('Iterating 2D....', 1, 'Err: {:.2f}cm-1 '.format(totalerror))
                        print('')
                except:
                    pass            
        
                    
    def runMC(self, max_iter, thresh, prec=None, tracker=True):
        '''self.runMC(max_iter, thresh):
    
                Function to run the 1D ALSCPD-MC Algorithm.
    
            [Args]:
                    max_iter[int]: Maximum amount of iterations to run.
                    thresh[float]: Maximum error to signal convergence.
                    prec[float]: Gives the epsilon for the regularization, standard is root of machine 
                                   precision for float (~1E-8).     
                    tracker[bool]: Set if the progress tracker should be displayed, default is True. 
                    
            [Changes]:
                       
                    self.iter
                    self.dyn_nu
                    self.weights
                    self.nu_smpl
                    self.errorl
                    
            If not existent already, initializes:
                    
                    self.cuts1D
                    self.smpl_idx
                    self.nu_smpl'''
        
        # if the 1Dcuts dont exist initialize them 
        if type(self.cuts1D) != list and type(self.smpl_idx) != np.ndarray:
            try:
                #print('hey')
                self.smpl_idx, self.cuts1D, self.nu_smpl = setup_MC(self.grids, self.nsmpl, self.dyn_nu, self.func1D)
                #print(self.smpl_idx)
            except:
                #print('ho')
                raise RuntimeError('''Something went wrong while initializing 1D MC ALSCPD, 
                perhaps your function isn't compatible?''')
        
        elif type(self.cuts1D) != list and type(self.smpl_idx) == np.ndarray:
            try:
                truesmpl = get_true_points(self.grids, self.smpl_idx)
                self.cuts1D = get_all_cuts_par(self.func1D, self.grids, truesmpl)
            except:
                raise RuntimeError('''Something went wrong while initializing 1D MC ALSCPD, 
                perhaps your function isn't compatible?''')
        
        # start tracker if requested
        if tracker == True:
            perc_iter, perc_cur, track, cur_perc = init_tracker(max_iter)
            #print('')        
            track_progress('Iterating 1DMC..', perc_cur)
        
        it = 0
    
        error = self.errorl[self.iter]
           
        # keep track of the best found solution
        best_weights = cp.deepcopy(self.weights)
        best_SPP = cp.deepcopy(self.dyn_nu)
        best_sigmas = cp.deepcopy(self.sigmas)
        best_error = self.errorl[-1]
        
        with open('{}'.format(self.filename), 'a') as file:
        #print('''{}: weights: {}'''.format(it, weights))
        
            file.write('! Running 1DMCALSCPD. \n')
            
            while it < max_iter and error > thresh:
        
                for i in range(np.ndim(self.v_ex)):
                    omega_i = get_omega_hole_smpl(self.nu_smpl, i)
                    self.weights, self.dyn_nu[i] = update_MC(self.nu_smpl, omega_i, self.cuts1D[i], prec=prec)

                    self.nu_smpl[i] = get_nu_smpl(self.dyn_nu[i], self.smpl_idx[:,i])
            
                err1 = geterrorleft(self.v_ex, self.weights, self.dyn_nu)
                err2 = geterrorright(self.v_ex, self.weights, prec)
                error = get_rmse(err1, err2)
            #print('''weights: {}'''.format(weights))
            #print('{},{},{}'.format(np.sqrt(err1)*au2ic,np.sqrt(err2)*au2ic,error))
                self.errorl.append(error)
          
                file.write('{} {} {} {} \n'\
                            .format(self.iter, np.sqrt(err1)*au2ic, np.sqrt(err2)*au2ic, error))
                
                if error < best_error:
                    best_weights = cp.deepcopy(self.weights)
                    best_SPP = cp.deepcopy(self.dyn_nu)
                    best_sigmas = get_sigmas(best_SPP)
                    best_error = error

                self.iter += 1                      
            #print('Finishing iteration {}'.format(it))
            #print('*'*50)
                it += 1
                try:
                    if it == cur_perc:
                        perc_iter, perc_cur, track, cur_perc =\
                        keep_track('Iterating 1DMC..', 'Err: {:.2f}cm-1 '.format(error),\
                                   perc_iter, perc_cur, track, cur_perc)                        
                except:
                    pass
        
        # set everything to the best result
        self.weights = best_weights
        self.dyn_nu = best_SPP
        self.sigmas = best_sigmas
        self.nu_smpl = get_all_nu_smpl(self.dyn_nu, self.smpl_idx)
        self.errorl[-1] = best_error

        if tracker == True:
            track_progress('Iterating 1DMC..', 1, 'Err: {:.2f}cm-1 '.format(error)) 
            print('')
       
    
    def run2DMC(self, max_iter, thresh, prec=None, tracker=True):
        '''Run the 2DMC-ALSCPD Algorithm.
    
            [Args]:
                    max_iter[int]: Maximum amount of iterations.
                    thresh[float]: Threshhold to signal convergence.
                    prec[float]: Value for the regularization, default is ~1E-8.
                    tracker[bool]: Set if the progress tracker should be displayed, default is True.
                    
            [Changes]:
                       
                    self.iter
                    self.dyn_nu
                    self.weights
                    self.nu_smpl
                    self.errorl

            If not existent already, initializes:
                    
                    self.cuts2D
                    self.smpl_idx
                    self.nu_smpl'''
    
        
        # if the 2D cuts dont exist we initialize them here
        
        if type(self.cuts2D) != list and type(self.smpl_idx) != np.ndarray:
            try:
                #print('hey')
                #print(type(self.smpl_idx))
                self.smpl_idx, comblist, self.cuts2D, self.nu_smpl =\
                     setup_MC2D(self.grids, self.nsmpl, self.dyn_nu, self.func2D)
            except:
                raise RuntimeError('''Something went wrong while initializing 2D MC ALSCPD, 
                perhaps your function isn't compatible?''')
        
        elif type(self.cuts2D) != list and type(self.smpl_idx) == np.ndarray:
            try:
                #print('ho')
                truesmpl = get_true_points(self.grids, self.smpl_idx)
                comblist = create_comblist(len(self.grids))
                #print(self.smpl_idx)
                self.cuts2D = get_cuts_comb_par(comblist, self.func2D, self.grids, truesmpl)
            except:
                raise RuntimeError('''Something went wrong while initializing 2D MC ALSCPD, 
                perhaps your function isn't compatible?''')

            
        else:
            comblist = create_comblist(len(self.grids))
        
        #start tracker if requested
        if tracker == True:
            perc_iter, perc_cur, track, cur_perc = init_tracker(max_iter)
            #print('')
            track_progress('Iterating 2DMC..', perc_cur)
        
        it = 0
    
        error = self.errorl[self.iter]
    
        counter = 0
        
        with open('{}'.format(self.filename), 'a') as file:
            file.write('! Running 2DMCALSCPD. \n')
            while it < max_iter and error > thresh:
                
                # if there is an updated version where we determine the correlated DOF in advance we can put them
                # to a list and just iterate through them here

                if counter == 0:
                    #print(0)
                    for n in np.arange(len(comblist))[::2]:
                        #print(comblist[n][0], comblist[n][1])
                        S_ij = assemble_S2D(self.sigmas, comblist[n][0], comblist[n][1])
                        omega_ij = get_omega_2hole_smpl(self.nu_smpl, comblist[n][0], comblist[n][1])
                        d_ij = build_d2d(self.cuts2D[n], omega_ij)
                        Z_ij = build_Z(omega_ij)
                        x_ij = solve_linear2DMC(Z_ij, d_ij, prec=prec)

                        self.weights, self.dyn_nu, self.sigmas = runsubMC(x_ij, S_ij,\
                                                         self.weights, self.dyn_nu, self.sigmas,\
                                                         comblist[n][0], comblist[n][1], prec=prec)

                        self.nu_smpl[comblist[n][0]] = get_nu_smpl(self.dyn_nu[comblist[n][0]],\
                                                                   self.smpl_idx[:,comblist[n][0]])
                        self.nu_smpl[comblist[n][1]] = get_nu_smpl(self.dyn_nu[comblist[n][1]],\
                                                                   self.smpl_idx[:,comblist[n][1]])                
                        counter = 1

                elif counter == 1:
                    #print(1)
                    for n in np.arange(1,len(comblist))[::2]:
                        S_ij = assemble_S2D(self.sigmas, comblist[n][0], comblist[n][1])                
                        omega_ij = get_omega_2hole_smpl(self.nu_smpl, comblist[n][0], comblist[n][1])
                        d_ij = build_d2d(self.cuts2D[n], omega_ij)
                        Z_ij = build_Z(omega_ij)
                        x_ij = solve_linear2DMC(Z_ij, d_ij, prec=prec)

                        self.weights, self.dyn_nu, self.sigmas = runsubMC(x_ij, S_ij,\
                                                         self.weights, self.dyn_nu, self.sigmas,\
                                                         comblist[n][0], comblist[n][1], prec=prec)

                        self.nu_smpl[comblist[n][0]] = get_nu_smpl(self.dyn_nu[comblist[n][0]],\
                                                                   self.smpl_idx[:,comblist[n][0]])
                        self.nu_smpl[comblist[n][1]] = get_nu_smpl(self.dyn_nu[comblist[n][1]],\
                                                                   self.smpl_idx[:,comblist[n][1]])                  
                        counter = 0

                err1 = geterrorleft(self.v_ex, self.weights, self.dyn_nu)
                err2 = geterrorright(self.v_ex, self.weights, prec)
                error = get_rmse(err1, err2)
                self.errorl.append(error)
                
                self.iter += 1                
                
                file.write('{} {} {} {} \n'\
                            .format(self.iter, np.sqrt(err1)*au2ic, np.sqrt(err2)*au2ic, error))
                it += 1
                try:
                    if it == cur_perc:
                        perc_iter, perc_cur, track, cur_perc =\
                        keep_track('Iterating 2DMC..', 'Err: {:.2f}cm-1 '.format(error),\
                                   perc_iter, perc_cur, track, cur_perc)
                    if it == max_iter:
                        track_progress('Iterating 2DMC..', 1, 'Err: {:.2f}cm-1 '.format(error))
                        print('')                        
                except:
                    pass
                                   

    def plot_error(self, marker='', show=True):
        '''self.plot_error(marker='')
            
                Function to plot the RMSE of the ALS functional over the iterations, first two elements
                are neglected as they are usually needlessly large and essentially meaningless.
                
                [Args]:
                        marker[str]: Set a marker for the indivual point along the curve. Default is
                                    no markers.
                        show[bool]: Sets if the plot is to be shown, default is True.'''
        
        fig, ax = plt.subplots(figsize=(8,5))
        ax.plot(np.arange(len(self.errorl)-2), self.errorl[2:], marker='{}'.format(marker), label='RMSE')
        ax.set_ylabel('RMSE [cm-1]')
        ax.set_xlabel('Iteration')
        ax.legend()
        plt.savefig('{}.pdf'.format(self.filename))
        if show == True:
            plt.show()
    
    
    def get_current(self):
        '''self.get_current()
               
               [Returns]:
                       [array]: self.weights
                       [list]: self.dyn_nu'''
        
        return self.weights, self.dyn_nu
    
    
    def change_rank(self, new_rank):
        '''self.change_rank(new_rank):
            
                Function to increase the CPD rank.
                
                [Args]:
                        new_rank[int]: New rank of the CPD.
                        
                [Changes]:
                    
                    self.rank
                    self.weights
                    self.dyn_nu
                    self.sigmas'''
        

        
        try:
            old_rank = self.rank
        
            new_weights = np.zeros(len(self.weights))
            new_nu_l = []
        
            for i, elem in enumerate(self.dyn_nu):
                # get the random matrices of shape (new_r, Ni)
                new_nu_l.append(np.random.rand(new_rank, elem.shape[1]))
                #print(new_nu_l[i].shape)
                # copy the old nu of rank (r, Ni) into the new random one
                new_nu_l[i][0:elem.shape[0], 0:elem.shape[1]] = elem
                # normalize the new nu of shape (new_r, Ni)
                new_nu_l[i] = get_norm(new_nu_l[i])[1]
            
            new_weights[0:self.rank] = self.weights
        
            # update the objects list of nu
            self.dyn_nu = new_nu_l
            # update the objects weights
            self.weights = new_weights
            # update the objects rank
            self.rank = new_rank
            # update the objects sigmas
            self.sigmas = get_sigmas(self.dyn_nu)
        
            with open('{}'.format(self.filename), 'a') as file:
                file.write('! Rank of expansion was changed from {} to {}. \n'.format(old_rank, self.rank))
        
        except ValueError:
            print('New rank cant be smaller than old rank')
             
            
    def get_perc(self):
        '''self.get_perc()
            
                Function to get the current percentage of points used in the expansion compared to the 
                full tensor.
                
                [Returns]:
                        [float]: Current percentage of points used in expansion compared to full tensor.'''
        
        exp = 0
        true = 1
        for elem in self.Nlist:
            exp += elem*self.rank
            true = true*elem
            
        return (len(self.weights)+exp)/true*100