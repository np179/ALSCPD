import numpy as np
from ALS.ALS1D import *
from ALS.twoDsub import *
from ALS.tracker import *
from os import sched_getaffinity
import multiprocessing as mp

def rndm(upper, nsmpl):
    '''Function to create a given amount of random integer values in a given half-open interval
    with lower border set to 0.
    
    [Args]:
            upper[float]: Upper bound below which values are chosen.
            amount[int]: Number of integer values to return.
            
    [Return]:
            [array]: Array containing the random integers.'''
    
    return np.random.randint(0, upper, size=nsmpl)


def get_points(Grid_List, nsmpl):
    '''Function to get a set amount of sampling points.
    
    [Args]:
            V[array]: The complete tensor.
            nsmpl[int]: Amount of sampling points.
            
    [Returns]:
            [array]: Array with the sampling points of shape (s, np.ndim(V)).'''
    
    #get the random sampling point matrix in index representation
    V_dim = len(Grid_List)
    smpl = np.zeros((nsmpl, V_dim), dtype=int)
    for i in range(V_dim):
        smpl[:,i] = rndm(len(Grid_List[i]), nsmpl)
    return smpl


def get_true_points(Grid_list, sample_points):
    '''Function to map the index representation of the sampling points onto the 
    corresponding grid points.
    
    [Args]:
            Grid_List[list]: List containing the complete grids for the coordinates.
            sample_points[array]: Array containing the sampling points in index representation.
            
    [Returns]:
            [array]: Array containing the sampling points with the corresponding grid coordinates. (s, np.ndim(V))'''
    
    # this will be pretty slow but we don't have to call it that often right?
    out = np.zeros(sample_points.shape)
    for i, elem in enumerate(sample_points):
        for j, k in enumerate(elem):
            try:
                out[i,j] = Grid_list[j][k]
            except IndexError:
                print('Invalid index encountered in row {}, column {}.'.format(i,j))
                raise RuntimeError('Encountered invalid index in sampling points.')
    return out


def get_cut(constructor, grid, gridindex, sample_point):
    '''Function to get a cut through a given function object along one individual grid for a 
    given sample point.
    
    [Args]:
            constructor[function]: Function to create the cut from, must be callable by the individual
                        grids along each coordinate.
            grid[array]: Grid along which the cut should be represented. Shape (Ni,)
            gridindex[int]: Index of the grid to cut along.
            sample_point[list]: List of len np.ndim(V) containing the value for each coordinate in this sample point.
            
    [Returns]:
            [array]: Cut through the potential of shape (Ni,).'''
    
    # get the cuts of the potential for a given sampling point
    # -> fix everything to sample and go along indicated coordinate
    internal = sample_point.copy()
    internal[gridindex] = grid
    #print(internal)
    # call the constructor function with the created list
    return constructor(*internal)


def get_cut2D(constructor, grd1, grdidx1, grd2, grdidx2, sample_point):
    '''Function to get a 2D cut through a given function object along two grids for a given sample point.
    
    [Args]:
            constructor[function]: Function to create the cut from, must be callable by the individual
                        grids along each coordinate.
            grd1[array]: Grid for the first coordinate of shape (Ni,).
            grdidx1[int]: Index for the first coordinate.
            grd2[array]: Grid for the second coordinate of shape (Nk,).
            grdidx2[int]: Index for the second coordinate.
            sample_point[array]: Array containing one sample point of shape (np.ndim(V),).
    
    [Returns]:
            [array]: 2D cut along designated coordinates of shape (Ni,Nk)'''
    
    internal = sample_point.copy()
    internal[grdidx1] = grd1
    internal[grdidx2] = grd2
    # formally reshape this here in case the function returns the fixed coordinates as empty axis
    return constructor(*internal).reshape(len(grd1), len(grd2))


def get_cuts_ind(constructor, grid, gridindex, sample_points):
    '''Function to get all cuts for all sample points for a given coordinate.
    
    [Args]:
            constructor[function]: Function to create the cut from, must be callable with the individual
                        grids along each coordinate.
            grid[array]: Grid along which the cut should be represented. Shape (Ni,)
            gridindex[int]: Index of the grid to cut along.
            sample_points[array]: Array of the sampling points of shape (s, np.ndim(V)).
    
    [Returns]:
            [array]: 1D Scan along coordinate for all sample points, shape (Ni,s).'''
    
    cutl = []
    for elem in sample_points:
        cutl.append(get_cut(constructor, grid, gridindex, elem.tolist()))
    out = np.array(cutl, dtype=float).T.reshape(len(grid), len(sample_points))
    return out


def get_cuts_ind2D(constructor, grd1, grdidx1, grd2, grdidx2, sample_points):
    '''Function to get all cuts for all sample points for a given combination of coordinates.
    
    [Args]:
            constructor[function]: Function to create the cut from, must be callable with the individual
                        grids along each coordinate.
            grd1[array]: Grid for the first coordinate of shape (Ni,).
            grdidx1[int]: Index for the first coordinate.
            grd2[array]: Grid for the second coordinate of shape (Nk,).
            grdidx2[int]: Index for the second coordinate.
            sample_points[array]: Array containing all sample points in shape (s,np.ndim(V)).
            
    [Returns]:
            [array]: 2D scan along coordinates for all sample points, shape (Ni,Nk,s).'''
    
    cutl = []
    for elem in sample_points:
        cutl.append(get_cut2D(constructor, grd1, grdidx1, grd2, grdidx2, elem.tolist()))
    return np.moveaxis(np.array(cutl, dtype=float), 0, 2)


def get_all_cuts(constructor, Grid_List, sample_points):
    '''Function to get all cuts along all coordinates for all given sampling points.
    
    [Args]:
            constructor[function]: Function to create the cut from, must be callable with the individual
                        grids along each coordinate.
            Grid_List[list]: List containing the grids for each coordinate used to create the original tensor.
            sample_points[array]: Array of the sampling points of shape (s, np.ndim(V)).
            
    [Returns]:
            [list]: List containing the 1D Scans along all of the coordinates for the sampling points
                    in shape (Ni,s).'''
    
    cutsl = []
    for gridindex, grid in enumerate(Grid_List):
        cutsl.append(get_cuts_ind(constructor, grid, gridindex, sample_points))
        track_progress('Building 1D cuts', (gridindex+1)/len(Grid_List), ' Mode:[{}/{}] '.format(gridindex+1, len(Grid_List)))
    #out = [np.flip(elem,axis=1) for elem in cutsl]
    return cutsl


def get_all_cuts_par(constructor, Grid_List, sample_points):
    
    
    out = []
    # for more comments on this, see the function below
    global job1
    def job1(idx):
        job_res = get_cuts_ind(constructor, Grid_List[idx], idx, sample_points)
        return job_res
    
    cpus = len(sched_getaffinity(0))    
    idx_l = np.arange(len(Grid_List), dtype=int)
    for i in range(len(idx_l))[::cpus]:
        sublist = idx_l[i:i+cpus]
             
        track_progress('Building 1D cuts', (i)/(len(idx_l)-1), 'Parallel:{} '.format(len(sublist)))        
            
        with mp.Pool(len(sublist)) as p:
            subsublist = p.map(job1, sublist)
            for elem in subsublist:
                out.append(elem)

    print('\r'+' '*100, end='')
    track_progress('Building 1D cuts', 1)        
    return out  


def get_cuts_comb_par(comblist, constructor, Grid_List, sample_points):
    '''Function to parallelize the task of building the 2D cuts. Will try to use as many CPU cores as
    it has access to.
    
    [Args]:
            comblist[list]: List containing list with the combinations, e.g. [[i,j],[i,k],...].
            constructor[function]: Function to compute the cuts, should take the grids individually.
            Grid_List[list]: List containing the grids.
            sample_points[array]: Array of the sampling points of shape (s, np.ndim(V)).

    [Returns]:
            [list]: List containing the 2D scans along the indicated coordinate combinations for the sampling
                    points in shape (Ni,Nk,s).'''
    
    out = []
    # apparently we have to declare the job function as global to pickle it for 
    # multiprocessing... However, we do also need the variables from our local
    # namespace. But this seems to work.
    global job
    def job(elem):
        '''Job dispatcher for the parallelization pool of the 2D cut routine'''
        job_res = get_cuts_ind2D(constructor, Grid_List[elem[0]], elem[0], Grid_List[elem[1]], elem[1], sample_points) 
        return job_res   
    
    # get the number of available CPU cores to determine how many jobs we can run in parallel 
    cpus = len(sched_getaffinity(0))
    # devide the length of the complete list of combinations by the number of cores available
    for i in range(len(comblist))[::cpus]:
        # extract elements for the individual cores into sublist
        sublist = comblist[i:i+cpus]
        #print(sublist)

        track_progress('Building 2D cuts', (i)/(len(comblist)-1), 'Parallel:{} '.format(len(sublist)))
            
        # create and dispatch the jobs to the cores
        with mp.Pool(len(sublist)) as p:
            # this will return a list of the individual results from the jobs in the order of the sublist
            subsublist = p.map(job, sublist)
            #print(subsublist[0].shape)
            # put the elements into one complete list which will correspond to the list of combinations
            for elem in subsublist:
                #print(elem.shape)
                out.append(elem)

    print('\r'+' '*100, end='')
    track_progress('Building 2D cuts', 1)        
    return out            


def get_cuts_comb(comblist, constructor, Grid_List, sample_points):
    '''Get 2D scans from the potential along the coordinate combinations indicated by the combinations list.
    
    [Args]:
            comblist[list]: List of lists containing the indices for the 2D scans e.g. [[i,j],[i,k]...].
            constructor[function]: Function to create the cut from, must be callable with the individual
                        grids along each coordinate.
            Grid_List[list]: List containing the grids for each coordinate used to create the original tensor.
            sample_points[array]: Array of the sampling points of shape (s, np.ndim(V)).
            
    [Returns]:
            [list]: List containing the 2D scans along the indicated coordinate combinations for the sampling
                    points in shape (Ni,Nk,s).'''
    
    twoDcuts = []
    for i, elem in enumerate(comblist):
        twoDcuts.append(get_cuts_ind2D(constructor, Grid_List[elem[0]],\
                            elem[0], Grid_List[elem[1]], elem[1], sample_points))
        track_progress('Building 2D cuts', (i+1)/len(comblist),' Comb:[{}/{}] '.format(i+1, len(comblist)))
    return twoDcuts


def get_nu_smpl(nu_k, smpl_idx_k):
    '''Function to get one specific sampling SPP by mapping the corresponding sampling
    index onto the grid axis of the original SPP.
    
    [Args]:
            nu_k[array]: Original SPP of shape (r, N).
            smpl_idx_k[array]: Array containing the grid indices for the sampling along coordinate.
    
    [Returns]:
            [array]: The mapped SPP from the SPP corresponding to the passed 
                    sampling index in shape (r,s).'''
    
    # get the values for the given sample point components along one
    # coordinate from the SPP of the coordinate
    nu_smpl_k = nu_k[:,smpl_idx_k]
    return nu_smpl_k


def get_all_nu_smpl(nu_list, smpl_idx):
    '''Function to get the mapped SPP for all DOF.
    
    [Args]:
            nu_list[list]: List containing the SPP in shape (r,N).
            smpl_idx[array]: Array of shape (s,len(nu_list)) containing the sampling points in
                        index representation.
                        
    [Returns]:
            [array]: Array containing the new mapped SPP in shape (r,s) along the first axis.'''
    
    # get the points from all the SPP which correspond to the sampling index for that coordinate
    nu_smpl_l = [get_nu_smpl(nu_list[idx], smpl_idx[:,idx]) for idx in range(len(nu_list))]
    return np.array(nu_smpl_l, dtype=float)

    
def get_omega_smpl(nu_smpl):
    '''Function to build the full sampled omega by elementwise multiplying the sampled SPP.
    
    [Args]:
            nu_smpl[array]: Array containing the sampled SPP in shape (r,s) along the first axis.
           
    [Returns]:
            [array]: Array of shape (r,s) containing the full sampled omega.'''
    
    # get the full omega
    omega_smpl = np.ones(nu_smpl[0].shape)
    for nu in nu_smpl:
        omega_smpl *= nu
    return omega_smpl


def get_omega_hole_smpl(nu_smpl, idx):
    '''Function to build the one-hole sampled omega by elementwise multiplying the sampled SPP
    neglecting one indicated.
    
    [Args]:
            nu_smpl[array]: Array containing the sampled SPP in shape (r,s) along the first axis.
            idx[int]: Index of the sampled SPP to be neglected.
           
    [Returns]:
            [array]: Array of shape (r,s) containing the one-hole sampled omega.'''
    
    # get the omega while neglecting one smplSPP
    omega_smpl = np.ones(nu_smpl[0].shape)
    for i, nu in enumerate(nu_smpl):
        if i != idx:
            omega_smpl = omega_smpl * nu
    return omega_smpl


def get_omega_2hole_smpl(nu_smpl, idx1, idx2):
    '''Function to build the two-hole sampled omega by elementwise multiplying the sampled SP
    neglecting two indicated.
    
    [Args]:
            nu_smpl[array]: Array containing the sampled SPP in shape (r,s) along the first axis.
            idx1[int]: Index of the first SPP to be neglected.
            idx2[int]: Index of the second SPP to be neglected.
            
    [Returns]:
            [array]: Array of shape (r,s) containing the two-hole sampled omega.'''
    
    omega_smpl = np.ones(nu_smpl[0].shape)
    for i, nu in enumerate(nu_smpl):
        if i != idx1 and i != idx2:
            omega_smpl = omega_smpl * nu
    return omega_smpl


def build_d(cut, omega):
    '''Build the d from the one-hole omega and the corresponding cuts.
    
    [Args]:
            cut[array]: 1D cuts through the potential along one specific coordinate, shape (Ni,s).
            omega[array]: One-hole omega of shape (r,s).
            
    [Retuns]:
            [array]: d for the LES of shape (r,Ni).'''
    
    return omega@cut.T


def build_d2d(cuts, omega_ij):
    '''Build the 2D-d from the two-hole omega and the corresponding cuts.
    
    [Args]:
            cuts[array]: 2D cuts through the potential of shape (Ni,Nk,s).
            omega_ij[array]: Two-hole omega of shape (r,s).
            
    [Returns]: 2D-d for the LES of shape (r,Ni,Nk).'''
    
    return np.einsum(cuts, [0,1,2], omega_ij, [3,2], [3,0,1])


def build_Z(omega):
    '''Build the Z from the one-hole omega.
    
    [Args]:
            omega[array]: One-hole omega of shape (r,s).
            
    [Returns]:
            [array]: Z for the LES of shape (r,r').'''
    
    return omega@omega.T


def solve_linear2DMC(Z_ij, d_ij, prec=None):
    '''Solve the LES for the 2DMC Algorithm.
    
    [Args]:
            Z_ij[array]: Two-hole Z of shape (r,r).
            d_ij[array]: Two-hole d of shape (Ni,Nk,s).
            prec[float]: Precision to be used for regularization. Default is sqrt machine prec (~1E-8).
            
    [Returns]:
            [array]: Solution to LES of shape (r,Ni,Nk).'''
    
    if prec == None:
        prec = np.sqrt(np.finfo(float).eps)
    
    Z_in = np.zeros(Z_ij.shape)
    Z_in = Z_ij + prec*np.identity(Z_ij.shape[0])
    
    d_resh = d_ij.reshape(d_ij.shape[0], d_ij.shape[1]*d_ij.shape[2], order='C')
    
    x_ij = np.linalg.solve(Z_in, d_resh)
    return x_ij.reshape(d_ij.shape[0], d_ij.shape[1], d_ij.shape[2], order='C')


def update_MC(nu_smpl, omega_idx, cut, prec=None):
    '''Function to update the SPP for one DOF.
    
    [Args]:
            nu_smpl[array]: Array of shape (np.ndim(V),r,s) containing all sampled SPP.
            omega_idx[array]: One-hole omega of shape (r,s).
            cut[array]: 1D  cuts through the potential along one specific coordinate, shape (Ni,s).
            
    [Returns]:
            [array]: Array of shape (r,) containing the new weights.
            [array]: Array of shape (r,Ni) containing the new normalized nu.'''
    
    # get the d
    d_idx = build_d(cut, omega_idx)
    #d_man = build_d_man(cut, omega_idx)
    #print('d_{} correct? {}'.format(idx, np.allclose(d_idx,d_man)))
    # get the Z
    Z_idx = build_Z(omega_idx)
    #print('Z_{} symmetrical? {}'.format(idx, all(Z_idx-Z_idx.T)==0))
    # solve the linear equation
    x_idx = solve_linear(Z_idx, d_idx, prec=prec)
    #print(x_idx.shape)
    # norm the SPP
    weights, SPP_idx = get_norm(x_idx)
    
    return weights, SPP_idx


def runMC(V_ex, weights, SPP, nu_smpl, cuts, smpl_idx, max_iter, thresh):
    '''Function to run the 1D ALSCPD-MC Algorithm.
    
    [Args]:
            V_ex[array]: Exact tensor to compute the error.
            weights[array]: Array of shape (r,) containing the weights.
            SPP[list]: List of the SPP in shape (r,Ni).
            nu_smpl[array]: Array of shape (np.ndim(V),r,s) containing the sampled SPP.
            cuts[list]: List containing the 1D Scans for all DOF.
            smpl_idx[array]: Index representation of the sampling points, shape (s, np.ndim(V)).
            max_iter[int]: Maximum amount of iterations to run.
            thresh[float]: Maximum error to signal convergence.
            
    [Returns]:
            [list]: List containing the error for all iterations.'''
    errorl = []
    it = 0
    
    err1 = geterrorleft(V_ex, weights, SPP)
    err2 = geterrorright(V_ex, weights)
    error = get_rmse(err1, err2)
    errorl.append(error)
    
    #print('''{}: weights: {}'''.format(it, weights))
    while it < max_iter and error > thresh:
        
        for i in range(np.ndim(V_ex)):
            omega_i = get_omega_hole_smpl(nu_smpl, i)
            weights, SPP[i] = update_MC(nu_smpl, omega_i, cuts[i])

            nu_smpl[i] = get_nu_smpl(SPP[i], smpl_idx[:,i])
            
        err1 = geterrorleft(V_ex, weights, SPP)
        err2 = geterrorright(V_ex, weights)
        error = get_rmse(err1, err2)
        #print('''weights: {}'''.format(weights))
        #print('{},{},{}'.format(np.sqrt(err1)*au2ic,np.sqrt(err2)*au2ic,error))
        errorl.append(error)
        #print('Finishing iteration {}'.format(it))
        #print('*'*50)
        it += 1
        
    return errorl


def runsubMC(x_ij, S_ij, weights, SPP, sigmas, i, j, max_it=20, prec=None):
    '''Run the subiterations for the 2DMC-ALS on the full indices.
    
    [Args]:
            x_ij[array]: Solution to the 2DMC LES of shape (r,Ni,Nk).
            S_ij[array]: Two-hole overlap for the full SPP. (r,r).
            weights[array]: Weights of shape (r,).
            SPP[list]: List of the full SPP in shape (r,N).
            sigmas[list]: List of the ovelapmatrices for the SPP of shape (r,r).
            i[int]: Index for the first DOF.
            j[int]: Index for the second DOF.
            max_it[int]: Maximum amount of subiterations, default set to 20.
            prec[float]: Value for the regularization, default is ~1E-8.
            
    [Returns]:
            [array]: New weights, shape (r,).
            [list]: Updated list of all SPP in shape (r,N).
            [list]: Updated list of all sigmas in shape (r,r).'''
    
    it = 0
    errorsub = []
    err1 = errorVttsq(S_ij, x_ij)
    errorsub.append(err1)
    errorsub.append(0)
    
    while abs(errorsub[-2]-errorsub[-1]) > errorsub[0]/100 and it < max_it:

        S_i = add_sigma(S_ij, sigmas, j)
        Y_i = get_ein_spec(x_ij, SPP[j], 2)
        x_i = solve_linearsub(S_ij, Y_i, S_i, prec=prec)
        weights, SPP[i] = get_norm(x_i)
        sigmas = update_sigma(SPP, sigmas, i)
        

        S_j = add_sigma(S_ij, sigmas, i)
        Y_j = get_ein_spec(x_ij, SPP[i], 1)
        x_j = solve_linearsub(S_ij, Y_j, S_j, prec=prec)
        weights, SPP[j] = get_norm(x_j)
        sigmas = update_sigma(SPP, sigmas, j)
        
        err2 = error2VttVt(S_ij, x_ij, weights, SPP[i], SPP[j])
        err3 = errorVtsq(x_ij, weights, S_j, sigmas, j)        
        errreg = errorreg(x_ij, S_j, SPP[j])
        error = getrmsesub(err1, err2, err3, errreg)
        errorsub.append(error)
        
        it += 1
        
    return weights, SPP, sigmas


def run2DMC(V_ex, weights, SPP, nu_smpl, comblist, smpl_idx, cuts, sigmas, max_iter, thresh, prec=None):
    '''Run the 2DMC-ALSCPD Algorithm.
    
    [Args]:
            V_ex[array]: The exact tensor in shape (Ni,Nj,...,Nf).
            weights[array]: The weights in shape (r,).
            nu_smpl[array]: Array of the sampled SPP in shape (r,s) along first axis.
            comblist[list]: List of list containing the combinations . [[i,j],[i,k]...]
            smpl_idx[array]: Array of the sampling points.
            cuts[list]: List of the 2D cuts for all combinations.
            sigmas[list]: List of the sigmas for the full SPP.
            max_iter[int]: Maximum amount of iterations.
            thresh[float]: Threshhold to signal convergence.
            prec[float]: Value for the regularization, default is ~1E-8.
            
    [Returns]:
            [list]: List containing the error for each iteration.'''
    
    errorl = []
    
    it = 0
    
    err1 = geterrorleft(V_ex, weights, SPP)
    err2 = geterrorright(V_ex, weights)
    error = get_rmse(err1, err2)
    errorl.append(error)
    
    counter = 0
    
    while it < max_iter and error > thresh:
        
        # if there is an updated version where we determine the correlated DOF in advance we can put them
        # to a list and just iterate through them here
        
        if counter == 0:
            #print(0)
            for n in np.arange(len(comblist))[::2]:
                #print(comblist[n][0], comblist[n][1])
                S_ij = assemble_S2D(sigmas, comblist[n][0], comblist[n][1])
                omega_ij = get_omega_2hole_smpl(nu_smpl, comblist[n][0], comblist[n][1])
                d_ij = build_d2d(cuts[n], omega_ij)
                Z_ij = build_Z(omega_ij)
                x_ij = solve_linear2DMC(Z_ij, d_ij, prec=prec)
                weights, SPP, sigmas = runsubMC(x_ij, S_ij, \
                                                 weights, SPP, sigmas, \
                                                 comblist[n][0], comblist[n][1], prec=prec)
                nu_smpl[comblist[n][0]] = get_nu_smpl(SPP[comblist[n][0]], smpl_idx[:,comblist[n][0]])
                nu_smpl[comblist[n][1]] = get_nu_smpl(SPP[comblist[n][1]], smpl_idx[:,comblist[n][1]])                
                counter = 1
        
        elif counter == 1:
            #print(1)
            for n in np.arange(1,len(comblist))[::2]:
                S_ij = assemble_S2D(sigmas, comblist[n][0], comblist[n][1])                
                omega_ij = get_omega_2hole_smpl(nu_smpl, comblist[n][0], comblist[n][1])
                d_ij = build_d2d(cuts[n], omega_ij)
                Z_ij = build_Z(omega_ij)
                x_ij = solve_linear2DMC(Z_ij, d_ij, prec=prec)
                weights, SPP, sigmas = runsubMC(x_ij, S_ij, \
                                                 weights, SPP, sigmas, \
                                                 comblist[n][0], comblist[n][1], prec=prec)
                nu_smpl[comblist[n][0]] = get_nu_smpl(SPP[comblist[n][0]], smpl_idx[:,comblist[n][0]])
                nu_smpl[comblist[n][1]] = get_nu_smpl(SPP[comblist[n][1]], smpl_idx[:,comblist[n][1]])                  
                counter = 0
                
        err1 = geterrorleft(V_ex, weights, SPP)
        err2 = geterrorright(V_ex, weights)
        error = get_rmse(err1, err2)
        errorl.append(error)
        
        it += 1
        
    return errorl


def setup_MC(grid_list, nsmpl, SPP, constructor):
    '''Function to set up the ALSCPD-MC Algorithm.
    
    [Args]:
            grid_list[list]: List containing the original grids for the exact tensor.
            nsmpl[int]: Number of sampling points.
            SPP[list]: List containing the SPP in shape (r,Ni).
            constructor[function]: Function to calculate the potential cuts.
            
    [Returns]:
            [array]: Sampling points in index representation of shape (s,np.nidm(V)).
            [list]: List containing the 1D cuts for all DOF in shape (Ni,s).
            [array]: Array of shape (np.ndim(V),r,s) containing the sampled SPP for all DOF.'''
    
    # get the points in index rep
    smpl_idx = get_points(grid_list, nsmpl)
    #print('S index: {}'.format(smpl_idx))
    # map them to the grids
    truesmpl = get_true_points(grid_list, smpl_idx)
    #print('True s: {}'.format(truesmpl))
    # get the cuts
    cuts = get_all_cuts_par(constructor, grid_list, truesmpl)
    #print('Cuts shape: {}'.format([cut.shape for cut in cuts]))
    # get the sampled SPP
    nu_smpl = get_all_nu_smpl(SPP, smpl_idx)
    #print('nu_smpl: {}'.format(nu_smpl))
    return smpl_idx, cuts, nu_smpl


def setup_MC2D(grid_list, nsmpl, SPP, constructor):
    '''Function to set up the 2D ALSCPD-MC Algorithm.
    
    [Args]:
            grid_list[list]: List containing the original grids for the exact tensor.
            nsmpl[int]: Number of sampling points.
            SPP[list]: List containing the SPP in shape (r,Ni).
            constructor[function]: Function to calculate the potential cuts.
            
    [Returns]:
            [array]: Sampling points in index representation of shape (s,np.nidm(V)).
            [list]: List of lists containing the mode combinations like [[i,j],[i,k],...].
            [list]: List containing the 2D cuts for all DOF in shape (Ni,Nj,s).
            [array]: Array of shape (np.ndim(V),r,s) containing the sampled SPP for all DOF.'''
    
    smpl_idx = get_points(grid_list, nsmpl)
    truesmpl = get_true_points(grid_list, smpl_idx)
    combl = create_comblist(len(grid_list))
    #cuts2D = get_cuts_comb(combl, constructor, grid_list, truesmpl)
    cuts2D = get_cuts_comb_par(combl, constructor, grid_list, truesmpl)
    nu_smpl = get_all_nu_smpl(SPP, smpl_idx)
    
    return smpl_idx, combl, cuts2D, nu_smpl


def grab_smpl(filename):
    '''Function to grab presampled points in index representation from a file.
    
    [Args]:
            filename[str]: File to extract the sampling points from. It is assumed that the
                    sampling information starts in the second line.
            
    [Returns]:
            [array]: Array with the sampling points in shape (s,np.ndim(V)).'''
    
    lines = []
    with open('{}'.format(filename), 'r') as file:
        for line in file:
            lines.append(line.split())
    smpl_idx = np.zeros((len(lines)-1, len(lines[1])), dtype=int)
    for i, elem in enumerate(lines[1:]):
        smpl_idx[i,:] = elem
    return smpl_idx


def create_comblist(vdim):
    '''Create the list of all possible and sensible mode combinations.
    
    [Args]:
            vdim[int]: Amount of modes in system.
            
    [Returns]:
            [list]: List of lists containing the combinations in index representation.'''
    
    fullcomb = []
    for i in range(vdim):
        for k in range(vdim):
            if i != k and i < k:
                fullcomb.append([i,k])
    return fullcomb