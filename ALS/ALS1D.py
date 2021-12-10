import numpy as np
import tensorly as tl


'''
Contains the components for the 1D ALSCPD Algorithm aswell as some basic functions for
the ALSCPD class.
'''


def initALS(r, point_list):
    '''Function to initialize the ALS.
    
    [Args]:
            r[int]: Current expansion rank in CPD.
            point_list[list]: List containing the number of grid points along each
                    axis of the full tensor.
                    
    [Returns]:
            c_r_init[array]: Array of shape (r) containing zeros, the initial weights.
            nu_r_list_init[list]: List containing the random normalized SPP of shape (r,Nk) for
                    all DOF.'''
    
    c_r_init = np.zeros(r)
    nu_r_list_init = []
    for i, Nk in enumerate(point_list):
        nu_r_list_init.append(np.random.rand(r, Nk))
        # normalize the initial nu
        c_r_init, nu_r_list_init[i] = get_norm(nu_r_list_init[i])
    c_r_init = np.zeros(r)
    sigmas = get_sigmas(nu_r_list_init)
    return c_r_init, nu_r_list_init, sigmas


def get_norm(nu_k):
    '''Function to get the new weights by normalizing rows of the solution to the
    linear equation of ALS.
    
    [Args]:
            x_k[array]: (r,Nk) shaped array containing the new SPP resulting from solving
                        the linear equation.
                        
    [Returns]:
            [array]: array of the new weights in shape (r).
            [array]: array of the new normalized SPP in shape (r,N).'''
    # old implementation (slow)
    #c_r = np.zeros(nu_k.shape[0])
    #print(x_k.shape)
    #for i in range(nu_k.shape[0]):
        #c_r[i] = np.sqrt(np.dot(nu_k[i],nu_k[i]))
        #nu_k[i,:] /= c_r[i]
        #print((row**2).sum())
        
    # new implementation
    # get the sqrt of the output from np.sum() along axis 1 of the squared array
    weights = np.sqrt((nu_k**2).sum(axis=1))
    # broadcast the weights onto the array to get normalized array
    nu_k /= weights[:, np.newaxis]
    return weights, nu_k


def get_sigmas(nu_store):
    '''Get the matrices which will be multiplied to result in the overlap matrix.
    
    [Args]:
            nu_store[array]: Data structure containing the SPP for each DOF in shape (r,Nk), index running over
                            the DOF.
    [Returns]:
            [List]: List containing all the sigmas (SPP @ {SPP}transposed) in shape (r,r).'''
    
    sigma_list = []
    for nu in nu_store:
        sigma_list.append(nu @ nu.T)
        
    return sigma_list


def update_sigma(nu_store, sigma_list, k):
    '''Update the list of sigmas with the one new one for every time you update the SPP.
    
    [Args]:
            nu_store[array]: Data structure containing the SPP for each DOF in shape (r,Nk), index running over
                            the DOF.
            sigma_list[list]: List containing all the current sigmas (SPP @ {SPP}transposed).
            k[int]: Index of the updated DOF.
            
    [Returns]:
            [list]: Updated list of sigmas.'''
    
    sigma_list[k] = nu_store[k] @ nu_store[k].T
    return sigma_list


def assemble_S(sigma_list, hole_index = None):
    '''Assemble the overlap matrix from a given list of sigmas. One can choose to neglect one sigma.
    
    [Args]:
            sigma_list[list]: List containing the individual terms for all the SPP.
            hole_index[int]: Index of the sigma to be neglected. Default=None returns full overlap matrix.
            
    [Returns]:
            [Array]: Array containing the overlap matrix for a given set of SPP.'''
    
    # if no hole we return the full overlap 
    if hole_index == None:
        S = np.ones(sigma_list[0].shape)
        for sigma in sigma_list:
            S = S * sigma
        return S
    
    # if hole is in position 0 we just skip the first element
    elif hole_index == 0:
        S = np.ones(sigma_list[0].shape)
        for sigma in sigma_list[1:]:
            S = S * sigma
        return S
    
    # else we check if current index is hole index
    else:
        S = np.ones(sigma_list[0].shape)
        for i, sigma in enumerate(sigma_list):
            if i != hole_index:
                S = S * sigma
        return S

    
def get_b_ein(V, nu_store, hole_index):
    '''Function to iteratively create input for the np.einsum routine to contract a given tensor with given
    set of SPP for all DOF except one indicated.
    
    [Args]:
            V[array]: The exact tensor to be contracted with shape (N0, N1, N2,... Nf).
            nu_store[list]: List of the SPP with index running over the DOF in shape (r,N).
            hole_index[int]: Index of the DOF to be neglected with SPP of shape (r,Nk).
            
    [Returns]:
            [Array]: Contracted tensor with shape (r,Nk).
            
    -tested for up to 6D potential.'''
    
    
    # get a copy of V
    b_out = V.copy()
    # get a reference, this could be any integer but cannot be lower than the DOF
    ref = np.ndim(V)
    # get a counter variable to go through the nu backwards
    i = len(nu_store)-1
    # get a counter variable to keep track of the dimensionality of the tensor, for the first
    # iteration it will be N->N, every following one will be N->N-1
    next_dim = ref
    # get the initial shape, this will always be the same
    old_shape = np.arange(0, ref).tolist()
    # go through all the nu
    for nu in nu_store:
        # check if i is currently indicating the hole
        if i != hole_index:
            # if no, check if we are in the last iteration
            if i > 0:
                # if no, set up the shape of the tensor after the contraction, it will be next_dim-dimensional
                # and we will have our 'contracted' coordinate in the i-th axis (going from back to front)
                new_shape = np.arange(0, next_dim).tolist()
                new_shape[i] = ref
                # if i is smaller than the hole index we have passed the hole already and the last element in the
                # list must be the hole index
                if i < hole_index:
                    new_shape[-1] = hole_index
            # if i==0 we have reached the last iteration, the next tensor should always be 2D and we define the
            # shape of the output. Doing this we always get the b(0) transposed (wrt the others), but
            # we can just transpose it later.
            elif i == 0:
                new_shape = np.arange(0, next_dim).tolist()
                new_shape[0] = ref
                new_shape[1] = hole_index
            #print('{}, [{},{}] -> {}'.format(old_shape, ref, i, new_shape), 'i: {}'.format(i), 'Next Dim: {}'.format(next_dim))
            # iteratively feed the current state of the tensor and the prepared future shape with the i-th nu
            # into the np.einsum routine
            b_out = np.einsum(b_out, old_shape, nu_store[i], [ref, i], new_shape)
            # to keep everything consistent we set the next starting shape to the current ending shape
            old_shape = new_shape
            # decrease the dimensionality of the N+1-th tensor by 1
            next_dim -= 1
        # decrease the counter indicating the nu by 1
        i -= 1
    if hole_index == 0:
        return b_out.T
    else:    
        return b_out
    

# solve the linear equation
def solve_linear(S, b, prec = None):
    '''A function to solve the linear equation in 1D ALS.
    
    [Args]:
            S[array]: (r,r) array containing the S_k k-hole overlap matrix.
            b[array]: (r,Nk) array containing the b_k k-hole overlap with the
                        exact tensor.
            prec[float]: Gives the epsilon for the regularization, standard is root of machine precision for
                    float (~1E-8).
                    
    [Returns]:
            [array]: (r,Nk) array containing the new nu_k non-normalized.'''
    if prec == None:
        prec = np.sqrt(np.finfo(float).eps)
    S_in = np.zeros(S.shape)
    S_in = S + prec*np.identity(S.shape[0])
    return np.linalg.solve(S_in, b)


# factor to go from au to cm-1
au2ic = 219474.63137


def geterrorleft(V_ex, weights, nu_list):
    '''Get the mean square error between the initial tensor and the current tensor which will
    be rebuild from factor matrices given as shape (r,N).
    
    [Args]:
            V_ex[array]: Exact tensor of shape (N1, N2,..., Nf).
            weights[array]: shape (r) array with the weights for the reconstruction of the current
                        tensor.
            nu_list[list]: List of the factor matrices of shape (r,N) to rebuild the current tensor.
            
    [Returns]:
            [float]: Mean square error [cm-1] between initial tensor and rebuild tensor.'''
    
    nu_listT = []
    # the tensorly routine needs the factor matrices in shape (N,r)
    for elem in nu_list:
        nu_listT.append(elem.T)
    
    tensor = tl.cp_to_tensor((weights, nu_listT))
    
    error = (((V_ex - tensor)**2).mean())
    return error


def geterrorright(v_ex, c_r, prec=None):
    '''Get the mean squared error for the right hand side of the ALS functional given as the 
    root of the sum over the squared weights multiplied by the regularization devided by the number of weights.
    
    [Args]:
            c_r[array]: (r) shaped array containing the current weights.
            prec[float]: Gives the epsilon for the regularization, standard is root of machine precision for
                    float (~1E-8).
                    
    [Returns]:
            [float]: The mean squared error [cm-1] for the right hand side ALS functional.'''
    
    if prec == None:
        prec = np.sqrt(np.finfo(float).eps)
    error = (((prec*(c_r**2)).sum())/v_ex.size)
    return error


def get_rmse(er1, er2):
    '''Get the RMSE of the complete 1D ALS functional in cm-1.
    
    [Args]:
        er1[float]: MSE of the left hand side of the ALS functional.
        er2[float]: MSE of the right hand side of the ALS functional.
        
    [Returns]:
        [float]: RMSE of the complete ALS functional in cm-1.'''
    return np.sqrt(er1+er2)*au2ic


def get_update(v_ex, nu_r, sigmas, k, prec=None):
    '''Function to get the updated weights and the normalized new nu for one DOF.
    
    [Args]:
            v_ex[array]: Array of shape (N1, N2,..., Nf) containing the exact potential.
            nu_r[list]: List of the SPP for all DOF given in shape (r,N).
            sigmas[list]: List of the sigmas to build the S-matrix. (Sigma_k = nu_r[k]@nu_r[k].T).
            k[int]: Index of the current DOF to update.
            
    [Returns]:
            [Array]: (r) shaped array containing the new weights.
            [array]: (r,Nk) shaped array containing the new normalized SPP.'''
    # get the S
    S_k = assemble_S(sigmas, hole_index=k)
    # get the b
    b_k = get_b_ein(v_ex, nu_r, k)
    # get the nu
    nu_k = solve_linear(S_k, b_k, prec=prec)
    # norm the nu
    c_r_k, nu_k = get_norm(nu_k)
    # return the weights, the new nu
    return c_r_k, nu_k