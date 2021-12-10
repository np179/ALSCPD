from ALS.ALS1D import *
from ALS.ALS2D import *
import numpy as np
import matplotlib.pyplot as plt


def runsub(x_ij, S_ij, weights, nu_list, sigmas, i, j, max_it=20, BSVD=False, YSVD=False):
    '''Function to perform the subALS iterativelly updating the ith and jth SPP.
    
    [Args]:
            x_ij[array]: Input array to be contracted, shape (r,Ni,Nj).
            S_ij[array]: Two-hole overlap matrix for the SPP of shape (r,r).
            weights[array]: Weights of the SPP of shape (r).
            nu_list[list]: List containing the SPP of all DOF in shape (r,Nk).
            sigmas[list]: List containing the individual terms for the SPP of all DOF.
            i[int]: ith index.
            j[int]: jth index.
            max_it[int]: Maximal amount of iterations.
            YSVD[bool]: Construct the Y from the SVD of x_ij. Default is False.
            BSVD[bool]: Construct the B from the SVD of x_ij. Default is False.
            
    [Returns]:
            [array]: Shape (r), the updated weights from the last iteration.
            [list]: List of the SPP with updated ith and jth elements.
            [list]: List of the sigmas with updated ith and jth elements.
            [list]: Shape (it), list of the RMSE for all iterations.''' 
    # run for each index

    it = 0
    # we dont use this error anyway
    errorsub = []
    # this is ugly but w/e, maybe make it dynamic later?
    err1 = errorVttsq(S_ij, x_ij)
    errorsub.append(err1)
    errorsub.append(0)        
    
    # get the SVD
    if YSVD == True or BSVD == True:
        
        U1, S1, Vh1 = np.linalg.svd(x_ij, full_matrices=False)
        # swap axes
        U2 = np.swapaxes(U1, 1, 2)
        # not really necessary
        S2 = S1
        Vh2 = np.swapaxes(Vh1, 1, 2)
        
        #find out how many singular vectors and values we need.
        SVrel = find_SVrel(S1)
        #print(SVrel)
    while abs(errorsub[-2]-errorsub[-1]) > errorsub[0]/100 and it < max_it:
    #while it < max_it:
        
        # get the new one-hole overlap matrix for i
        S_i = add_sigma(S_ij, sigmas, j)
        # get the one hole overlap of the SPP with the x_ij for i
        
        if BSVD == False:
            if YSVD == False:
                Y_i = get_ein_spec(x_ij, nu_list[j], 2)
            
            elif YSVD == True:
                Y_i = construct_YSVD(U1, S1, Vh1, nu_list[j], SVrel)
            
            # solve the linear equation
            x_i = solve_linearsub(S_ij, Y_i, S_i,)
            
        elif BSVD == True:
            B_i = construct_BSVD(S_ij, U1, S1, Vh1, nu_list[j], SVrel)
            x_i = solve_linearsub1(B_i, S_i)
            
        # normalize, get weights and new SPP
        weights, nu_list[i] = get_norm(x_i)
        sigmas = update_sigma(nu_list, sigmas, i)
        
        # do the same for j
        S_j = add_sigma(S_ij, sigmas, i)
        
        if BSVD == False:
            if YSVD == False:
                Y_j = get_ein_spec(x_ij, nu_list[i], 1)
            
            elif YSVD == True:
                # note that we change Vh and U in the function call
                Y_j = construct_YSVD(Vh2, S2, U2, nu_list[i], SVrel)
            
            x_j = solve_linearsub(S_ij, Y_j, S_j)
            
        elif BSVD == True:
            # note that we change Vh and U in the function call
            B_j = construct_BSVD(S_ij, Vh2, S2, U2, nu_list[i], SVrel)
            x_j = solve_linearsub1(B_j, S_j)

        weights, nu_list[j] = get_norm(x_j)
        sigmas = update_sigma(nu_list, sigmas, j)
        
        # get the error (actually we dont use it, so why should we?)
        err2 = error2VttVt(S_ij, x_ij, weights, nu_list[i], nu_list[j])
        err3 = errorVtsq(x_ij, weights, S_j, sigmas, j)
        errreg = errorreg(x_ij, S_j, nu_list[j])
        error = getrmsesub(err1, err2, err3, errreg)
        errorsub.append(error)
        
        it +=1
    #print(it)
    return weights, nu_list, sigmas# , errorsub


def get_ein_spec(x, nu, index):
    '''Function to get the einstein sum over a specified index.
    
    [Args]:
            x[array]: Input array to be contracted, shape (r,Ni,Nj).
            nu[array]: SPP to be contracted with, shape (r,Ni) or (r,Nj).
            index[int]: 1 or 2. Index so that the indices from x and nu are aligned.
            
    [Returns]:
            [array] Array of shape (r,r,Nj) for index = 1 or (r,r,Ni) for index = 2.'''
    
    # initialize y as copy of x
    y = x.copy()
    # if index is 1 we take the einsum wrt the first index
    
    if index == 1:
        y = np.einsum(y, [0, 1, 2], nu, [3, 1], [0, 3, 2])
                      
    # elif index is 2 we take the einsum wrt the second index
    elif index == 2:
        y = np.einsum(y, [0, 1, 2], nu, [3, 2], [0, 3, 1])
        
    # return y which is x contracted along either the first or second axis
    return y


def construct_YSVD(U, S, Vh, nu, SVrel):
    '''Use a SVD of x_ij for the construction of Y_ind for the subALS LES.
    
    [Args]:
            U[array]: (r,Ni,Ni) shaped U array.
            S[array]: (r,Nj) shaped array containing the singular values.
            Vh[array]: (r,Nj,Nj) shaped V array.
            nu[array]: SPP of shape (r,Ni or Nj) to create alpha.
            ind[int]: Indicates which index is considered, 1=j, 2=i)
            SVrel[int]: Number of singular values and vectors to be included.
            
    [Returns]:
            [array]: Y for subALS LES of shape (r,r,Ni or Nj).'''
    
    # check if passed value is bigger than the last index of the SVD S, if so just use all 
    # singular values
    if SVrel > S.shape[1]:
        SVrel = S.shape[1]
    #SVrel = S.shape[1]
    # get the needed objects by einstein summation
   
    # (r',s,ik'),(r,ik')->(r',r,s)
    alpha = np.einsum(Vh[:,:SVrel,:], [0,1,2], nu, [3,2], [0,3,1])
    # (r',s),(r',r,s),(r',ik,s)->(r',r,ik)
    Y = np.einsum(S[:,:SVrel], [0, 1], alpha, [0,2,1], U[:,:,:SVrel], [0,3,1], [0,2,3])        
    
    return Y


def construct_BSVD(S_ij, U, S, Vh, nu, SVrel):
    '''Construct the B for the subALS LES from the SVD of x_ij.
    
    [Args]:
            S_ij[array]: Two-hole overlap matrix of the SPP. (r,r')
            U[array]: Left-hand unitary matrix of SVD for x_ij. (r',ik,s)
            S[array]: Singular values from SVD for x_ij. (r',s)
            Vh[array]: Right-hand unitary matrix of SVD for x_ij. (ik',s,r')
            nu[array]: SPP to be included. (r,ik)
            ind[int]: Indicates which index is considered, 1=j, 2=i)
            SVrel[int]: Amount of singular values and vectors to be considered.
            
    [Returns]:
            [array]: Array of shape (r,ik) containing the B to be use in the subALS LES.'''
    
    
    if SVrel > S.shape[1]:
        SVrel = S.shape[1]
     
    # build the alpha by einsum over ik'
    # (r',s,ik'),(r,ik')->(r',r,s)
    alpha = np.einsum(Vh[:,:SVrel,:], [0,1,2], nu, [3,2], [0,3,1])
    # build the gamma by einsum over r'
    # (r,r'),(r',r,s),(r',s),(r',ik,s)->(r,ik,s)
    gamma = np.einsum(S_ij, [0,1], alpha,[1,0,2], S[:,:SVrel], [1,2], U[:,:,:SVrel], [1,3,2], [0,3,2])
    # build the B by einsum over s
    # (r,ik,s)->(r,ik)
    B = np.einsum(gamma,[0,1,2], [0,1])
           
    return B


def solve_linearsub(S_ij, Y, S_ind, prec = None):
    '''Function to solve the LES for the subALS functional.
    
    [Args]:
            S_ij[array]: Two-hole overlap matrix for the SPP of shape (r,r).
            Y[array]: One hole overlap of the SPP with the input x_ij of shape (r,r,Ni or Nj).
            S_ind[array]: One-hole overlap matrix for the SPP of shape (r,r).
            prec[float]: Precision to be used for regularization. Default is sqrt machine prec (~1E-8).
                        
    [Returns]:
            [array]: Result of the LES of shape (r, Ni or Nj).'''
    
    # set the regularization
    if prec == None:
        prec = np.sqrt(np.finfo(float).eps)
        
    # get the S matrix for the LES from the one-hole overlap matrix and the regularization
    S_in = np.zeros(S_ind.shape)
    S_in = S_ind + prec * np.identity(S_ij.shape[0])
    # get the 'b' by taking the einsum over the third and second index of y and the second index of the two-hole
    # overlap matrix
    b = np.zeros((S_ind.shape[0], Y.shape[2]))
    b = np.einsum(Y, [0,1,2], S_ij, [1,0], [1,2])

    #print(np.allclose(b, b1))
    return np.linalg.solve(S_in, b)


def solve_linearsub1(B_ind, S_ind, prec = None):
    '''Function to solve the LES for the subALS functional.
    
    [Args]:
            B_ind[array]: B for subALS LES (r,ik).
            S_ind[array]: One-hole overlap matrix for the SPP of shape (r,r).
            prec[float]: Precision to be used for regularization. Default is sqrt machine prec (~1E-8).
            
    [Returns]:
            [array]: Result of the LES of shape (r, Ni or Nj).'''
    
    # set the regularization
    if prec == None:
        prec = np.sqrt(np.finfo(float).eps)
        
    # get the S matrix for the LES from the one-hole overlap matrix and the regularization
    S_in = np.zeros(S_ind.shape)
    S_in = S_ind + prec * np.identity(S_ind.shape[0])
    # get the 'b' by taking the einsum over the third and second index of y and the second index of the two-hole
    # overlap matrix

    #print(np.allclose(b, b1))
    return np.linalg.solve(S_in, B_ind)


def find_SVrel(S, thresh=1E-4):
    '''Function to find the number of relevant singular values and vectors for a given
    SVD by comparing the sum of the squared neglected singular values against a threshhold.
    
    [Args]:
            S[array]: Array of shape (r,s) containing the singular values along the second axis.
            thresh[float]: Threshold to signal convergence, default is 1E-4. This value was chosen
                    by running multiple tests and adjusting until no inconsistant behaviour was 
                    observed.
            
    [Returns]:
            [int]: Number of relevant singular values and vectors wrt threshold.'''
    
    SVrel = 1
    error = max([(S[i,SVrel:]**2).sum() for i in range(S.shape[0])])
    while error > thresh:
        if SVrel + 1 <= S.shape[1]:
            SVrel += 1
            error = max([(S[i,SVrel:]**2).sum() for i in range(S.shape[0])])
        # make sure we cant get out of bounds
        elif SVrel + 1 > S.shape[1]:
            break
    return SVrel
        
        
def errorVttsq(S_ij, x_ij):
    '''Function to get the constant error (a² term) for the subALS functional.
    
    [Args]:
            S_ij[array]: Two-hole overlap matrix of the SPP of shape (r,r).
            x_ij[array]: Solution to the ALS LES of shape (r,Ni,Nj).
            
     [Returns]:
             [float]: Constant a² error in au² divided by r*Ni*Nj.'''
    
    x_ijresh = x_ij.reshape(x_ij.shape[0], x_ij.shape[1]*x_ij.shape[2])
    return (S_ij*(x_ijresh @ x_ijresh.T)).sum()/x_ij.size


def error2VttVt( S_ij, x_ij, weights, nu_i, nu_j):
    '''Function to get the 2ab term of the error.
    
    [Args]:
            S_ij[array]: Two-hole overlap matrix of the SPP of shape (r,r).
            x_ij[array]: Solution to the ALS LES of shape (r,Ni,Nj).
            weights[array]: Weights of the SPP of shape (r).
            nu_i[array]: ith SPP of shape (r, Ni).
            nu_j[array]: jth SPP of shape (r, Nj).
            
    [Returns]:
            [float]: 2ab error in au², divided by r*Ni*Nj.'''
    
    # we can exploit that we know exactly what our input looks like
    outer_nu = np.einsum(x_ij, [0, 1, 2], nu_i, [3, 1], nu_j, [3, 2], [0,3]) # use einstein summation convention
    outerweight = outer_nu * weights # get the weights in there
    
    return 2 *( S_ij * outerweight).sum()/x_ij.size


def errorVtsq(x_ij, weights, S_ind, sigmas, ind):
    '''Funtion to get the b² term of the error.
    
    [Args]:
            weights[array]: Weights of the SPP of shape (r).
            S_ind[array]: One-hole overlap matrix for the SPP of shape (r,r).
            sigmas[list]: List containing the individual terms for the SPP of all DOF.
            ind[int]: 1 or 2. Index so that the indices from x and nu are aligned.
            
    [Returns]:
            [float]: b² error in au² devided by r*Ni*Nj.'''
    
    S_tot = S_ind * sigmas[ind]
    outerweight = np.outer(weights, weights)
    return (S_tot*outerweight).sum()/x_ij.size


def errorreg(x_ij, S_ind, nu_ind, prec=None):
    '''Function to get the error arrising from the regularization.
    
    [Args]:
            x_ij[array]: Solution to the ALS LES of shape (r,Ni,Nj).
            S_ind[array]: One-hole overlap matrix for the SPP of shape (r,r).
            nu_ind[array]: SPP corresponding to the hole of S_ind.
            prec[float]: Regularization parameter, default is sqrt of float machine precision (~1E-8).
    
    [Returns]:
            [float]: Error arrising from the regularization.'''
    
    # set the regularization
    if prec == None:
        prec = np.sqrt(np.finfo(float).eps)
    
    summ = np.einsum(S_ind, [0,0], nu_ind**2, [0, 1], [0,1])
    return (prec*summ).sum()/x_ij.size


def getrmsesub(err1, err2, err3, errreg):
    '''Function to get the RMSE for the subALS functional.
    
    [Args]:
            err1[float]: a² error.
            err2[float]: 2ab error.
            err3[float]: b² error.
            
    [Returns]:
            [float]: RMSE in cm-1.'''
    
    return (np.sqrt(err1-err2+err3+errreg))*au2ic #au2ic is defined in ALS1D.py and ALSclass.py (au2ic = 219474.63137)


def error1man(S_ij, x_ij):
    '''Test function a² error.'''
    err = np.zeros((S_ij.shape[0], S_ij.shape[1]))
    
    for r in range(S_ij.shape[0]):
        for rdash in range(S_ij.shape[1]):
            for ik in range(x_ij.shape[1]):
                for ij in range(x_ij.shape[2]):
                    err[r, rdash] += S_ij[r, rdash]*x_ij[r, ik, ij]*x_ij[rdash, ik, ij]
    return err.mean()

def error2man(S_ij, x_ij, weights, nu_i, nu_j):
    '''Test function 2ab error.'''
    err = np.zeros((S_ij.shape[0], S_ij.shape[1]))
    
    for r in range(S_ij.shape[0]):
        for rdash in range(S_ij.shape[1]):
            for ik in range(nu_i.shape[1]):
                for ij in range(nu_j.shape[1]):
                    err[r, rdash] += S_ij[r, rdash]*x_ij[rdash, ik, ij]* nu_i[r,ik]* nu_j[r,ij]* weights[r]
    return (2*err.sum())/x_ij.size

def error3man(weights, S_ind, nu_ind):
    '''Test function b² error.'''
    err = np.zeros((S_ind.shape[0], S_ind.shape[1]))
    for r in range(S_ind.shape[0]):
        for rdash in range(S_ind.shape[1]):
            for ik in range(nu_ind.shape[1]):
                err[r, rdash] += weights[r]*weights[rdash]*nu_ind[r, ik]* nu_ind[rdash, ik]* S_ind[r, rdash]
    return err.mean()