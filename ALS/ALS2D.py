from ALS.ALS1D import *

'''
Contains the components for the 2D ALSCPD Algorithm aswell as some basic functions for
the ALSCPD class.
'''

def assemble_S2D(sigma_list, hole1, hole2):
    '''Function to assemble the overlap matrix from a given set of sigmas neglecting two sigmas.
    
    [Args]:
            sigma_list[list]: List containing the individual terms for the SPP of all DOF.
            hole1[Int]: Index of the first term to be neglected.
            hole2[Int]: Index of the second term to be neglected.
            
    [Returns]:
            [array]: Array containing the two-hole overlap matrix for a given set of SPP.'''
    
    S = np.ones(sigma_list[0].shape)
    for i, sigma in enumerate(sigma_list):
        if i != hole1 and i != hole2:
            S = S * sigma
            
    return S


def add_sigma(S_ij, sigma_list, j):
    '''Function to implicitly add a sigma of certain index from the complete list to a given two-hole overlap
    matrix effectively yielding a one-hole overlap matrix.
    
    [Args]:
            S[array]: Two-hole overlap matrix of shape (r,r).
            sigma_list[list]: List containing the individual terms for the SPP of all DOF.
            j[int]: Index of sigma to be multiplied into the two-hole overlap matrix.
            
    [Returns]:
            [array]: One-hole overlap matrix of shape (r,r).'''
    
    return S_ij*sigma_list[j]


def get_b_ein2D(V, nu_r, hole1, hole2):   
    '''Function to iteratively create input for the np.einsum routine to contract a given tensor
    with given set of SPP for all DOF except two indicated.
    
    [Args]:
            V[array]: The exact tensor to be contracted with shape (N0,N1,N2,...,Nf).
            nu_r[list]: List containing the SPP of all DOF in shape (r,Nk).
            hole1[int]: Index of the first SPP to be neglected.
            hole2[int]: Index of the second SPP to be neglected.
            
    [Returns]:
            [Array]: Contracted tensor in shape (r,N[hole1],N[hole2]).
            
    -conceptualized and tested assuming hole1 < hole2'''
    
    # start with a copy of the potential
    b_ij = V.copy()
    # just set the reference to something which can never interfere
    ref = np.ndim(V)
    # get the counter to go though the nu backwards
    it = len(nu_r)-1
    # keep the dimension of the tensor after the current iteration in mind
    next_dim = np.ndim(V)
    # initialize the shape of the tensor, this will always look the same
    oldshape = np.arange(0, np.ndim(V)).tolist()
    
    for nu in nu_r:
        # if we are at one of the two hole indices we skip
        if it != hole1 and it != hole2:
            # if it > 1 one of the following cases occured:
            if it > 1:
                newshape = np.arange(0, next_dim).tolist()
                newshape[it] = ref
               
                # 1: We are not finished and have passed the smaller hole index, in this case we must
                # have passed both indices already and set the last elem in the list to the bigger index,
                # and the second to last to the smaller index
                if it < hole1:
                    newshape[-1] = hole2
                    newshape[-2] = hole1                
                # 2: We are not finished and have passed the bigger hole index, now it is the last
                # elem in the list 
                elif it < hole2:
                    newshape[-1] = hole2
                # 3: We are in the last iteration if the next_dim will be three and the indices which we
                # exclude are the first two, arange output as (r, N1, N2)
                elif next_dim == 3:
                    newshape = np.arange(0, next_dim).tolist()
                    newshape[0] = ref
                    newshape[1] = hole1
                    newshape[2] = hole2
                    
            # If it == 0 we must have passed both indices and be in the last iteration, arange output
            # as (r, N1, N2)
            elif it == 0:
                newshape = np.arange(0, next_dim).tolist()
                newshape[0] = ref
                newshape[1] = hole1
                newshape[2] = hole2
            
            # If it == 1 one of the following cases occured:
            elif it == 1:
                # the smaller index is 0, we are thus in the last iteration and arange output as 
                # (r, N1, N2)
                if hole1 == 0:
                    newshape = np.arange(0, next_dim).tolist()
                    newshape[0] = ref
                    newshape[1] = hole1
                    newshape[2] = hole2
                # We have skipped the two indices already and are in the second to last iteration
                elif hole1 != 0:
                    newshape = np.arange(0, next_dim).tolist()
                    newshape[it] = ref
                
                    if it < hole1:
                        newshape[-1] = hole2
                        newshape[-2] = hole1
                    
                    elif it < hole2:
                        newshape[-1] = hole2
            
            #print("""
            #{}, [{},{}] -> {}
            #iter: {}
            #Next Dim: {}""".format(oldshape, ref, it, newshape, it, next_dim))
            
            # do the contraction with the aranged input
            b_ij = np.einsum(b_ij, oldshape, nu_r[it], [ref, it], newshape)
            # the current shape of the tensor is now changed, update it
            oldshape = newshape
            # lower the dimensionality for the next iteration
            next_dim -= 1
        #print('Skipped, it = {}, it-=1'.format(it))
        # decrement counter
        it -= 1

    return b_ij


def solve_linear2D(S_ij, b_ij, prec = None):
    '''Function to formally solve the linear equation in 2D ALS. The two-hole overlap with the exact tensor
    in shape (r, Ni, Nj) is flattened to shape (r, Ni*Nj), the equation is solved and the result
    reshaped to(r, Ni, Nj).
    
    [Args]:
            S_ij[array]: Array containing the two-hole overlap matrix of shape (r,r).
            b_ij[array]: Array containing the two-hole overlap with the exact tensor of shape (r, Ni, Nj).
            prec[float]: Gives the epsilon for the regularization, standard is root of machine precision for
                    float (~1E-8).
                    
    [Returns]:
            [array]: Solution of shape (r, Ni, Nj)'''
    
    if prec == None:
        prec = np.sqrt(np.finfo(float).eps)
    
    S_in = np.zeros(S_ij.shape)
    S_in = S_ij + prec*np.identity(S_ij.shape[0])
    
    #b_resh = np.zeros((b_ij.shape[0], b_ij.shape[1]*b_ij.shape[2]))
    b_resh = b_ij.reshape(b_ij.shape[0], b_ij.shape[1]*b_ij.shape[2], order='C')
    
    x_ij = np.linalg.solve(S_in, b_resh)
    x_ij = x_ij.reshape(b_ij.shape[0], b_ij.shape[1], b_ij.shape[2], order='C')
    
    return x_ij


# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# Not used

def reconstruct(x_ij, SV_error=None):
    '''Function to recover the two SPP and the weights from the solution to the 2D AlS LSE.
    It is assumed that only the first singular value and vectors are important for now.
    
    [Args]:
            x_ij[array]: Array of shape (r, Ni, Nj) containing the formal solution to the 2D ALS LSE.
            SV_error[float]: Not used at the moment.
            
    [Returns]:
            [array]: (r) shaped array containing the new weights (the first singular value along each r coordinate).
            [array]: (r,Ni) shaped array containing the new normalized SPP for the ith DOF.
            [array]: (r,Nj) shaped array containing the new normalized SPP for the jth DOF.'''
    
    nu_i = np.zeros((x_ij.shape[0], x_ij.shape[1]))
    nu_j = np.zeros((x_ij.shape[0], x_ij.shape[2]))
    new_cr = np.zeros(x_ij.shape[0])
    
    #check_grow = True
    #with open('svd_monitor', 'a') as file:
        #file.write('Solving new SVD. \n')
        
    for i, matrix in enumerate(x_ij):
        U, S, Vh = np.linalg.svd(matrix)
        #if not np.isclose(S[0]-S[1], S[0], atol=S[0]*SV_error):
            #check_grow = False
        #print(S.shape)
        #with open('svd_monitor', 'a') as file:
            #file.write('Solving {}th matrix. \n'.format(i))
            #file.write('{} \n'.format(S[:10]))
            #file.write('-'*70)
            #file.write('\n')
        new_cr[i] = S[0]
        nu_i[i, :] = U[:, 0]
        nu_j[i, :] = Vh[0, :]
    #print(new_cr)
    return new_cr, nu_i, nu_j#, check_grow


def update2D(v_ex, nu_r, sigmas, i, j, SV_error=None):
    '''Function to update the weights, the SPP of the ith and the jth DOF.
    
    [Args]:
            v_ex[array]: (N0, N1,...,Nf) shaped array containing the exact tensor.
            nu_r[list]: List containing the SPP for all DOF in shape (r,N).
            sigmas[list]: List containing the sigmas for all DOF in shape (r,r).
            i[int]: First hole index.
            j[int]: Second hole index.
            SV_error[float]: Not used at the moment.
            
    [Returns]:
            [array]: (r) shaped array containing the new weights (the first singular value along each r coordinate).
            [array]: (r,Ni) shaped array containing the new normalized SPP for the ith DOF.
            [array]: (r,Nj) shaped array containing the new normalized SPP for the jth DOF.'''
    
    #with open('svd_monitor', 'a') as file:
        #file.write('*'*100)
        #file.write('\n')
        #file.write('Combination {}, {}: \n'.format(i,j))
    # build the two-hole overlap matrix
    S_ij = assemble_S2D(sigmas, i, j)
    # build the two-hole overlap with the exact tensor
    b_ij = get_b_ein2D(v_ex, nu_r, i, j)
    # formally solve the linear equation -> reshape appropriate?
    x_ij = solve_linear2D(S_ij, b_ij, prec=None)
    # get new stuff
    #new_weights, nu_i, nu_j, check = reconstruct(x_ij, SV_error)
    new_weights, nu_i, nu_j = reconstruct(x_ij, SV_error)
    
    # the vectors coming from svd should be normalized already since U and Vh are unitary
    #_, nu_i = get_norm(nu_i)
    #_, nu_j = get_norm(nu_j)
    
    return new_weights, nu_i, nu_j#, check