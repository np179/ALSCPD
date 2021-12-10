from ALS.h2o import *
#from ALS.hfco import potentialv 
import numpy as np


def grab_full(filename):
    '''Function to load already computed potentials from directory'''
    return np.fromfile('{}'.format(filename), dtype=float)

    
def vectorize_zundel(coordl):
    '''function to get cuts through the zundel potential'''
    # get the location and the length of the grids that are not fixed
    tmp = []
    idx = []
    for i, elem in enumerate(coordl):
        try: 
            tmp.append(np.arange(len(elem)))
            idx.append(i)
        except TypeError:
            # if this throws a TypeError the element is a fixed point (or something weird is going on),
            # in this case we just skip
            pass
    #for i, elem in enumerate(idx):
        #print(coordl[elem][tmp[i][:]])
    out = np.zeros((len(tmp[0]), len(tmp[1])))
    # this is pretty slow, can I not do this faster?
    for j in range(len(tmp[0])):
        for k in range(len(tmp[1])):
            zundel_inp = coordl.copy()
            zundel_inp[idx[0]] = coordl[idx[0]][j]
            zundel_inp[idx[1]] = coordl[idx[1]][k]
            out[j,k] = zundel.zundel(zundel_inp)
    #print(out.shape)
    return out


def wrapper_zundel(grid1, grid2, grid3, grid4, grid5, grid6, grid7, grid8,\
                  grid9, grid10, grid11, grid12, grid13, grid14, grid15):
    '''wrapper for the above routine'''
    return vectorize_zundel([grid1, grid2, grid3, grid4, grid5, grid6, grid7, grid8,\
                  grid9, grid10, grid11, grid12, grid13, grid14, grid15])


def geth2o(x,y,z):
    X,Y,Z = np.meshgrid(x,y,z,indexing='ij')
    return PJT2(X,Y,Z)

def h2o1D(x,y,z):
    '''Call from h2o for 1D.'''
    return PJT2(x,y,z)

def h2o2D(x,y,z):
    '''Call from h2o for 2D.'''
    return PJT2_2D(x,y,z)

def hfco(x,y,z,w1,w2,w3):
    '''Call from hfco.'''
    V = np.ascontiguousarray(potentialv(x,y,z,w1,w2,w3))
    return V