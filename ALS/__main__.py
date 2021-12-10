from ALS.ALSclass import *
from ALS.potentials import *
from ALS.dvr import *
from pathlib import Path
import os
import numpy as np
import sys
import shutil

# global defaults

__author__ = "Christian Delavier"
__copyright__ = "Theoretical Chemistry, University of Heidelberg"
__version__ = "0.1"
__maintainer__ = "Christian Delavier"
__email__ = "delavier@stud.uni-heidelberg.de"
__status__ = "Development"

def versionInfo(progname="ALSCPD",
                version="Unknown",
                status="Unknown",
                author="Unknown",
                copyright="Unknown",
                maintainer=None,
                support="N/A"):
    """
    Print version info

    versionInfo(progname="Unknown",
            version="Unknown",
            status="Unknown",
            author="Unknown",
            copyright="Unknown",
            maintainer="Unknown",
            support="N/A")

    """

    message = "-"*70 + "\n\n"
    message += " ****** Source code info ******\n\n"
    message += " Program....... " + str(progname) + "\n"
    message += " Version....... " + str(version) + "\n"
    message += " Status........ " + str(status) + "\n"
    message += " Author........ " + str(author) + "\n"
    message += " Copyright..... " + str(copyright) + "\n"
    message += " Maintainer.... " + str(maintainer) + "\n"
    message += " Support....... " + str(support) + "\n\n"
    message += "-"*70 + "\n"
    sys.stdout.write(message)


def work(inputfile):
    
    # get all the information from the input file
    grids = []
    job = []
    filename = ''
    sampl = ''
    maxiter = 0
    rank = 0
    nsmpl = 0
    func = None
    tracker = True
    pot = 0
    thresh = 10000
    reset = False
    plot = False
    
    with open('{}'.format(inputfile), 'r') as inp:
        
        readgrids = False
        work = False
        
        for line in inp:
            
            if line.split():
                split = line.split()
                if split[0] != '#':
       
       # Variablen 
       #########################################################################             
                        
                    if split[0] == 'name': filename = split[2]
                    elif split[0] == 'maxiter': maxiter = int(split[2])
                    elif split[0] == 'rank': rank = int(split[2])
                    elif split[0] == 'nsmpl': nsmpl = int(split[2])
                    elif split[0] == 'potential': func = split[2]
                    elif split[0] == 'load' and split[2] != 'False': 
                        print('Loading potential from ' + split[2] + '.')
                        print('Keep in mind that that the parameters for the grids need to'\
                        +' correspond to the ones used to compute the stored potential.')
                        pot = grab_full(split[2])
                    elif split[0] == 'sampling': sampl = split[2]
                    elif split[0] == 'thresh': thresh = float(split[2])
                    elif split[0] == 'tracker': tracker = bool(split[2])
                    elif split[0] == 'reset':
                        if split[2] == 'True':
                            reset = True
                        elif split[2] == 'False':
                            reset = False
                    elif split[0] == 'plot':
                        if split[2] == 'True':
                            plot = True
                        elif split[2] == 'False':
                            plot = False
       # Grids
       ##########################################################################                     
                            
                    if line.strip() == 'PBASIS-SECTION':
                        readgrids = True
                        print('Reading grids from file {}.'.format(inputfile))
                        print('*'*90)
                        print('Label   DVR    N   Parameters')
                    elif line.strip() == 'end-pbasis-section':
                        readgrids = False
                        print('# Finished reading grids.')
                        
                    if readgrids and line.strip() != 'PBASIS-SECTION':
                        print(line.strip())
                        if split[1] == 'sin':
                            grids.append(sinDVR(int(split[2]), xi=float(split[3]), xf=float(split[4])).grid)
                        elif split[1] == 'ho':
                            grids.append(hoDVR(int(split[2]), xi=float(split[3]), xf=float(split[4])).grid)
       # Jobs
       ############################################################################                 
                        
                    if line.strip() == 'ALS-SECTION':
                        work = True
                    elif line.strip() == 'end-als-section':
                        work = False
                            
                    if work and line.strip() != 'ALS-SECTION' and split[0] != 'reset' and split[0] != 'plot':
                        job.append(line.strip())
                                            
       # Start
       #############################################################################         
    return grids, job, filename, sampl, maxiter, rank, nsmpl, thresh, tracker, func, pot, reset, plot


if __name__ == "__main__":
    
    versionInfo(progname="ALSCPD", version=__version__, status=__status__, author=__author__,\
               copyright=__copyright__, maintainer=__maintainer__, support=__email__)
    
    initialized = False
    Obj = False
    INPUT = None
    # read in the input file from the current directory
    files = os.listdir('.')
    file = [f for f in files if '.inp' in f]
    # if there is only one we can just try to start the job
    if len(file) == 1:
        try:
            grids, job, filename, sampl, maxiter, rank, nsmpl, thresh,\
            tracker, func, pot, reset, plot = work(file[0])
            initialized = True
            INPUT = file[0]
        except:
            print('Something went wrong while reading from {}.'.format(file[0]))
        
    # if there are multiple we ask which we should run, maybe inconvenient for
    # automated processing? -> here every job should have its own directory I guess
    elif len(file) != 1:
        print('Multiple Input files available:')
        for i, elem in enumerate(file):
            print('{}: {}'.format(i,elem))
        idx = input('Choose input by index or abort with n:')
        if idx != 'n':
            try:
                grids, job, filename, sampl, maxiter, rank, nsmpl, thresh,\
                tracker, func, pot, reset, plot = work(file[int(idx)])
                initialized = True
                INPUT = file[int(idx)]
            except ValueError:
                print('Something went wrong while reading from {}.'.format(file[int(idx)]))
        elif idx == 'n':
            print('Aborting!')
            sys.exit()     
    # we have the grids, if we didn't read the potential from file we can calculate it here
    # I'll just assume we only want to work with h2o and hfco for now, as the zundel potential
    # would need additional restructuring of the class to only compare to the potential on
    # certain sampling points -> Update the error calculation for this later if neccessary
    # if I put this online the whole potentials should be removed I guess, or just leave
    # H2O as it is pure .py anyways
    # HFCO has to be compiled from f2py on each machine
    if initialized == True:
        
        if func == 'h2o' and type(pot) != np.ndarray:
            print('Building h2o potential with shape {}.'.format([len(g) for g in grids]))
            grids[2] = np.arccos(grids[2])            
            pot = geth2o(*grids)
        
        elif func == 'hfco' and type(pot) != np.ndarray:
            print('Building hfco potential with shape {}.'.format([len(g) for g in grids]))            
            pot = hfco(*grids)
        
        elif type(pot) == np.ndarray:
            pot = pot.reshape(*[len(g) for g in grids])
            print('Loaded potential of shape {}'.format(pot.shape))
    # if we have the potential we can now initialize the object
    if '1DMCALSCPD' in job or '2DMCALSCPD' in job:
        if func == 'h2o':
            if sampl != '':
                try:
                    Object = ALSCPD(filename, pot, rank, func1D=h2o1D, func2D=h2o2D,\
                                    grids=grids, nsmpl=nsmpl, presmpl=sampl)
                    Obj = True
                except FileNotFoundError:
                    print('Presampling file could not be found at path {}.'.format(sampl))
                    
            elif sampl == '':
                Object = ALSCPD(filename, pot, rank, func1D=h2o1D, func2D=h2o2D,\
                                grids=grids, nsmpl=nsmpl)
                Obj = True
        if func == 'hfco':
            if sampl != '':
                try:
                    Object = ALSCPD(filename, pot, rank, func1D=hfco, func2D=hfco,\
                                    grids=grids, nsmpl=nsmpl, presmpl=sampl)
                    Obj = True
                except FileNotFoundError:
                    print('Presampling file could not be found.')
                    
            elif sampl == '':
                Object = ALSCPD(filename, pot, rank, func1D=hfco, func2D=hfco,\
                                grids=grids, nsmpl=nsmpl)            
                Obj = True
    # if there is no monte carlo queued we can just initialize the object with the normal parameters
    else:
        Object = ALSCPD(filename, pot, rank)
        Obj = True
    
    if Obj == True:
        print('*'*90)
        print('Running jobs: {}'.format(job))
        for i, elem in enumerate(job):
            
            if reset == True:
                Object.filename=filename+'_{}'.format(i)                
                Object.copy_reset(Object)
            
            if elem == '1DALSCPD':
                Object.run(maxiter, thresh, tracker=tracker)
        
            elif elem == '2DALSCPD':
                Object.run2D(maxiter, thresh, tracker=tracker)
             
            elif elem == '2DALSCPD/SVD':
                Object.run2D(maxiter, thresh, YSVD=True, tracker=tracker)
                
            elif elem == '1DMCALSCPD':
                Object.runMC(maxiter, thresh, tracker=tracker)
                
            elif elem == '2DMCALSCPD':
                Object.run2DMC(maxiter, thresh, tracker=tracker)
                
            path = "{}_store".format(Object.filename)
            try:
                if os.path.exists(path):
                    shutil.rmtree(path)
                os.mkdir(path)
                DATAPATH = path+"/CPDDATA"
                os.mkdir(DATAPATH)
                weights, SPP = Object.get_current()
                for i, spp in enumerate(SPP):
                    np.save(DATAPATH+"/SPP_{}".format(i), spp)
                np.save(DATAPATH+"/weights", weights)
                shutil.copy("{}".format(Object.filename), "{}/".format(path))
                shutil.copy("{}".format(INPUT), "{}/".format(path))
            except OSError:
                print("Could not create storage directory at {}.".format(path))

            if plot == True:
                Object.plot_error(show=False)
                shutil.move("{}.pdf".format(Object.filename), "{}/".format(path))

        print('*'*90)
