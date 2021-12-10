import numpy as np

def track_progress(task, progress, status=''):    
    '''Function to display the progress of a task.
    
    [Args]:
            task[str]: What task is beeing tracked.
            progress[float]: Value between 0 and 1.
            status[str]: Status to be printed.'''
    
    # you can change the length of the 
    length = 20
    if progress < 0:
        status = 'Received negative progress value...'
    
    elif progress >= 1:
        status += 'Task finished.'
        
    block = int(round(length*progress))
    output = '\r'+task+': {} {:.0f}% {}'.format('■'*block+'□'*(length-block), progress*100, status)
    print(output, end='')
    
    
def init_tracker(max_iter):
    '''Function to initialize the tracker for the ALSCPD class run-Routines.
    
    [Args]:
            max_iter[int]: Maximum amount of iterations possible.
            
    [Returns]:
            [array]: Array of length 100 containing the iteration value for each percent of progress.
            [float]: Value between 0 and 1 representing the current progress.
            [int]: Counter keeping track of the currently relevant element from the array above.
            [int]: Curently relevant element from the array above.'''
     
    perc=max_iter/100
    perc_iter=np.arange(1,101,dtype=int)*perc
    perc_cur=0
    track = 0
    cur_perc=perc_iter[track]
    return perc_iter, perc_cur, track, cur_perc


def keep_track(task, error, perc_iter, perc_cur, track, cur_perc):
    '''Function to update the tracker for the ALSCPD class run-Routines.'''
    
    track += 1
    perc_cur += 0.01
    track_progress(task, perc_cur, error)
    if track != 100:
        cur_perc = perc_iter[track]
    
    return perc_iter, perc_cur, track, cur_perc