#%%
import h5py
import numpy as np

def load_wavelets_and_outputs(path_in):
    '''loads wavelets and fourier freqs from input path and stores them as numpy arrays'''
    hdf5_file = h5py.File(path_in, mode='r')
    # print(hdf5_file['/inputs/'].keys())
    wavelets = np.array(hdf5_file['inputs/wavelets'])
    freq_decoder= np.array(hdf5_file['inputs/fourier_frequencies'])
    print('wavelets dimensionality:' + str(wavelets.shape))
    return(wavelets,freq_decoder)



path_in='processed_R2478.h5'
wavelets,freq_decoder=load_wavelets(path_in)







# %%
