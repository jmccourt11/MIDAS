import numpy as np
import blosc2 #compressed file format is *.blosc
import zipfile #to read *zip files
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.signal import convolve2d as conv2
from skimage.transform import resize
import sys
#import cupy as cp
#from cupyx.scipy.signal import convolve2d as conv2cp

save=False
plot=True
cuda=False


#index=int(sys.argv[1])

def add_value(dict_obj, key, value):
    ''' Adds a key-value pair to the dictionary.
        If the key already exists in the dictionary, 
        it will associate multiple values with that 
        key instead of overwritting its value'''
    if key not in dict_obj:
        dict_obj[key] = value
    elif isinstance(dict_obj[key], list):
        dict_obj[key].append(value)
    else:
        dict_obj[key] = [dict_obj[key], value]

filename='/home/beams/PTYCHOSAXS/opt/MIDAS/FF_HEDM/ExampleCindy/Parameters_Test.txt'
file1 = open(filename, 'r')
print(f'filename: {filename}\n')

Lines = file1.readlines()
params={}
for line in Lines:
    print(line.split()[0])
    add_value(params, line.split()[0], line.split()[1:])

print(f'Scan params: {params}\n')



#archive = zipfile.ZipFile('/home/beams/B304014/opt/MIDAS/FF_HEDM/Example/Au_FF_SAXS_gen_000001.ge3.zip','r')
#archive = zipfile.ZipFile('/home/beams/B304014/opt/MIDAS/FF_HEDM/Example/Quartz_FF_SAXS_Gen_000001.ge3.zip','r')
archive = zipfile.ZipFile('/home/beams/PTYCHOSAXS/opt/MIDAS/FF_HEDM/ExampleCindy/Au_Nano_Test_ff_scanNr_0.zip','r')


frame_total=np.zeros([int(params['NrPixels'][0]),int(params['NrPixels'][0])])
deg_step_size=abs(float(params['OmegaStep'][0]))
for frameNr in tqdm(range(int(360/deg_step_size))): # 1440/360deg = 0.25 deg step
#for frameNr in tqdm(range(0,360)): # 1440/360deg = 0.25 deg step
    #fn = f'{frameNr}.blosc'
    fn = f'exchange/data/{frameNr}.0.0'
    data = archive.read(fn)
    frame = np.frombuffer(memoryview(blosc2.decompress(bytearray(data))),dtype=np.uint16)
    nS = np.sum(frame>0) #check which pixels are greater than 0
    frame_total+=frame.reshape(int(params['NrPixels'][0]),int(params['NrPixels'][0]))


#with cupy
if cuda:
    with cp.cuda.Device(1):
        psf=cp.load('/home/beams/B304014/ptychosaxs/NN/probe_FT.npy')
        idealDP=resize(frame_total[32:-32,32:-32],(256,256),preserve_range=True, anti_aliasing=True)
        idealDP=cp.asarray(idealDP)
        convDP=conv2cp(idealDP,psf,'same', boundary='symm')
        idealDP=idealDP.get()
        convDP=convDP.get()
        if save:
            filename='/mnt/micdata2/12IDC/ptychosaxs/data/midas_DPs/quartz_output_{:05d}.npz'
            np.savez(filename.format(index),convDP=convDP,idealDP=idealDP)
            print(f'File saved to {filename.format(index)}.')
        
#with numpy
else:
    #psf=np.load('/home/beams/B304014/ptychosaxs/NN/probe_FT.npy')
    #idealDP=resize(frame_total[32:-32,32:-32],(256,256),preserve_range=True, anti_aliasing=True)
    #convDP=conv2(idealDP,psf,'same', boundary='symm')
    idealDP=frame_total
    if plot:
        #fig,ax=plt.subplots(1,3,layout='constrained')
        fig,ax=plt.subplots(1,2,layout='constrained')
        ax[0].imshow(frame_total,cmap='jet')
        ax[1].imshow(idealDP)
        #ax[2].imshow(np.abs(convDP))
        plt.show()

    if save:
        filename='/mnt/micdata2/12IDC/ptychosaxs/data/midas_DPs/2/in_output_ideal_{:05d}.npz'
        #np.savez(filename.format(index),convDP=convDP,idealDP=idealDP)
        np.savez(filename.format(index),idealDP=idealDP)
        print(f'File saved to {filename.format(index)}.')
        


