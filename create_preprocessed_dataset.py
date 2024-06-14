# taken from https://github.com/MansoorehMontazerin/Vision-Transformer_EMG

import numpy as np
import pickle
import os
from scipy import signal
from scipy import ndimage
import scipy
import sys

# Defining the preprocessing functions
def low_pass_filter(data, N, f, fs= 2048):
    fnew=[x / (fs/2) for x in f]
    data = np.abs(data)
    b, a = signal.butter(N=N, Wn = fnew, btype="lowpass")
    output = signal.filtfilt(b, a, data,  axis=0)    
    return output

def encode_mu_law(x,mu):
    mu = mu-1
    fx = np.sign(x)*np.log(1+mu*np.abs(x))/np.log(1+mu)
    return (fx+1)/2*mu+0.5


def loaddata_filt(idx):
    xf=[]
    xe=[]
    yf=[]
   
    num_gst=66
    num_rep=5
   
    for gst in range(0,num_gst):
        for rep in range(0,num_rep):
            if not(rep==1 and gst==33): 
                if not(rep==4 and gst==65): # only for subject 13
                
                    with open('../final_dataset/subj_{}/class{}{}/repetition{}/flexors.pkl'\
                              .format(idx,(gst+1)//10,(gst+1)%10,rep+1), 'rb') as f:
                        emg_sigf_final=pickle.load(f)
                    
                    with open('../final_dataset/subj_{}/class{}{}/repetition{}/extensors.pkl'\
                              .format(idx,(gst+1)//10,(gst+1)%10,rep+1), 'rb') as f:
                        emg_sige_final=pickle.load(f)
                
                    emg_sigf_filtered=low_pass_filter(emg_sigf_final,N=1,f=[1])
                    emg_sigf_clipped=np.clip(emg_sigf_filtered,0,0.1)
                    emg_sigf_norm = encode_mu_law(emg_sigf_clipped, mu=8)-4
              
                    emg_sige_filtered=low_pass_filter(emg_sige_final,N=1,f=[1])
                    emg_sige_clipped=np.clip(emg_sige_filtered,0,0.1)                                         
                    emg_sige_norm = encode_mu_law(emg_sige_clipped, mu=8)-4
                
                    emg_sigt_norm=np.concatenate((emg_sigf_norm,emg_sige_norm),axis=2)[:,:,:]
                    xf.append(emg_sigt_norm)
                    yf.append([idx, gst, rep+1]) # subject, gesture, repetition
                
                    
    return xf, yf

if __name__ == "__main__":
    idx = sys.argv[1]    
    data, labels = loaddata_filt(idx)
        
    with open('preprocessed/subj' + str(idx) + '_samples', 'wb') as fp:
        pickle.dump(data, fp)
    #np.save("preprocessed/subj" + str(idx) + "_samples", data)
    np.save("preprocessed/subj" + str(idx) + "_labels", labels)
