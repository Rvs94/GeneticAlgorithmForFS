# -*- coding: utf-8 -*-
"""
@author: rschulte
"""
import numpy as np
import scipy.signal as ss
import scipy.stats as stat
import antropy as entropy
from nitime.algorithms.autoregressive import AR_est_LD as AR
import pywt

class Features():
    """
    Class with all implemented features. Per window the features should be
    initialized using 'initialize_feats' and hereafter all features can
    be extracted by calling the appropriate function. Feature extraction methods
    are implemented to be used per window, thus not multiple windows simultaneously.
    
    For definitions see: Navigating features: a topologically informed chart of 
    electromyographic features space - Phinyomark et al 2017
    """
    def __init__(self):
        b = np.hamming(32) # for AFB
        self.b = b/b.sum()
        self.a = 1 # for AFB
        self.fs = 1000 # for ffts
        self.w1 = [0] # for MAV1
        self.w2 = [0] # for MAV2
        self.t1 = 20 # for MYOP
        self.t2 = 0.02 # for MYOP
        self.t3 = 5e-5 # for MYOP
        
    def initialize_feats(self,x):
        self.freq,self.psd = ss.welch(x,fs=self.fs)
        self.gradient = np.diff(x)
        self.dwt = pywt.wavedec(x,'db7',level=3)
        
        # TDPSD
        log_x = np.log(x**2)
        self.log_x = log_x
        grad_log_x = np.diff(np.log(x**2))
        self.grad_log_x = grad_log_x
        self.m0 = self.get_m0(x)
        self.m2 = self.get_m2(self.gradient)
        self.m4 = self.get_m4(self.gradient)
        self.m0_log = self.get_m0(log_x)
        self.m2_log = self.get_m2(grad_log_x)
        self.m4_log = self.get_m4(grad_log_x)
        
        if len(self.w1) != len(x):
            self.w1 = np.ones_like(x)
            self.w1[:int(len(x)/4)] = 0.5
            self.w1[-int(len(x)/4):] = 0.5
        
        if len(self.w2) != len(x):
            self.w2 = np.ones_like(x)
            self.w2[:int(len(x)/4),] = np.linspace(0,1,int(len(x)/4))
            self.w2[-int(len(x)/4):,] = np.linspace(1,0,int(len(x)/4))
    
    def get_m0(self,x):
    # Helper function for TDPSD
        return np.power(np.sqrt(sum(x**2)),0.1)/0.1

    def get_m2(self,grad_x):
        # Helper function for TDPSD
        return np.power(np.sqrt(np.mean(grad_x**2)),0.1)/0.1
    
    def get_m4(self,grad_x):
        # Helper function for TDPSD
        dgrad_x = np.diff(grad_x)
        return np.power(np.sqrt(np.mean(dgrad_x**2)),0.1)/0.1
    
    # Hargrove feat
    def MEAN(self,x): return np.mean(x,axis=0)
    def STD(self,x): return np.std(x,axis=0)
    def MAX(self,x): return np.max(x,axis=0)
    def MIN(self,x): return np.min(x,axis=0)
    def StartVal(self,x): return x[0]
    def EndVal(self,x): return x[-1]
    def ZC(self,x): return len(np.where(np.diff(np.sign(x),axis=0)!=0)[0]) # Zero-crossings
    def SSC(self,x): return len(np.where(np.diff(np.sign(np.diff(x,axis=0)),axis=0)!=0)[0]) # slope sign changes
    def MAV(self,x): return np.mean(abs(x),axis=0) # Mean Absolute Value 
    def WL(self,x): return np.sum(abs(np.diff(x,axis=0)),axis=0)# Waveform length
#    
#    # Phyomark feat # Navigating features: A topologically informed chart of electromyographic features space
    def AFB(self,x): return ss.lfilter(self.b,self.a,x).max() # Amplitude of first burst
    def ApEn(self,x): 
        return entropy.app_entropy(x) # Approximate entropy
    def SampEn(self,x):
        return entropy.sample_entropy(x) # Sample entropy
        
    def ARC(self,x,order=4): # Autoregressive coeffs
        self.ar_coeff,_ = AR(x,order)
        return self.ar_coeff
    
    def DARC(self,x,order=4): # Differentiated Autoregressive coeffs
        self.dar_coeff,_ = AR(self.gradient,order)
        return self.dar_coeff
    
    def CC(self,x): # Cepstrum coeffs
        a = self.ar_coeff
        c = np.zeros_like(a)
        c[0] = a[0]
        
        for i in range(len(a)):
            c[i] = -a[i] - np.sum([(1-l/i)*a[-1]*c[i-1] for l in range(i)])
        return c
    
    def DCC(self,x): # Differentiated cepstrum coeffs
        a = self.dar_coeff
        c = np.zeros_like(a)
        c[0] = a[0]
        
        for i in range(len(a)):
            c[i] = -a[i] - np.sum([(1-l/i)*a[-1]*c[i-1] for l in range(i)])
        return c
    
    def CEA(self,x): # Critical exponent analysis
        # calculate standard deviation of differenced series using various lags
        lags = range(2, 20)
        tau = [np.sqrt(np.std(np.subtract(x[lag:], x[:-lag]))) for lag in lags]
        m = np.polyfit(np.log(lags), np.log(tau), 1)
        return 2-m[0]*2 # FDc = 2 - H, where H = hurst exponent. https://robotwealth.com/demystifying-the-hurst-exponent-part-1/

    def DMAV(self,x): return self.MAV(self.gradient) # also called DAMV, Differentiated mean absolute value
    def DStd(self,x): return np.std(self.gradient) # also called DASDV, Differentiated standard deviation
    def DFA(self,x): 
        return entropy.detrended_fluctuation(x) # fractal dimension using detrended fluctuation analysis
    def DPR(self,x): # max-to-min drop in PSD ratio
        psd = self.psd
        return psd.max()-psd.min()
    def FR(self,x): # frequency ratio
        A = abs(np.fft.rfft(x))
        freq = np.fft.rfftfreq(len(x),d=1/self.fs)
        return np.mean(A[np.logical_and(freq>=20,freq<=45)])/np.mean(A[freq>=95])
    def HG(self,x): 
        return entropy.higuchi_fd(x,kmax=128)
    def HIST3(self,x): 
        h,_ = np.histogram(x,bins=3)
        return h
    def HIST10(self,x): 
        h,_ = np.histogram(x,bins=10)
        return h
    def IEMG(self,x):
        return abs(x).sum()
    def KATZ(self,x):
        return entropy.katz_fd(x)
    def KURT(self,x): return stat.kurtosis(x)
    def SKEW(self,x): return stat.skew(x)
    def LOG(self,x): return np.exp(np.mean(np.log1p(abs(x)))) # log1p instead of log to remove infs
    def DLD(self,x): return np.exp(np.mean(np.log1p(abs(self.gradient))))
    def M2(self,x): return stat.moment(x,moment=2)
    def MAV1(self,x):
        return np.mean(abs(self.w1*x),axis=0)
    def MAV2(self,x):
        return np.mean(abs(self.w2*x),axis=0)
    def MDF(self,x):
        psd = self.psd
        spsd = np.cumsum(psd)
        return self.freq[spsd >= 0.5*spsd[-1]][0]
    def mDWT(self,x): return np.array([sum(abs(lvl)) for lvl in self.dwt])
    def MP(self,x): return np.mean(self.psd)
    def MNF(self,x):
        freq = self.freq
        psd = self.psd
        return np.sum(freq*psd)/psd.sum()
    def MYOP1(self,x): return np.mean(x>self.t1)
    def MYOP2(self,x): return np.mean(x>self.t2)
    def MYOP3(self,x): return np.mean(x>self.t3)
    def OHM(self,x):
        freq = self.freq
        psd = self.psd
        m0 = np.sum(psd)
        m1 = np.sum(psd*freq)
        m2 = np.sum(psd*freq**2)
        return np.sqrt(m2/m0)/(m1/m0)
    def PKF(self,x):
        A = abs(np.fft.rfft(x))
        freq = np.fft.rfftfreq(len(x),d=1/self.fs)
        return freq[np.argmax(A)]
    def PSDFD(self,x):
        return entropy.katz_fd(self.psd)
    def PSR(self,x):
        psd = self.psd
        i = np.argmax(psd)
        return psd[max([0,i-10]):min(len(psd),i+10)].sum()/psd.sum()
    def RMS(self,x):
        return np.sqrt(np.mean(x**2))
    def SM(self,x):
        freq = self.freq
        psd = self.psd
        return np.sum(psd*freq**2)
    def SMR(self,x):
        freq = self.freq
        psd = self.psd
        return 20*np.log10(np.mean(psd[np.logical_and(freq>=10,freq<450)])/np.mean(psd[freq<10]))
    def SNR(self,x):
        freq = self.freq
        psd = self.psd
        return 20*np.log10(np.mean(psd[np.logical_and(freq>=10,freq<450)])/np.mean(psd[freq>450]))
    def SSI(self,x): return np.sum(x**2)
    def TDPSD1(self,x): 
        # log(m0)
        # Extract a from x
        a = np.log(self.m0)
        
        # Extract b from log(x**2)
        b = np.log(self.m0_log)
        
        return -2*a*b/(a**2 + b**2)

    def TDPSD2(self,x): 
        # log(m0-m2)
        a = np.log(self.m0-self.m2)
        b = np.log(self.m0_log-self.m2_log)
        return -2*a*b/(a**2 + b**2)
    def TDPSD3(self,x): 
        a = np.log(self.m0-self.m4)
        b = np.log(self.m0_log-self.m4_log)
        return -2*a*b/(a**2 + b**2) 
    def TDPSD4(self,x): 
        a = np.log(self.m0/(np.sqrt(self.m0-self.m2)*np.sqrt(self.m0-self.m4)))
        b = np.log(self.m0_log/(np.sqrt(self.m0_log-self.m2_log)*np.sqrt(self.m0_log-self.m4_log)))
        return -2*a*b/(a**2 + b**2) 
         
    def TDPSD5(self,x): 
        a = np.log(self.m2/np.sqrt(self.m0*self.m4))
        b = np.log(self.m2_log/np.sqrt(self.m0_log*self.m4_log))
        return -2*a*b/(a**2 + b**2)
    
    def TDPSD6(self,x):  
        a = np.log(self.WL(x)/self.WL(self.gradient))
        b = np.log(self.WL(self.log_x)/self.WL(self.grad_log_x))
        return -2*a*b/(a**2 + b**2)
    
    def TM(self,x): return abs(stat.moment(x,moment=3))
    def DTM(self,x): return abs(stat.moment(np.gradient(x),moment=3))
    def TP(self,x): return sum(self.psd)
    def VAR(self,x): return np.var(x)
    def DVAR(self,x): return np.var(self.gradient)
    def VCF(self,x):
        freq = self.freq
        psd = self.psd
        m0 = np.sum(psd)
        m1 = np.sum(psd*freq)
        m2 = np.sum(psd*freq**2)
        return m2/m0 - (m1/m0)**2
    def V(self,x): return np.power(np.mean(np.power(x,3)),(1/3))
    def DV(self,x): return np.power(np.mean(np.power(self.gradient,3)),(1/3))
    def WAMP1(self,x): return np.sum(abs(x - np.roll(x,1))>self.t1)
    def WAMP2(self,x): return np.sum(abs(x - np.roll(x,1))>self.t2)
    def WAMP3(self,x): return np.sum(abs(x - np.roll(x,1))>self.t3)

