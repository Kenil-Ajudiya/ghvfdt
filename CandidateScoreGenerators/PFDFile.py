"""
Script which manages PFD files.
Script which manages PFD files. Contains implementations of the functions used to generate scores for PFD files only.

Rob Lyon <robert.lyon@cs.man.ac.uk>

+-----------------------------------------------------------------------------------------+
+                       PLEASE RECORD ANY MODIFICATIONS YOU MAKE BELOW                    +
+-----------------------------------------------------------------------------------------+
+ Revision |   Author    | Description                                       |    DATE    +
+-----------------------------------------------------------------------------------------+

 Revision:1    Rob Lyon    Initial version of code.                            06/02/2014 
"""

import struct, sys, numbers, logging
import numpy as np
from scipy.special import i0
from scipy.optimize import leastsq
from scipy.stats import skew
from scipy.stats import kurtosis
import matplotlib.pyplot as plt # Revision:1

from ProfileOperations import ProfileOperations
from Candidate import CandidateFileInterface

isintorlong = lambda x: (isinstance(x, (int, np.integer)) or isinstance(x, numbers.Integral)) and not isinstance(x, bool)

class PFDOperations(ProfileOperations):
    """
    Contains the functions used to generate the scores that describe the key features of
    a pulsar candidate.
    """
    
    def __init__(self, debugFlag: bool, logger_name: str):
        """
        Default constructor.
        
        Parameters:
        
        debugFlag     -    the debugging flag. If set to True, then detailed
                           debugging messages will be printed to the terminal
                           during execution.
        logger_name   -    the name of the logger to be used for logging messages.
        """
        ProfileOperations.__init__(self, debugFlag, logger_name)
        self.logger = logging.getLogger(logger_name)
    # ****************************************************************************************************
    #
    # Candidate parameters
    #
    # ****************************************************************************************************
    
    def getCandidateParameters(self,profile):
        """
        Computes the candidate parameters. There are four scores computed:
        
        Score 12. The candidate period.
                 
        Score 13. The best signal-to-noise value obtained for the candidate. Higher values desired.
        
        Score 14. The best dispersion measure (dm) obtained for the candidate. Low DM values 
                  are assocaited with local RFI.
                 
        Score 15. The best pulse width.
        
        Parameters:
        profile    -    the PFDFile candidate object NOT profile data.
        
        Returns:
        The candidate period.
        The best signal-to-noise value obtained for the candidate. Higher values desired.
        The best dispersion measure (dm) obtained for the candidate.
        The best pulse width.
        
        """
        
        # Please note that the parameter passed in to this function is actually an
        # instance of the PFDFile class. This is done to keep the code similar for both
        # PFD files. However this means we may get confused when we see that
        # the parameter passed in is called profile - this is the PFDFile object. Thus
        # to access the profile we must call profile.profile. I know this may seem
        # confusing, but it is done on purpose to ensure that the code in the PFDFile
        # scripts is as similar as possible. Despite the fact that these formats are very different.
         
        # Score 12
        self.period = profile.bary_p1 *1000
        
        # Score 13
        avg = profile.profile.mean()
        var = profile.profile.var()
        sigma = np.sqrt(var)

        # Don't we need to worry about the contribution from the pulse
        # itself here?  - BWS 20140314 - How many iterations...?

        snrprofile = []
        nbin = 0
        while nbin < len(profile.profile):
            if profile.profile[nbin] > avg - 3 * sigma and profile.profile[nbin] < avg + 3 * sigma:
                snrprofile.append(profile.profile[nbin])
            nbin += 1

        snr_profile = np.array(snrprofile)

        avg = snr_profile.mean()
        var = snr_profile.var()
        
        self.snr = ((profile.profile-avg)/np.sqrt(var)).sum()
        if self.snr < 0:
            self.snr = 0.1

        # Score 14
        self.dm = profile.bestdm
        
        # Score 15
        # Calculate the width of the pulse BWS 20140316

        peak = profile.profile.argmax() # Finds the index of the largest value across the x-axis.

        xData = np.array(list(range(len(profile.profile))))

        # Rotate profile to put it in the centre 
        shift = peak - len(profile.profile) / 2
        rot_profile = self.fft_rotate(profile.profile,shift) - min(profile.profile)
        
        # Determine the pulse width, assume that it can be gotten by finding extrema
        # of the Half maximum points. 
        peak = rot_profile.argmax()
        # self.logger.debug("Peak: %s", peak)
        halfmax_profile = max(rot_profile) / 2
        left_lim = peak
        while left_lim > 0:
            # self.logger.debug("%s %s", rot_profile[left_lim], halfmax_profile)
            if rot_profile[left_lim] < halfmax_profile:
                break
            else:
                left_lim -= 1
        # self.logger.debug("LL: %s", left_lim)
        right_lim = peak
        while right_lim < len(rot_profile):
            # self.logger.debug("%s %s", rot_profile[right_lim], halfmax_profile)
            if rot_profile[right_lim] < halfmax_profile:
                break
            else:
                right_lim += 1
        # self.logger.debug("RL: %s", right_lim)
        
        if(self.debug):
            plt.plot(xData,rot_profile,left_lim,rot_profile[left_lim], 'o',right_lim,rot_profile[right_lim],'o',peak,halfmax_profile,'o')
            plt.show()

        self.width = (1.0 * (right_lim - left_lim - 1.0)) / len(rot_profile);

        return [self.period,self.snr,self.dm,self.width]
        
    
    # ****************************************************************************************************
    #
    # DM Curve Fittings
    #
    # ****************************************************************************************************
    
    def getDMCurveData(self,data):
        """
        Extracts the DM curve data from the PFD file.
        """
        
        lodm = data.dms[0]
        hidm = data.dms[-1]
        y_values,dm_index = data.plot_chi2_vs_DM(lodm, hidm)
        
        return y_values
    
    def getDMCurveDataNormalised(self,data):
        """
        Extracts the DM curve data from the PFD file.
        
        """
        
        lodm = data.dms[0]
        hidm = data.dms[-1]
        y_values,dm_index = data.plot_chi2_vs_DM(lodm, hidm)
            
        # Extract DM curve.
        curve=[]
        curve.append(y_values)
        curve.append(list(range(len(y_values))))
            
        yData = curve[0]
        yData = 255./max(yData)*yData
        
        return yData
        
    def getDMFittings(self,data):
        """
        Computes the dispersion measure curve fitting parameters. There are four scores computed:
        
        Score 16. This score computes SNR / SQRT( (P-W) / W ).
                 
        Score 17. Difference between fitting factor Prop, and 1. If the candidate is a pulsar,
                  then prop should be equal to 1.
        
        Score 18. Difference between best DM value and optimised DM value from fit. This difference
                  should be small for a legitimate pulsar signal. 
                 
        Score 19. Chi squared value from DM curve fit, smaller values indicate a smaller fit. Thus
                  smaller values will be possessed by legitimate signals.
        
        Parameters:
        rawData    -    the raw candidate xml data.
        profile    -    the profile data.
        
        Returns:
        SNR / SQRT( (P-W) / W ).
        Difference between fitting factor Prop, and 1.
        Difference between best DM value and optimized DM value from fit.
        Chi squared value from DM curve fit, smaller values indicate a smaller fit.
        
        """
        
        # Calculates the residuals.
        def __residuals(paras, x, y):     
            Amp,Prop,Shift,Up = paras
            weff = np.sqrt(wint + pow(Prop*kdm*abs((self.dm + Shift)-x)*df/pow(f,3),2))
            for wind in range(len(weff)):
                if ( weff[wind] > self.period ):
                    weff[wind] = self.period
            SNR  = Up+Amp*np.sqrt((self.period-weff)/weff)
            err  = y - SNR
            return err
        
        # Evaluates the function.
        def __evaluate(x, paras):
            Amp,Prop,Shift,Up = paras
            weff = np.sqrt(wint + pow(Prop*kdm*abs((self.dm + Shift)-x)*df/pow(f,3),2))
            for wind in range(len(weff)):
                if ( weff[wind] > self.period ):
                    weff[wind] = self.period
            SNR  = Up+Amp*np.sqrt((self.period-weff)/weff)
            return SNR
        
        lodm = data.dms[0]
        hidm = data.dms[-1]
        y_values,dm_index = data.plot_chi2_vs_DM(lodm, hidm)
            
        # Extract DM curve.
        curve=[]
        curve.append(y_values)
        curve.append(list(range(len(y_values))))
            
        yData = curve[0]
        yData = 255./max(yData)*yData
        length_all = len(y_values)
        length = len(yData)
                    
        # Get start and end DM value and calculate step width.
        dm_start,dm_end = float(dm_index[1]),float(dm_index[len(dm_index)-1])
        dm_step = abs(dm_start-dm_end)/length_all
        
        # SNR and pulse parameters.
        wint = (self.width * self.period)**2
        kdm = 8.3*10**6
        df = 32
        f = 135
        
        peak = self.snr/np.sqrt((self.period-np.sqrt(wint))/np.sqrt(wint))
        
        # Scale x-data.
        xData = []
        for i in range(length):
            xData.append(dm_start+curve[1][i]*dm_step)    
        xData = np.array(xData)
        
        # Calculate theoretic dm-curve from best values.
        _help = []
        for i in range(length):
            weff = np.sqrt(wint + pow(kdm*abs(self.dm-xData[i])*df/pow(f,3),2))
            if weff > self.period:
                weff = self.period
            SNR = np.sqrt((self.period-weff)/weff)
            _help.append(float(SNR))
            
        theo = (255./max(_help))*np.array(_help)
        
        # Start parameter for fit.
        Amp = (255./max(_help))
        Prop,Shift  = 1,0
        p0 = (Amp,Prop,Shift,0)
        plsq = leastsq(__residuals, p0, args=(xData,yData))
        fit = __evaluate(xData, plsq[0])

        if(self.debug):
            plt.plot(xData,fit,xData,yData,xData,theo)
            plt.title("DM Curve, theoretical curve and fit.")
            plt.show()
            
        # Chi square calculation.
        chi_fit,chi_theo = 0,0
        ndeg = 0
        for i in range(length):
            if theo[i] > 0:
                chi_fit  += (yData[i]-fit[i])**2  / fit[i]
                chi_theo += (yData[i]-theo[i])**2 / theo[i]
                ndeg += 1
                
        chi_fit  =  chi_fit/ndeg
        chi_theo = chi_theo/ndeg
        
        # self.logger.debug("CHISQ: %s %s", chi_fit, chi_theo)

        diffBetweenFittingFactor = abs(1-plsq[0][1])
        diffBetweenBestAndOptimisedDM = plsq[0][2]
        return peak, diffBetweenFittingFactor, diffBetweenBestAndOptimisedDM , chi_theo
    
    # ****************************************************************************************************
    #
    # Sub-band scores
    #
    # ****************************************************************************************************
    
    def getSubbandParameters(self,data=None,profile=None):
        """
        Computes the sub-band scores. There are three scores computed:
        
        Score 20. RMS of peak positions in all sub-bands. Smaller values should be possessed by
                  legitimate pulsar signals.
                 
        Score 21. Average correlation coefficient for each pair of sub-bands. Larger values should be
                  possessed by legitimate pulsar signals.
        
        Score 22. Sum of correlation coefficients between sub-bands and profile. Larger values should be
                  possessed by legitimate pulsar signals.
        
        Parameters:
        data       -    the raw candidate data.
        profile    -    a numpy.ndarray containing profile data.
        
        Returns:
        RMS of peak positions in all sub-bands.
        Average correlation coefficient for each pair of sub-bands.
        Sum of correlation coefficients between sub-bands and profile.
        
        """
        
        if(data==None and profile==None):
            return [0.0,0.0,0.0]
        
        # First, sub-bands.
        subbands = data.plot_subbands()
        prof_bins = data.proflen
        band_subbands = data.nsub
        
        RMS,mean_corr = self.getSubband_scores(subbands, prof_bins, band_subbands, self.width)
        correlation = self.getProfileCorr(subbands, band_subbands, profile)
        
        # Now calculate integral of correlation coefficients.
        correlation_integral = 0
        for i in range( len( correlation ) ):
            correlation_integral += correlation[i]
                    
        return [RMS,mean_corr,correlation_integral]
    
    # ******************************************************************************************
 
    def getProfileCorr(self,subbands, band_subbands, profile):
        """
        Calculates the correlation of the profile with the subbands, -integrals.
        
        Parameters:
        subbands         -    the sub-band data.
        band_subbands    -    the number of sub-bands.
        bestWidth        -    the best pulse width.
        
        Returns:
        
        A list with the correlation data in decimal format.            
        
        """
        
        corrlist = []
        for j in range(band_subbands):
            coef = abs(np.corrcoef(subbands[j],profile))
            if coef[0][1] > 0.0055:
                corrlist.append(coef[0][1])
        
        return np.array(corrlist)
     
    # ****************************************************************************************************
    #
    # Other Utility Functions
    #
    # ****************************************************************************************************
    
    def delay_from_DM(self,DM, freq_emitted):
        """
        Return the delay in seconds caused by dispersion, given
        a Dispersion Measure (DM) in cm-3 pc, and the emitted
        frequency (freq_emitted) of the pulsar in MHz.
        """
        if (type(freq_emitted)==type(0.0)):
            if (freq_emitted > 0.0):
                return DM/(0.000241*freq_emitted*freq_emitted)
            else:
                return 0.0
        else:
            return np.where(freq_emitted > 0.0,DM/(0.000241*freq_emitted*freq_emitted), 0.0)
        
    # ****************************************************************************************************
    
    def fft_rotate(self,arr, bins):
        """
        Return array 'arr' rotated by 'bins' places to the left.  The
        rotation is done in the Fourier domain using the Shift Theorem.
        'bins' can be fractional.  The resulting vector will have the
        same length as the original.
        """
        arr = np.asarray(arr)
        freqs = np.arange(arr.size/2+1, dtype=np.float)
        phasor = np.exp(complex(0.0, (2.0*np.pi)) * freqs * bins / float(arr.size))
        return np.fft.irfft(phasor * np.fft.rfft(arr))       
    
    # ****************************************************************************************************
    
    def span(self,Min, Max, Number):
        """
        span(Min, Max, Number):
        Create a range of 'np' floats given inclusive 'Min' and 'Max' values.
        """
        assert isintorlong(Number)
        if isintorlong(Min) and isintorlong(Max) and (Max-Min) % (Number-1) != 0:
            Max = float(Max) # force floating points
        
        return Min+(Max-Min)*np.arange(Number)/(Number-1)
        
    # ****************************************************************************************************
    
    def rotate(self,arr, bins):
        """
        Return an array rotated by 'bins' places to the left
        """
        bins = bins % len(arr)
        if bins==0:
            return arr
        else:
            return np.concatenate((arr[bins:], arr[:bins]))
    
    # ****************************************************************************************************
    
    def interp_rotate(self,arr, bins, zoomfact=10):
        """
        Return a sinc-interpolated array rotated by 'bins' places to the left.
        'bins' can be fractional and will be rounded to the closest
        whole-number of interpolated bins.  The resulting vector will
        have the same length as the oiginal.
        """
        newlen = len(arr)*zoomfact
        rotbins = int(np.floor(bins*zoomfact+0.5)) % newlen
        newarr = self.periodic_interp(arr, zoomfact)
        return self.rotate(newarr, rotbins)[::zoomfact]

    # ****************************************************************************************************
    
    def periodic_interp(self,data, zoomfact, window='hanning', alpha=6.0):
        """
        Return a periodic, windowed, sinc-interpolation of the data which
        is oversampled by a factor of 'zoomfact'.
        """
        zoomfact = int(zoomfact)
        if (zoomfact < 1):
            # self.logger.warning("zoomfact must be >= 1.")
            return 0.0
        elif zoomfact==1:
            return data
        
        newN = len(data)*zoomfact
        # Space out the data
        comb = np.zeros((zoomfact, len(data)), dtype='d')
        comb[0] += data
        comb = np.reshape(np.transpose(comb), (newN,))
        # Compute the offsets
        xs = np.zeros(newN, dtype='d')
        xs[:newN/2+1] = np.arange(newN/2+1, dtype='d')/zoomfact
        xs[-newN/2:]  = xs[::-1][newN/2-1:-1]
        # Calculate the sinc times window for the kernel
        if window.lower()=="kaiser":
            win = _window_function[window](xs, len(data)/2, alpha)
        else:
            win = _window_function[window](xs, len(data)/2)
        kernel = win * self.sinc(xs)
        
        return np.fft.irfft(np.fft.rfft(kernel) * np.fft.rfft(comb))
    
    # ****************************************************************************************************
    
    def sinc(self,xs):
        """
        Return the sinc function [i.e. sin(pi * xs)/(pi * xs)] for the values xs.
        """
        pxs = np.pi*xs
        return np.where(np.fabs(pxs)<1e-3, 1.0-pxs*pxs/6.0, np.sin(pxs)/pxs)
    
    # ****************************************************************************************************
    
    # The code below is a little bit of a mess. But There was little I could do to
    # clean in up, since this is PRESTO code being retro-fitted to work for our purposes.
    
def kaiser_window(xs, halfwidth, alpha):
    """
        Return the kaiser window function for the values 'xs' when the
            the half-width of the window should be 'haldwidth' with
            the folloff parameter 'alpha'.  The following values are
            particularly interesting:

            alpha
            -----
            0           Rectangular Window
            5           Similar to Hamming window
            6           Similar to Hanning window
            8.6         Almost identical to the Blackman window 
    """
    win = i0(alpha*np.sqrt(1.0-(xs/halfwidth)**2.0))/i0(alpha)
    return np.where(np.fabs(xs)<=halfwidth, win, 0.0)

def hanning_window(xs, halfwidth):
    """
    hanning_window(xs, halfwidth):
        Return the Hanning window of halfwidth 'halfwidth' evaluated at
            the values 'xs'.
    """
    win =  0.5 + 0.5*np.cos(np.pi*xs/halfwidth)
    return np.where(np.fabs(xs)<=halfwidth, win, 0.0)

def hamming_window(xs, halfwidth):
    """
    hamming_window(xs, halfwidth):
        Return the Hamming window of halfwidth 'halfwidth' evaluated at
            the values 'xs'.
    """
    win =  0.54 + 0.46*np.cos(np.pi*xs/halfwidth)
    return np.where(np.fabs(xs)<=halfwidth, win, 0.0)

def blackman_window(xs, halfwidth):
    """
    blackman_window(xs, halfwidth):
        Return the Blackman window of halfwidth 'halfwidth' evaluated at
            the values 'xs'.
    """
    rat = np.pi*xs/halfwidth
    win =  0.42 + 0.5*np.cos(rat) + 0.08*np.cos(2.0*rat) 
    return np.where(np.fabs(xs)<=halfwidth, win, 0.0)

def rectangular_window(xs, halfwidth):
    """
    rectangular_window(xs, halfwidth):
        Return a rectangular window of halfwidth 'halfwidth' evaluated at
            the values 'xs'.
    """
    return np.where(np.fabs(xs)<=halfwidth, 1.0, 0.0)

_window_function = {"rectangular": rectangular_window,
                    "none": rectangular_window,
                    "hanning": hanning_window,
                    "hamming": hamming_window,
                    "blackman": blackman_window,
                    "kaiser": kaiser_window}

class PFD(CandidateFileInterface):
    """
    Represents a PFD file.
    """
    
    def __init__(self, debugFlag: bool, candidateName: str, logger_name: str):
        """
        Default constructor.
        
        Parameters:
        
        debugFlag     -    the debugging flag. If set to True, then detailed
                           debugging messages will be printed to the terminal
                           during execution.
        candidateName -    the name for the candidate, typically the file path.
        """
        CandidateFileInterface.__init__(self, debugFlag, logger_name)
        self.cand = candidateName
        self.scores=[]
        self.profileOps = PFDOperations(self.debug, logger_name)
        self.setNumberOfScores(22)
        self.load()
        self.logger = logging.getLogger(logger_name)

    # ****************************************************************************************************
           
    def load(self):
        """
        Attempts to load candidate data from the file, performs file consistency checks if the
        debug flag is set to true.
        
        Parameters:
        N/A
        
        Return:
        N/A
        """
        infile = open(self.cand, "rb")
        
        # The code below appears to have been taken from Presto. So it maybe
        # helpful to look at the Presto github repository to get a better feel
        # for what this code is doing. I certainly have no idea what is going on.
            
        swapchar = '<' # this is little-endian
        data = infile.read(5*4)
        testswap = struct.unpack(swapchar+"i"*5, data)
        # This is a hack to try and test the endianness of the data.
        # None of the 5 values should be a large positive number.
        
        if (np.fabs(np.asarray(testswap))).max() > 100000:
            swapchar = '>' # this is big-endian
            
        (self.numdms, self.numperiods, self.numpdots, self.nsub, self.npart) = struct.unpack(swapchar+"i"*5, data)
        (self.proflen, self.numchan, self.pstep, self.pdstep, self.dmstep, self.ndmfact, self.npfact) = struct.unpack(swapchar+"i"*7, infile.read(7*4))
        self.filenm = infile.read(struct.unpack(swapchar+"i", infile.read(4))[0])
        self.candnm = infile.read(struct.unpack(swapchar+"i", infile.read(4))[0])
        self.telescope = infile.read(struct.unpack(swapchar+"i", infile.read(4))[0])
        self.pgdev = infile.read(struct.unpack(swapchar+"i", infile.read(4))[0])
        
        test = infile.read(16)
        has_posn = 1
        for ii in range(16):
            if test[ii:ii+1] not in b'0123456789:.-\0':
                has_posn = 0
                break
            
        if has_posn:
            self.rastr = test.split(b'\0', 1)[0].decode('ascii', 'ignore')
            test = infile.read(16)
            self.decstr = test.split(b'\0', 1)[0].decode('ascii', 'ignore')
            (self.dt, self.startT) = struct.unpack(swapchar+"dd", infile.read(2*8))
        else:
            self.rastr = "Unknown"
            self.decstr = "Unknown"
            (self.dt, self.startT) = struct.unpack(swapchar+"dd", test)
            
        (self.endT, self.tepoch, self.bepoch, self.avgvoverc, self.lofreq,self.chan_wid, self.bestdm) = struct.unpack(swapchar+"d"*7, infile.read(7*8))
        (self.topo_pow, tmp) = struct.unpack(swapchar+"f"*2, infile.read(2*4))
        (self.topo_p1, self.topo_p2, self.topo_p3) = struct.unpack(swapchar+"d"*3,infile.read(3*8))
        (self.bary_pow, tmp) = struct.unpack(swapchar+"f"*2, infile.read(2*4))
        (self.bary_p1, self.bary_p2, self.bary_p3) = struct.unpack(swapchar+"d"*3,infile.read(3*8))
        (self.fold_pow, tmp) = struct.unpack(swapchar+"f"*2, infile.read(2*4))
        (self.fold_p1, self.fold_p2, self.fold_p3) = struct.unpack(swapchar+"d"*3,infile.read(3*8))
        (self.orb_p, self.orb_e, self.orb_x, self.orb_w, self.orb_t, self.orb_pd,self.orb_wd) = struct.unpack(swapchar+"d"*7, infile.read(7*8))
        self.dms = np.asarray(struct.unpack(swapchar+"d"*self.numdms,infile.read(self.numdms*8)))
        
        if self.numdms==1:
            self.dms = self.dms[0]
            
        self.periods = np.asarray(struct.unpack(swapchar + "d" * self.numperiods,infile.read(self.numperiods*8)))
        self.pdots = np.asarray(struct.unpack(swapchar + "d" * self.numpdots,infile.read(self.numpdots*8)))
        self.numprofs = self.nsub * self.npart
        
        if (swapchar=='<'):  # little endian
            self.profs = np.zeros((self.npart, self.nsub, self.proflen), dtype='d')
            for ii in range(self.npart):
                for jj in range(self.nsub):
                    try:
                        self.profs[ii,jj,:] = np.fromfile(infile, np.float64, self.proflen)
                    except Exception: # Catch *all* exceptions.
                        pass
        else:
            self.profs = np.asarray(struct.unpack(swapchar+"d"*self.numprofs*self.proflen,infile.read(self.numprofs*self.proflen*8)))
            self.profs = np.reshape(self.profs, (self.npart, self.nsub, self.proflen))
                
        self.binspersec = self.fold_p1 * self.proflen
        self.chanpersub = self.numchan / self.nsub
        self.subdeltafreq = self.chan_wid * self.chanpersub
        self.hifreq = self.lofreq + (self.numchan-1) * self.chan_wid
        self.losubfreq = self.lofreq + self.subdeltafreq - self.chan_wid
        self.subfreqs = np.arange(self.nsub, dtype='d')*self.subdeltafreq + self.losubfreq
        self.subdelays_bins = np.zeros(self.nsub, dtype='d')
        self.killed_subbands = []
        self.killed_intervals = []
        self.pts_per_fold = []
        
        # Note: a foldstats struct is read in as a group of 7 doubles
        # the correspond to, in order:
        # numdata, data_avg, data_var, numprof, prof_avg, prof_var, redchi
        self.stats = np.zeros((self.npart, self.nsub, 7), dtype='d')
        
        for ii in range(self.npart):
            currentstats = self.stats[ii]
            
            for jj in range(self.nsub):
                if (swapchar=='<'):  # little endian
                    try:
                        currentstats[jj] = np.fromfile(infile, np.float64, 7)
                    except Exception: # Catch *all* exceptions.
                        pass
                else:
                    try:
                        currentstats[jj] = np.asarray(struct.unpack(swapchar+"d"*7,infile.read(7*8)))
                    except Exception: # Catch *all* exceptions.
                        pass
                    
            self.pts_per_fold.append(self.stats[ii][0][0])  # numdata from foldstats
            
        self.start_secs = np.add.accumulate([0]+self.pts_per_fold[:-1])*self.dt
        self.pts_per_fold = np.asarray(self.pts_per_fold)
        self.mid_secs = self.start_secs + 0.5*self.dt*self.pts_per_fold
        
        if (not self.tepoch==0.0):
            self.start_topo_MJDs = self.start_secs/86400.0 + self.tepoch
            self.mid_topo_MJDs = self.mid_secs/86400.0 + self.tepoch
        
        if (not self.bepoch==0.0):
            self.start_bary_MJDs = self.start_secs/86400.0 + self.bepoch
            self.mid_bary_MJDs = self.mid_secs/86400.0 + self.bepoch
            
        self.Nfolded = np.add.reduce(self.pts_per_fold)
        self.T = self.Nfolded*self.dt
        self.avgprof = (self.profs/self.proflen).sum()
        self.varprof = self.calc_varprof()
        self.barysubfreqs = self.subfreqs
        infile.close()
            
        # If explicit debugging required.
        if(self.debug):
            
            # If candidate file is invalid in some way...
            if(self.isValid()==False):

                self.logger.error("Invalid PFD candidate: %s", self.cand)
                scores=[]
                
                # Return only NaN values for scores.
                for n in range(0, self.numberOfScores):
                    scores.append(float("nan"))
                return scores
            
            # Candidate file is valid.
            else:
                self.logger.info("Candidate file valid.")
                self.profile = np.array(self.getprofile())
            
        # Just go directly to score generation without checks.
        else:
            self.out( "Candidate validity checks skipped.","")
            self.profile = np.array(self.getprofile())
    
    # ****************************************************************************************************
    
    def getprofile(self):
        """
        Obtains the profile data from the candidate file.
        
        Parameters:
        N/A
        
        Returns:
        The candidate profile data (an array) scaled to within the range [0,255].
        """
        if 'subdelays' not in self.__dict__:
            self.dedisperse()
          
        normprof = self.sumprof - min(self.sumprof)
        
        s = normprof / np.mean(normprof)
        
        if(self.debug):
            plt.plot(s)
            plt.title("Profile.")
            plt.show()
            
        return self.scale(s)
    
    # ****************************************************************************************************
    
    def scale(self,data):
        """
        Scales the profile data for pfd files so that it is in the range 0-255.
        So  by performing this scaling the scores for both type of candidates are
        directly comparable. Before it was harder to determine if the scores
        generated for pfd files were working correctly.
        
        Parameter:
        data    -    the data to scale to within the 0-255 range.
        
        Returns:
        A new array with the data scaled to within the range [0,255].
        """
        min_=min(data)
        max_=max(data)
        
        newMin=0;
        newMax=255
        
        newData=[]
        
        for n in range(len(data)):
            
            value=data[n]
            x = (newMin * (1-( (value-min_) /( max_-min_ )))) + (newMax * ( (value-min_) /( max_-min_ ) ))
            newData.append(x)
            
        return newData
    
    # ****************************************************************************************************
        
    def calc_varprof(self):
        """
        This function calculates the summed profile variance of the current pfd file.
        Killed profiles are ignored. I have no idea what a killed profile is. But it
        sounds fairly gruesome.
        """
        varprof = 0.0
        for part in range(self.npart):
            if part in self.killed_intervals: continue
            for sub in range(self.nsub):
                if sub in self.killed_subbands: continue
                varprof += self.stats[part][sub][5] # foldstats prof_var
        return varprof
    
    # ****************************************************************************************************
        
    def dedisperse(self, DM=None, interp=0):
        """
        Rotate (internally) the profiles so that they are de-dispersed
        at a dispersion measure of DM.  Use FFT-based interpolation if
        'interp' is non-zero (NOTE: It is off by default!).
        
        """

        if DM is None:
            DM = self.bestdm
            
        # Note:  Since TEMPO pler corrects observing frequencies, for
        #        TOAs, at least, we need to de-disperse using topocentric
        #        observing frequencies.
        self.subdelays = self.profileOps.delay_from_DM(DM, self.subfreqs)
        self.hifreqdelay = self.subdelays[-1]
        self.subdelays = self.subdelays-self.hifreqdelay
        delaybins = self.subdelays*self.binspersec - self.subdelays_bins
        
        if interp:
            
            new_subdelays_bins = delaybins
            
            for ii in range(self.npart):
                for jj in range(self.nsub):
                    tmp_prof = self.profs[ii,jj,:]
                    self.profs[ii,jj] = self.profileOps.fft_rotate(tmp_prof, delaybins[jj])
                    
            # Note: Since the rotation process slightly changes the values of the
            # profs, we need to re-calculate the average profile value
            self.avgprof = (self.profs/self.proflen).sum()
            
        else:
            
            new_subdelays_bins = np.floor(delaybins+0.5)
            
            for ii in range(self.nsub):
                
                rotbins = int(new_subdelays_bins[ii]) % self.proflen
                if rotbins:  # i.e. if not zero
                    subdata = self.profs[:,ii,:]
                    self.profs[:,ii] = np.concatenate((subdata[:,rotbins:],subdata[:,:rotbins]), 1)
                    
        self.subdelays_bins += new_subdelays_bins
        self.sumprof = self.profs.sum(0).sum(0)
    
    # ******************************************************************************************
    
    def plot_chi2_vs_DM(self, loDM, hiDM, N=100, interp=0):
        """
        Plot (and return) an array showing the reduced-chi^2 versus DM 
        (N DMs spanning loDM-hiDM). Use sinc_interpolation if 'interp' is non-zero.
        """

        # Sum the profiles in time
        sumprofs = self.profs.sum(0)
        
        if not interp:
            profs = sumprofs
        else:
            profs = np.zeros(np.shape(sumprofs), dtype='d')
            
        DMs = self.profileOps.span(loDM, hiDM, N)
        chis = np.zeros(N, dtype='f')
        subdelays_bins = self.subdelays_bins.copy()
        
        for ii, DM in enumerate(DMs):
            
            subdelays = self.profileOps.delay_from_DM(DM, self.barysubfreqs)
            hifreqdelay = subdelays[-1]
            subdelays = subdelays - hifreqdelay
            delaybins = subdelays*self.binspersec - subdelays_bins
            
            if interp:
                
                interp_factor = 16
                for jj in range(self.nsub):
                    profs[jj] = self.profileOps.interp_rotate(sumprofs[jj], delaybins[jj],zoomfact=interp_factor)
                # Note: Since the interpolation process slightly changes the values of the
                # profs, we need to re-calculate the average profile value
                avgprof = (profs/self.proflen).sum()
                
            else:
                
                new_subdelays_bins = np.floor(delaybins+0.5)
                for jj in range(self.nsub):
                    profs[jj] = self.profileOps.rotate(profs[jj], int(new_subdelays_bins[jj]))
                subdelays_bins += new_subdelays_bins
                avgprof = self.avgprof
                
            sumprof = profs.sum(0)        
            chis[ii] = self.calc_redchi2(prof=sumprof, avg=avgprof)

        return (chis, DMs)
    
    # ******************************************************************************************
    
    def calc_redchi2(self, prof=None, avg=None, var=None):
        """
        Return the calculated reduced-chi^2 of the current summed profile.
        """
        
        if 'subdelays' not in self.__dict__:
            self.dedisperse()
            
        if prof is None:  prof = self.sumprof
        if avg is None:  avg = self.avgprof
        if var is None:  var = self.varprof
        return ((prof-avg)**2.0/var).sum()/(len(prof)-1.0)
    
    # ******************************************************************************************
    
    def plot_subbands(self):
        """
        Plot the interval-summed profiles vs subband.  Restrict the bins
        in the plot to the (low:high) slice defined by the phasebins option
        if it is a tuple (low,high) instead of the string 'All'. 
        """
        if 'subdelays' not in self.__dict__:
            self.dedisperse()
        
        lo, hi = 0.0, self.proflen
        profs = self.profs.sum(0)
        lof = self.lofreq - 0.5*self.chan_wid
        hif = lof + self.chan_wid*self.numchan
        
        return profs
                        
    # ****************************************************************************************************
        
    def isValid(self):
        """
        Tests the data loaded from a pfd file.
        
        Parameters:
        
        Returns:
        True if the data is well formed and valid, else false.
        """
        
        # These are only basic checks, more in depth checks should be implemented
        # by someone more familiar with the pfd file format.
        if(self.proflen > 0 and self.numchan > 0):
            return True
        else:
            return False
    
    # ****************************************************************************************************
    
    def computeProfileScores(self):
        """
        Builds the scores using raw profile intensity data only. Returns the scores.
        
        Parameters:
        N/A
        
        Returns:
        An array of profile intensities as floating point values.
        """
        for intensity in self.profile:
            self.scores.append(float(intensity))
            
        return self.scores
    
    def getDMCurveData(self):
        """
        Returns a list of integer data points representing the candidate DM curve.
        
        Parameters:
        N/A
        
        Returns:
        A list data type containing data points.
        
        """
        
        try:
            curve = self.profileOps.getDMCurveData(self)
            #curve = self.profileOps.getDMCurveDataNormalised(self)
            # Add first scores.
            
            if(self.debug==True):
                self.logger.debug("curve = %s", curve)

            return curve

        except Exception as e: # catch *all* exceptions
            self.logger.error("Error getting DM curve data from PFD file\n\t%s", sys.exc_info()[0])
            self.logger.error(self.format_exception(e))
            raise Exception("DM curve extraction exception")
    
    def computeProfileStatScores(self):
        """
        Builds the stat scores using raw profile intensity data only. Returns the scores.
        
        Parameters:
        N/A
        
        Returns:
        An array of profile intensities as floating point values.
        """
        
        try:
            
            bins=[] 
            for intensity in self.profile:
                bins.append(float(intensity))
            
            mn = np.mean(bins)
            stdev = np.std(bins)
            skw = skew(bins)
            kurt = kurtosis(bins)
            
            stats = [mn,stdev,skw,kurt]
            return stats
        
        except Exception as e: # catch *all* exceptions
            self.logger.error("Error getting Profile stat scores from PFD file\n\t%s", sys.exc_info()[0])
            self.logger.error(self.format_exception(e))
            raise Exception("Profile stat score extraction exception")
    
    def computeDMCurveStatScores(self):
        """
        Returns a list of integer data points representing the candidate DM curve.
        
        Parameters:
        N/A
        
        Returns:
        A list data type containing data points.
        
        """
        
        try:
            bins=[]
            bins = self.profileOps.getDMCurveData(self)
            #curve = self.profileOps.getDMCurveDataNormalised(self)
            # Add first scores.
            
            mn = np.mean(bins)
            stdev = np.std(bins)
            skw = skew(bins)
            kurt = kurtosis(bins)
            
            stats = [mn,stdev,skw,kurt]
            return stats  
        
        except Exception as e: # catch *all* exceptions
            self.logger.error("Error getting DM curve stat scores from PFD file\n\t%s", sys.exc_info()[0])
            self.logger.error(self.format_exception(e))
            raise Exception("DM curve stat score extraction exception")
        
    # ****************************************************************************************************
    
    def compute(self):
        """
        Builds the scores using the PFDOperations.py file. Returns the scores.
        
        Parameters:
        N/A
        
        Returns:
        An array of 22 candidate scores as floating point values.
        """
        
        # Get scores 1-4
        self.computeSinusoidFittingScores()
        
        # Get scores 5-11
        self.computeGaussianFittingScores()

        # Get scores 12-15
        self.computeCandidateParameterScores()
        
        # Get scores 16-19
        self.computeDMCurveFittingScores()
        
        # Get scores 20-22
        self.computeSubBandScores()

        return self.scores
        
    # ****************************************************************************************************
    
    def computeSinusoidFittingScores(self):
        """
        Computes the sinusoid fitting scores for the profile data. There are four scores computed:
        
        Score 1. Chi-Squared value for sine fit to raw profile. This score attempts to fit a sine curve
                 to the pulse profile. The reason for doing this is that many forms of RFI are sinusoidal.
                 Thus the chi-squared value for such a fit should be low for RFI (indicating
                 a close fit) and high for a signal of interest (indicating a poor fit).
                 
        Score 2. Chi-Squared value for sine-squared fit to amended profile. This score attempts to fit a sine
                 squared curve to the pulse profile, on the understanding that a sine-squared curve is similar
                 to legitimate pulsar emission. Thus the chi-squared value for such a fit should be low for
                 RFI (indicating a close fit) and high for a signal of interest (indicating a poor fit).
                 
        Score 3. Difference between maxima. This is the number of peaks the program identifies in the pulse
                 profile - 1. Too high a value may indicate that a candidate is caused by RFI. If there is only
                 one pulse in the profile this value should be zero.
                 
        Score 4. Sum over residuals.  Given a pulse profile represented by an array of profile intensities P,
                 the sum over residuals subtracts ( (max-min) /2) from each value in P. A larger sum generally
                 means a higher SNR and hence other scores will also be stronger, such as correlation between
                 sub-bands. Example,
                 
                 P = [ 10 , 13 , 17 , 50 , 20 , 10 , 5 ]
                 max = 50
                 min = 5
                 (abs(max-min))/2 = 22.5
                 so the sum over residuals is:
                 
                  = (22.5 - 10) + (22.5 - 13) + (22.5 - 17) + (22.5 - 50) + (22.5 - 20) + (22.5 - 10) + (22.5 - 5)
                  = 12.5 + 9.5 + 5.5 + (-27.5) + 2.5 + 12.5 + 17.5
                  = 32.5
        
        Parameters:
        N/A
        
        Returns:
        
        Four candidate scores.
        """
        try:
            sin_fit = self.profileOps.getSinusoidFittings(self.profile)
            # Add first scores.
            self.scores.append(float(sin_fit[0])) # Score 1.  Chi-Squared value for sine fit to raw profile.
            self.scores.append(float(sin_fit[1])) # Score 2.  Chi-Squared value for sine-squared fit to amended profile.
            self.scores.append(float(sin_fit[2])) # Score 3.  Difference between maxima.
            self.scores.append(float(sin_fit[3])) # Score 4.  Sum over residuals.
            
            if(self.debug==True):
                self.logger.debug("Score 1. Chi-Squared value for sine fit to raw profile = %s", sin_fit[0])
                self.logger.debug("Score 2. Chi-Squared value for sine-squared fit to amended profile = %s", sin_fit[1])
                self.logger.debug("Score 3. Difference between maxima = %s", sin_fit[2])
                self.logger.debug("Score 4. Sum over residuals = %s", sin_fit[3])

        except Exception as e: # catch *all* exceptions
            self.logger.error("Error computing scores 1-4 (Sinusoid Fitting) \n\t%s", sys.exc_info()[0])
            self.logger.error(self.format_exception(e))
            raise Exception("Sinusoid fitting exception")

    # ****************************************************************************************************
    
    def computeGaussianFittingScores(self):
        """
        Computes the Gaussian fitting scores for the profile data. There are seven scores computed:
        
        Score 5. Distance between expectation values of Gaussian and fixed Gaussian fits to profile histogram.
                 This scores fits a two Gaussian curves to a histogram of the profile data. One of these
                 Gaussian fits has its mean value set to the value in the centre bin of the histogram,
                 the other is not constrained. Thus it is expected that for a candidate arising from noise,
                 these two fits will be very similar - the distance between them will be zero. However a
                 legitimate signal should be different giving rise to a higher score value.
                 
        Score 6. Ratio of the maximum values of Gaussian and fixed Gaussian fits to profile histogram.
                 The score compute the maximum height of the fixed Gaussian curve (mean fixed to the centre
                 bin) to the profile histogram, and the maximum height of the non-fixed Gaussian curve
                 to the profile histogram. This ratio will be equal to 1 for perfect noise, or close to zero
                 for legitimate pulsar emission.
        
        Score 7. Distance between expectation values of derivative histogram and profile histogram. A histogram
                 of profile derivatives is computed. This score finds the absolute value of the mean of the 
                 derivative histogram, minus the mean of the profile histogram. A value close to zero indicates 
                 a candidate arising from noise, a value greater than zero some form of legitimate signal.
        
        Score 8. Full-width-half-maximum (FWHM) of Gaussian fit to pulse profile. Describes the width of the
                 pulse, i.e. the width of the Gaussian fit of the pulse profile. Equal to 2*sqrt( 2 ln(2) )*sigma.
                 Not clear whether a higher or lower value is desirable.
        
        Score 9. Chi squared value from Gaussian fit to pulse profile. Lower values are indicators of a close fit,
                 and a possible profile source.
        
        Score 10. Smallest FWHM of double-Gaussian fit to pulse profile. Some pulsars have a doubly peaked
                  profile. This score fits two Gaussians to the pulse profile, then computes the FWHM of this
                  double Gaussian fit. Not clear if higher or lower values are desired.
        
        Score 11. Chi squared value from double Gaussian fit to pulse profile. Smaller values are indicators
                  of a close fit and possible pulsar source.

        Parameters:
        N/A
        
        Returns:
        
        Seven candidate scores.
        """
        
        try:
            guassian_fit = self.profileOps.getGaussianFittings(self.profile)
            
            self.scores.append(float(guassian_fit[0]))# Score 5. Distance between expectation values of Gaussian and fixed Gaussian fits to profile histogram.
            self.scores.append(float(guassian_fit[1]))# Score 6. Ratio of the maximum values of Gaussian and fixed Gaussian fits to profile histogram.
            self.scores.append(float(guassian_fit[2]))# Score 7. Distance between expectation values of derivative histogram and profile histogram.
            self.scores.append(float(guassian_fit[3]))# Score 8. Full-width-half-maximum (FWHM) of Gaussian fit to pulse profile. 
            self.scores.append(float(guassian_fit[4]))# Score 9. Chi squared value from Gaussian fit to pulse profile.
            self.scores.append(float(guassian_fit[5]))# Score 10. Smallest FWHM of double-Gaussian fit to pulse profile. 
            self.scores.append(float(guassian_fit[6]))# Score 11. Chi squared value from double Gaussian fit to pulse profile.
            
            if(self.debug==True):
                self.logger.debug("Score 5. Distance between expectation values of Gaussian and fixed Gaussian fits to profile histogram = %s", guassian_fit[0])
                self.logger.debug("Score 6. Ratio of the maximum values of Gaussian and fixed Gaussian fits to profile histogram = %s", guassian_fit[1])
                self.logger.debug("Score 7. Distance between expectation values of derivative histogram and profile histogram. = %s", guassian_fit[2])
                self.logger.debug("Score 8. Full-width-half-maximum (FWHM) of Gaussian fit to pulse profile = %s", guassian_fit[3])
                self.logger.debug("Score 9. Chi squared value from Gaussian fit to pulse profile = %s", guassian_fit[4])
                self.logger.debug("Score 10. Smallest FWHM of double-Gaussian fit to pulse profile = %s", guassian_fit[5])
                self.logger.debug("Score 11. Chi squared value from double Gaussian fit to pulse profile = %s", guassian_fit[6])

        except Exception as e: # catch *all* exceptions
            self.logger.error("Error computing scores 5-11 (Gaussian Fitting) \n\t%s", sys.exc_info()[0])
            self.logger.error(self.format_exception(e))
            raise Exception("Gaussian fitting exception")
    
    # ****************************************************************************************************
    
    def computeCandidateParameterScores(self):
        """
        Computes the candidate parameters. There are four scores computed:
        
        Score 12. The candidate period.
                 
        Score 13. The best signal-to-noise value obtained for the candidate. Higher values desired.
        
        Score 14. The best dispersion measure (dm) obtained for the candidate. Low DM values 
                  are assocaited with local RFI.
                 
        Score 15. The best pulse width.
                   
        Parameters:
        N/A
        
        Returns:
        
        Four candidate scores.
        """
        
        try:
            candidateParameters = self.profileOps.getCandidateParameters(self)
            
            self.scores.append(float(candidateParameters[0]))# Score 12. Best period.
            self.scores.append(self.filterScore(13,float(candidateParameters[1])))# Score 13. Best S/N value.
            self.scores.append(self.filterScore(14,float(candidateParameters[2])))# Score 14. Best DM value.
            self.scores.append(float(candidateParameters[3]))# Score 15. Best pulse width.
            
            if(self.debug==True):
                self.logger.debug("Score 12. Best period = %s", candidateParameters[0])
                self.logger.debug("Score 13. Best S/N value = %s Filtered value = %s", candidateParameters[1], self.filterScore(13,float(candidateParameters[1])))
                self.logger.debug("Score 14. Best DM value = %s Filtered value = %s", candidateParameters[2], self.filterScore(14,float(candidateParameters[2])))
                self.logger.debug("Score 15. Best pulse width = %s", candidateParameters[3])

        except Exception as e: # catch *all* exceptions
            self.logger.error("Error computing candidate parameters 12-15\n\t%s", sys.exc_info()[0])
            self.logger.error(self.format_exception(e))
            raise Exception("Candidate parameters exception")
    
    # ****************************************************************************************************
    
    def computeDMCurveFittingScores(self):
        """
        Computes the dispersion measure curve fitting parameters. There are four scores computed:
        
        Score 16. This score computes SNR / SQRT( (P-W) / W ).
                 
        Score 17. Difference between fitting factor Prop, and 1. If the candidate is a pulsar,
                  then prop should be equal to 1.
        
        Score 18. Difference between best DM value and optimised DM value from fit. This difference
                  should be small for a legitimate pulsar signal. 
                 
        Score 19. Chi squared value from DM curve fit, smaller values indicate a smaller fit. Thus
                  smaller values will be possessed by legitimate signals.
                   
        Parameters:
        N/A
        
        Returns:
        
        Four candidate scores.
        """
        
        try:
            DMCurveFitting = self.profileOps.getDMFittings(self)
            
            self.scores.append(float(DMCurveFitting[0]))# Score 16. SNR / SQRT( (P-W)/W ).
            self.scores.append(float(DMCurveFitting[1]))# Score 17. Difference between fitting factor, Prop, and 1.
            self.scores.append(self.filterScore(18,float(DMCurveFitting[2])))# Score 18. Difference between best DM value and optimised DM value from fit, mod(DMfit - DMbest).
            self.scores.append(float(DMCurveFitting[3]))# Score 19. Chi squared value from DM curve fit.
            
            if(self.debug==True):
                self.logger.debug("Score 16. SNR / SQRT( (P-W) / W ) = %s", DMCurveFitting[0])
                self.logger.debug("Score 17. Difference between fitting factor, Prop, and 1 = %s", DMCurveFitting[1])
                self.logger.debug("Score 18. Difference between best DM value and optimised DM value from fit, mod(DMfit - DMbest) = %s Filtered value = %s", DMCurveFitting[2], self.filterScore(18,float(DMCurveFitting[2])))
                self.logger.debug("Score 19. Chi squared value from DM curve fit = %s", DMCurveFitting[3])

        except Exception as e: # catch *all* exceptions
            self.logger.error("Error computing DM curve fitting 16-19\n\t%s", sys.exc_info()[0])
            self.logger.error(self.format_exception(e))
            raise Exception("DM curve fitting exception")
    
    # ****************************************************************************************************
    
    def computeSubBandScores(self):
        """
        Computes the sub-band scores. There are three scores computed:
        
        Score 20. RMS of peak positions in all sub-bands. Smaller values should be possessed by
                  legitimate pulsar signals.
                 
        Score 21. Average correlation coefficient for each pair of sub-bands. Larger values should be
                  possessed by legitimate pulsar signals.
        
        Score 22. Sum of correlation coefficients between sub-bands and profile. Larger values should be
                  possessed by legitimate pulsar signals.
                   
        Parameters:
        N/A
        
        Returns:
        
        Three candidate scores.
        """
        try:
            subbandScores = self.profileOps.getSubbandParameters(self,self.profile)
            
            self.scores.append(float(subbandScores[0]))# Score 20. RMS of peak positions in all sub-bands.
            self.scores.append(float(subbandScores[1]))# Score 21. Average correlation coefficient for each pair of sub-bands.
            self.scores.append(float(subbandScores[2]))# Score 22. Sum of correlation coefficients between sub-bands and profile.
            
            if(self.debug==True):
                self.logger.debug("Score 20. RMS of peak positions in all sub-bands = %s", subbandScores[0])
                self.logger.debug("Score 21. Average correlation coefficient for each pair of sub-bands = %s", subbandScores[1])
                self.logger.debug("Score 22. Sum of correlation coefficients between sub-bands and profile = %s", subbandScores[2])

        except Exception as e: # catch *all* exceptions
            self.logger.error("Error computing subband scores 20-22\n\t%s", sys.exc_info()[0])
            self.logger.error(self.format_exception(e))
            raise Exception("Subband scoring exception")
    
    # ****************************************************************************************************