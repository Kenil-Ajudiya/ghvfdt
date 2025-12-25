"""
Script which generates scores for pulsar candidates. These scores are used as the
input features for machine learning classification algorithms. In total 22 scores
are generated from each individual candidate. Each score summarises a candidate
in some way.
  
This code runs on python 2.4 or later. I've tested the code to ensure that any changes
made here did not change the functionality of the original code. In other words the scores
output by this code are mathematically identical to those output by the original code
(unless improvements have been made). We know this for sure, since scores were recomputed
for a sample of candidates generated during the HTRU survey -
(score data stored at /local/scratch/cands). The scores generated using the original
script where exactly the same as those generated using this new script. 

Rob Lyon <robert.lyon@cs.man.ac.uk>

+-----------------------------------------------------------------------------------------+
+                       PLEASE RECORD ANY MODIFICATIONS YOU MAKE BELOW                    +
+-----------------------------------------------------------------------------------------+
+ Revision |   Author    | Description                                       |    DATE    +
+-----------------------------------------------------------------------------------------+

 Revision:0    Rob Lyon    Initial version of the re-written code.            05/02/2014
"""

import logging
from cyclopts import App, Parameter
from typing import Annotated

from Utilities import Utilities, MultiColorFormatter
from DataProcessor import DataProcessor

app = App("Candidate Score Generator",
          help="Generates scores for pulsar candidates from their HDF5 or PFD files.",
          version="1.0")

def configure_logger(name: str = "ML_classifier", level=logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not logger.handlers:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(MultiColorFormatter())
        logger.addHandler(stream_handler)
    return logger

@app.default
def main(
    candDir: Annotated[str, Parameter(name=["--candDir", "-c"], help="Path to directory containing candidates")] = "",
    outputPath: Annotated[str, Parameter(name=["--outputPath", "-o"], help="Path to write scores (file)")] = "scores.arff",
    hdf5: Annotated[bool, Parameter(name=["--hdf5"], help="Process only .hdf5 files")] = False,
    pfd: Annotated[bool, Parameter(name=["--pfd"], help="Process only .pfd files")] = False,
    arff: Annotated[bool, Parameter(name=["--arff"], help="Write candidate data to ARFF")] = False,
    profile: Annotated[bool, Parameter(name=["--profile"], help="Generate profile rather than score data")] = False,
    label: Annotated[bool, Parameter(name=["--label"], help="Enable interactive candidate labelling")] = False,
    dmprof: Annotated[bool, Parameter(name=["--dmprof"], help="Generate DM and profile summary stats")] = False,
    verbose: Annotated[bool, Parameter(name=["--verbose", "-v"], help="Verbose debugging output")] = False,
):
    """
    Generates 22 scores that describe the key features of pulsar candidate, from the
    candidate's own pfd or hdf5 file. The scores generated are as follows:
    
    Score number    Description of score                                                                                Group
        1            Chi squared value from fitting since curve to pulse profile.                                    Sinusoid Fitting
        2            Chi squared value from fitting sine-squared curve to pulse profile.                             Sinusoid Fitting
        
        3            Number of peaks the program identifies in the pulse profile - 1.                                Pulse Profile Tests
        4            Sum over residuals.                                                                             Pulse Profile Tests
        
        5            Distance between expectation values of Gaussian and fixed Gaussian fits to profile histogram.   Gaussian Fitting
        6            Ratio of the maximum values of Gaussian and fixed Gaussian fits to profile histogram.           Gaussian Fitting
        7            Distance between expectation values of derivative histogram and profile histogram.              Gaussian Fitting    
        8            Full-width-half-maximum (FWHM) of Gaussian fit to pulse profile.                                Gaussian Fitting
        9            Chi squared value from Gaussian fit to pulse profile.                                           Gaussian Fitting
        10           Smallest FWHM of double-Gaussian fit to pulse profile.                                          Gaussian Fitting
        11           Chi squared value from double Gaussian fit to pulse profile.                                    Gaussian Fitting
        
        12           Best period.                                                                                    Candidate Parameters
        13           Best SNR value.                                                                                 Candidate Parameters
        14           Best DM value.                                                                                  Candidate Parameters
        15           Best pulse width (original reported as Duty cycle (pulse width / period)).                      Candidate Parameters
        
        16           SNR / SQRT( (P-W)/W ).                                                                          Dispersion Measure (DM) Curve Fitting
        17           Difference between fitting factor, Prop, and 1.                                                 Dispersion Measure (DM) Curve Fitting
        18           Difference between best DM value and optimised DM value from fit, mod(DMfit - DMbest).          Dispersion Measure (DM) Curve Fitting
        19           Chi squared value from DM curve fit.                                                            Dispersion Measure (DM) Curve Fitting
        
        20           RMS of peak positions in all sub-bands.                                                         Sub-band Scores
        21           Average correlation coefficient for each pair of sub-bands.                                     Sub-band Scores
        22           Sum of correlation coefficients between sub-bands and profile.                                  Sub-band Scores
        
    Check out Sam Bates' thesis for more information, "Surveys Of The Galactic Plane For Pulsars" 2011.
    """
    logger_name = "ML_classifier"
    logger = configure_logger(logger_name)
    # Initialise variables with command line parameters.
    genProfileData = profile
    processSingleCandidate = False
    
    # Helper files.
    utils = Utilities(verbose, logger_name)
    dp = DataProcessor(verbose, logger_name)
    
    # Process -s argument if provided, make sure file the user
    # want to write to exists - otherwise we default to 
    # writing candidates to separate files.
    if(utils.fileExists(outputPath)):
        singleFile = True
    else:
        try:
            output = open(outputPath, 'w') # First try to create file.
            output.close()
        except IOError:
            pass
        
        # Now check again if it exists.
        if(utils.fileExists(outputPath)):
            singleFile = True
        else:
            singleFile = False # Must be an invalid path.
    
    # Process -c argument if provided, make sure directory containing
    # the candidates the user wants to process exists - otherwise we default to 
    # looking for candidates in the local directory. 
    if(utils.dirExists(candDir)):
        searchLocalDirectory = False
    elif(utils.fileExists(candDir)):
        searchLocalDirectory = False
        processSingleCandidate = True
    else:
        searchLocalDirectory = True
        
    # We have to determine the directory we would like to process. 
    if(searchLocalDirectory):
        search = ""
    elif(processSingleCandidate==False):
        # We add a / here as the method we call next will then search the directory
        # by appending *.pfd. Without the additional / we would
        # only look for directories that end with .pfd etc.
        search = candDir+"/"
    elif(processSingleCandidate==True):
        search = candDir
        
    logger.info("***********************************")
    logger.info("| Executing score generation code |")
    logger.info("***********************************")
    logger.info("\tCommand line arguments:")
    logger.info(f"\tDebug:{verbose}")
    logger.info(f"\tWrite to single file:{singleFile}")
    logger.info(f"\tOutput path:{outputPath}")
    logger.info(f"\tExpect HDF5 files:{hdf5}")
    logger.info(f"\tExpect PFD files:{pfd}")
    logger.info(f"\tProduce ARFF file:{arff}")
    logger.info(f"\tCandidate directory:{candDir}")
    logger.info(f"\tProcess single candidate:{processSingleCandidate}")
    logger.info(f"\tLabel candidates:{label}")
    logger.info(f"\tGenerate DM and profile stats as scores only:{dmprof}")
    logger.info(f"\tSearch local directory:{searchLocalDirectory}")
    
    if(label):
        if(pfd):
            dp.labelPFD(search, verbose)
            
    elif(dmprof):
        if(hdf5):
            dp.dmprofHDF5(search, verbose, outputPath, arff, processSingleCandidate)
        elif(pfd):
            dp.dmprofPFD(search, verbose, outputPath, arff, processSingleCandidate)
    elif(pfd):
        if(not singleFile):
            logger.info("Processing .pfd files and writing their scores to separate files.")
            dp.processPFDSeparately(search, verbose, processSingleCandidate)
        else:
            logger.info(f"Processing .pfd files and writing their scores to: {outputPath}")
            dp.processPFDCollectively(search, verbose, outputPath, arff, genProfileData, processSingleCandidate)
    else:
        logger.error("Don't know what to do with your input.")
        exit(1)

    logger.info("Done.")

    # ****************************************************************************************************
      
if __name__ == '__main__':
    app()