"""
This code runs on python 2.4 or later.

By Rob Lyon <robert.lyon@cs.man.ac.uk>

+-----------------------------------------------------------------------------------------+
+                       PLEASE RECORD ANY MODIFICATIONS YOU MAKE BELOW                    +
+-----------------------------------------------------------------------------------------+
+ Revision |   Author    | Description                                       |    DATE    +
+-----------------------------------------------------------------------------------------+

 Revision:0    Rob Lyon    Initial version of the code.                        01/02/2014 

"""

import traceback
import sys, os, logging

class MultiColorFormatter(logging.Formatter):
    LOG_LEVEL_NUM = 25
    logging.addLevelName(LOG_LEVEL_NUM, "LOG")
    # Define colors
    COLORS = {
        'DEBUG': '\033[95m',    # Bright Magenta
        'INFO': '\033[96m',     # Bright Cyan
        'LOG': '\033[92m',      # Bright Green
        'WARNING': '\033[93m',  # Bright Yellow
        'ERROR': '\033[91m',    # Bright Red
        'CRITICAL': '\033[41m', # Red background
    }
    BOLD = '\033[1m'
    RESET = '\033[0m'

    def format(self, record):
        if record.levelname == 'INFO':
            color = self.COLORS[record.levelname]
            dt = color + self.formatTime(record, "%Y-%m-%d %H:%M:%S")
            lvl = record.levelname
            msg = self.RESET + record.getMessage()
        else:
            color = self.BOLD + self.COLORS[record.levelname]
            dt = color + self.formatTime(record, "%Y-%m-%d %H:%M:%S")
            lvl = record.levelname
            msg = record.getMessage() + self.RESET

        return f"{dt} # {lvl} # {msg}"

# **********************************************************************************************

class Utilities(object):
    """
    Provides utility functions used when computing scores.
    
    """
    
    # ******************************************************************************************

    def __init__(self, debugFlag: bool, logger_name: str):
        self.debug = debugFlag
        self.logger = logging.getLogger(logger_name)
    
    def appendToFile(self,path,text):
        """
        Appends the provided text to the file at the specified path.
        
        Parameters:
        path    -    the path to the file to append text to.
        text    -    the text to append to the file.
        
        Returns:
        N/A
        """
        
        destinationFile = open(path,'a')
        destinationFile.write(str(text))
        destinationFile.close()
    
    # ******************************************************************************************
    
    def fileExists(self,path):
        """
        Checks a file exists, returns true if it does, else false.
        
        Parameters:
        path    -    the path to the file to look for.
        
        Returns:
        True if the file exists, else false.
        """
        
        try:
            fh = open(path)
            fh.close()
            return True
        except IOError:
            return False
    
    # ******************************************************************************************
    
    def dirExists(self,path):
        """
        Checks a directory exists, returns true if it does, else false.
        
        Parameters:
        path    -    the path to the directory to look for.
        
        Returns:
        True if the file exists, else false.
        """
        
        try:
            if(os.path.isdir(path)):
                return True
            else:
                return False
        except IOError:
            return False
    
    # ******************************************************************************************
            
    def format_exception(self,e):
        """
        Formats error messages.
        
        Parameters:
        e    -    the exception.
        
        Returns:
        
        The formatted exception string.
        """
        exception_list = traceback.format_stack()
        exception_list = exception_list[:-2]
        exception_list.extend(traceback.format_tb(sys.exc_info()[2]))
        exception_list.extend(traceback.format_exception_only(sys.exc_info()[0], sys.exc_info()[1]))
        
        exception_str = "\nTraceback (most recent call last):\n"
        exception_str += "".join(exception_list)
        
        # Removing the last \n
        exception_str = exception_str[:-1]
        
        return exception_str
    
    # ******************************************************************************************
    
    def out(self, message, parameter):
        """
        Writes a debug statement out if the debug flag is set to true.
        
        Parameters:
        message    -    the string message to write out
        parameter  -    an accompanying parameter to write out.
        
        Returns:
        N/A
        """
        
        if(self.debug):
            self.logger.debug("%s %s", message, parameter)
            
    # ******************************************************************************************
    
    def outMutiple(self, parameters):
        """
        Writes a debug statement out if the debug flag is set to true.
        
        Parameters:
        parameters  -    the values to write out.
        
        Returns:
        N/A
        """
        
        if(self.debug):
            
            output =""
            for p in parameters:
                output+=str(p)

            self.logger.debug(output)

    # ******************************************************************************************