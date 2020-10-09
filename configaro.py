"""Module to provide full configuration flexibility.
Uses both, ConfigParser and ArgumentParser to read in config files and command
line arguments. 

Attention: single 1 and 0 values for a configuration key will be interpreted as booleans

@author Eric "Dr. Dre" Drechsler (eric.drechsler@cern.ch)
"""

import sys,os
import pwd
from time import strftime
import logging
logging.basicConfig()
logger = logging.getLogger(__name__)

from argparse import ArgumentParser
from configparser import ConfigParser

def convertStringList(strList):
    newList=[]
    for it in strList:
        floatVal=None
        try:
            floatVal=float(it)
        except:
            pass
        if floatVal:
            floatVal=int(floatVal) if floatVal.is_integer() else floatVal
            newList.append(floatVal)
        else:
            newList.append(val)
    return newList

class Configaro(object):
    """
    Central configuration class. Reads in list of cfg files and cmd line args. Cmd line args are prioritised.
    """
    def __init__(self, name='Configaro'):
        self.name = name
        
        argParser = ArgumentParser(add_help=False)
        argParser.add_argument( '-c', 
                '--configFiles', 
                required=True,
                help='Configuration file',
                default=['{0}/configs/example.cfg'.format(os.environ.get('PWD'))],
                )
        argParser.add_argument( '-d', '--debug', help='Activate Debug Mode', action='store_true')
        argParser.add_argument( '-h', '--helpMessage', help='Print Help message', action='store_true')
        argParser.add_argument( '-t', '--testRun', help='Set test run mode', action='store_true')
        argParser.add_argument( '-s', '--section', help='Read only specific sections in cfg file', default=None)
        self._parseArgs(argParser)

        if self.defaultArgs.debug:
            logging.root.setLevel( logging.DEBUG )
        
        #strict does not allow for duplicated entries!
        self.cfgParser = ConfigParser()
        #set case sensitive cfg file
        self.cfgParser.optionxform = str
        #do not allow for duplicated entries
        self.cfgParser.strict = False

        #get list of cfg files
        cfgFiles = self.defaultArgs.configFiles.split(',')
        try:
            self._parseCfgFile(cfgFiles, sections=self.defaultArgs.section)
        except:
            logger.error("Cannot parse with {0}".format(cfgFiles))
            raise 
        #merge config file dict and argparse dict, prioritise command line arguments
        self._createConfig()

        self.user=pwd.getpwuid(os.getuid()).pw_name
        return

    def __str__(self):
        configDump=""
        for key,val in sorted(self.__dict_.items()):
            configDump+='{0} = {1}\n'.format(key,val)
        return "Configuration:\n{0}".format(configDump)
    
    @property
    def config(self):
        return self.__dict__
    
    @property
    def allConfigKeys(self):
        return self.__dict__.items()
    
    #helper for pickling. Return value is pickled instead of self.__dict__. Allows for __getattr__() definition
    def __getstate__(self):
        return self.__dict__
    
    def __setstate__(self,state):
        self.__dict__=state
        return self.__dict__ 
    
    #important - this enables us to use dictionary internally but keep cfg.attribute logic
    def __getattr__(self, cfgEntry):
        try:
            return self.__dict__[cfgEntry]
        except KeyError:
            logger.error('Could not locate {0} in __dict__.'.format(cfgEntry))
            raise Exception('Could not locate {0} in __dict__.'.format(cfgEntry))

    def _parseArgs(self,argParser=None):
        try:
            #get args registered so far for configFile setting
            self.defaultArgs=argParser.parse_known_args()[0]
            self.addArgsDict=vars(self.defaultArgs)
            #get additionalArgs to be compared to config
            addArgsList=argParser.parse_known_args()[1]
            for i in range(0,len(addArgsList)):
                if not addArgsList[i].startswith('--'): continue
                nextArg=None
                try:
                    nextArg=addArgsList[i+1]
                except:
                    pass
                key=addArgsList[i].replace('--','')
                val=None
                #catch boolean arguments without further input
                if not nextArg or nextArg.startswith('--'):
                    val=1
                #catch argument with input 
                else:
                    #IMPORTANT: only create a list in config if argument splits by COMMA. otherwise assume one arg value.
                    #has consequences downstream for batchBuddha - automatic loop of list-config
                    val=nextArg.split(',') if ',' in nextArg else nextArg
            
                logger.debug("Configaro setting value 1/0 to True/False.")
                logger.debug("Warning: list of booleans not supported")
                val=True if val=='1' else (False if val=='0' else val)
                
                if val is not None:
                    self.addArgsDict[key]=val
                else:
                    logger.error("Configaro could not determine value of arg. Check validity.")
        except:
            logger.error("Configaro cannot parse arguments.")
            raise 
        logger.info("Successfully parsed default arguments")
        return
    
    def _parseCfgFile(self,cfgFiles,sections=None):
        #no effect: configparser automatically removes duplicates
        logger.warning("Attention: In case of duplicated cfg entry, last one will be prioritised.")
        self.cfgParser.read(cfgFiles)
        #consider all available sections in cfg file if not stated otherwise
        if not sections:
            sections=self.cfgParser.sections()
        for section in sections:
            for key,val in dict(self.cfgParser.items(section)).items():
                if ',' in val:
                    logger.debug("Splitting config list for - {0} : {1}".format(key,val))
                    val=val.split(',')
                    val=convertStringList(val)
                logger.debug("Setting value 1/0 to True/False.")
                logger.debug("Warning: list of booleans not supported")
                val=True if val=='1' else (False if val=='0' else val)
                
                #no effect: configparser automatically removes duplicates by favouring last entry
                if str(key) in self.__dict__.keys():
                    logger.error("Found duplicated entry in cfg-File(s). Ignoring second find {0}:{1}".format(key,val))
                    continue
                else:
                    logger.debug("Updating cfg-dict with  {0}:{1}".format(key,val))
                    floatVal=None
                    try:
                        floatVal=float(val)
                    except:
                        pass
                    if floatVal:
                        self.__dict__[key]=int(floatVal) if floatVal.is_integer() else floatVal
                    else:
                        self.__dict__[key]=val

        logger.info("Successfully parsed input  files {0}".format(cfgFiles))
        return

    def _createConfig(self):
        for key, val in self.addArgsDict.items():
            if not key in self.__dict__.keys():
                logger.warning("The key {0} is not in the configuration file. Adding it.".format(key))
            else:
                logger.warning("Updating cfg-file setting for {0}: {1}".format(key,val))
            self.__dict__[key]=val
        logger.info("Successfully merged file and cmd line configurations")
        return

if __name__=="__main__":
    c=Configaro()
    print(c)