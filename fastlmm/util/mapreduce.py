import logging
from fastlmm.util.runner import *
from contextlib import contextmanager
import threading

dyn = threading.local()

# from short example in http://stackoverflow.com/questions/2001138/how-to-create-dynamical-scoped-variables-in-python999
@contextmanager
def dyn_vars(**new):
    old = {}
    for name, value in new.items():
        old[name] = getattr(dyn, name, None)
        setattr(dyn, name, value)
    yield
    for name, value in old.items():
        setattr(dyn, name, value)

class MapReduce(object): #implements IDistributable
    """
    class to run distributed map using the idistributable back-end
    """


    def __init__(self, input_seq, mapper, nested, reducer, input_files=None, output_files=None, name=None):

        self.input_seq = input_seq
        self.mapper = mapper
        self.nested = nested
        if (self.mapper is not identity) and (self.nested is not None):
            raise Exception("'mapper' and 'nested' should not both be set")
        self.reducer = reducer
        self.name = name

        if input_files is None:
            self.input_files = []
        else:
            self.input_files = input_files

        if output_files is None:
            self.output_files = []
        else:
            self.output_files = output_files


#start of IDistributable interface--------------------------------------
    @property
    def work_count(self):
        return len(self.input_seq)

    def work_sequence_range(self, start, end):
        for i in xrange(start,end):
            input_arg = self.input_seq[i]
            if self.nested is None:
                logging.debug("random access executing %i" % i)
                with dyn_vars(is_in_nested=False):
                    yield lambda i=i, input_arg=input_arg: self.dowork(i, input_arg)   # the 'i=i',etc is need to get around a strangeness in Python
            else:
                assert self.nested is not None, "real assert"
                with dyn_vars(is_in_nested=True):
                    dist = apply(self.nested, [input_arg])
                    yield dist

    def work_sequence(self):
        for i, input_arg in enumerate(self.input_seq):
            if self.nested is None:
                logging.debug("executing %i" % i)
                with dyn_vars(is_in_nested=False):
                    yield lambda i=i, input_arg=input_arg: self.dowork(i, input_arg)  # the 'i=i',etc is need to get around a strangeness in Python
            else:
                assert self.nested is not None, "real assert"
                with dyn_vars(is_in_nested=True):
                    dist = apply(self.nested, [input_arg])
                    yield dist


    def reduce(self, output_seq):
        '''
        '''

        return self.reducer(output_seq)


    #optional override -- the str name of the instance is used by the cluster as the job name
    def __str__(self):
        if self.name is None:
            return "map_reduce()"
        else:
            return self.name
 #end of IDistributable interface---------------------------------------

    def dowork(self, i, input_arg):
        #logging.info("{0}, {1}".format(len(train_snp_idx), len(test_snp_idx)))
        logging.debug("executing {0}".format(input_arg))
        work = lambda : apply(self.mapper, [input_arg])
        result = run_all_in_memory(work)
        return result

   
    # required by IDistributable
    @property
    def tempdirectory(self):
        return ".work_directory.{0}".format(self.name)
        

    def copyinputs(self, copier):
        for fn in self.input_files:
            copier.input(fn)

    def copyoutputs(self,copier):
        for fn in self.output_files:
            copier.output(fn)

def identity(x):
    return x

def is_in_nested():
    return hasattr(dyn,"is_in_nested") and dyn.is_in_nested

def map_reduce(input_seq,mapper=identity,reducer=list,input_files=None, output_files=None,name=None,runner=None,nested=None):
    #!!! need docs
    if runner is None and nested is None and not is_in_nested():
        result = reducer(apply(mapper,[x]) for x in input_seq)
        return result

    dist = MapReduce(input_seq, mapper=mapper, nested=nested, reducer=reducer, input_files=input_files, output_files=output_files,name=name)
    if runner is None and is_in_nested():
        return dist

    if runner is None:
        runner = Local()

    result = runner.run(dist)
    return result
    
