import logging
from fastlmm.util.runner import *

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
            #try: #Try random access
            for i in xrange(start,end):
                input_arg = self.input_seq[i]
                if self.nested is None:
                    logging.debug("random access executing %i" % i)#!!!cmk change to debug
                    yield lambda i=i, input_arg=input_arg: self.dowork(i, input_arg)
                    # the 'i=i',etc is need to get around a strangeness in Python
                else:
                    assert self.nested is not None, "real assert"
                    dist = apply(self.nested, [input_arg])
                    yield dist

            #except: #If that doesn't work, start at the front
            #    import itertools
            #    for l in islice(self.work_sequence(),start,end):
            #        yield l

    def work_sequence(self):
        for i, input_arg in enumerate(self.input_seq):
            if self.nested is None:
                logging.debug("executing %i" % i) #!!!cmk change to debug
                yield lambda i=i, input_arg=input_arg: self.dowork(i, input_arg)
                # the 'i=i',etc is need to get around a strangeness in Python
            else:
                assert self.nested is not None, "real assert"
                dist = apply(self.nested, [input_arg])
                yield dist


    def reduce(self, output_seq):
        '''
        '''

        return self.reducer(output_seq)


    #optional override -- the str name of the instance is used by the cluster as the job name
    def __str__(self):
        return "{0}{1}".format(self.mapper.__name__, self.name or "" ) #!!!cmk fix up
 #end of IDistributable interface---------------------------------------

    def dowork(self, i, input_arg):
        #logging.info("{0}, {1}".format(len(train_snp_idx), len(test_snp_idx)))
        logging.debug("executing %s" % str(input_arg))
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

#!!!cmk test output files
def map_reduce(input_seq,mapper=identity,nested=None,reducer=list,input_files=None, output_files=None,name=None,runner="simple"):
    #!!!cmk need docs
    if runner is "simple": #!!!change this to a class
        result = reducer(mapper(x) for x in input_seq)
    else:
        dist = MapReduce(input_seq, mapper=mapper, nested=nested, reducer=reducer, input_files=input_files, output_files=output_files,name=name)
        if runner is None:
            return dist
        else:
            result = runner.run(dist)
            return result
    return result
