from fastlmm.util.runner import *
import logging
import fastlmm.pyplink.plink as plink
from pysnptools.snpreader import Pheno
import pysnptools.util as pstutil
import fastlmm.util.util as flutil
import numpy as np
import scipy.stats as stats
from pysnptools.snpreader import Bed
from fastlmm.util.pickle_io import load, save
import time
import pandas as pd
from fastlmm.inference.lmm_cov import LMM as fastLMM
import warnings
from pysnptools.snpreader import SnpReader
from pysnptools.snpreader import SnpData
from pysnptools.standardizer import Unit
from pysnptools.standardizer import Identity as StandardizerIdentity
from pysnptools.standardizer import DiagKtoN
from pysnptools.kernelreader import Identity as KernelIdentity
from pysnptools.kernelreader import KernelData
from pysnptools.kernelreader import SnpKernel
from pysnptools.kernelreader import KernelNpz
from fastlmm.util.mapreduce import map_reduce
from pysnptools.util import create_directory_if_necessary
from pysnptools.snpreader import wrap_matrix_subset #!!!cmk why does this need to be here for cluster to work
            


#!!!cmk test that it works with two identity matrices -- even if it doesn't do linearregression shortcut
def single_snp(test_snps, pheno,
                 K0=None,
                 K1=None, mixing=None, #!!!cmk c update comments, etc for G0->G0_or_K0
                 covar=None, output_file_name=None, h2=None, log_delta=None,
                 cache_file = None, G0=None, G1=None, force_full_rank=False, force_low_rank=False, batch_size=None, interact_with_snp=None, runner=None):
    """
    #!!!cmk document batch_size, etc
    Function performing single SNP GWAS with REML

    :param test_snps: SNPs to test. If you give a string, it should be the base name of a set of PLINK Bed-formatted files.
    :type test_snps: a :class:`.SnpReader` or a string

    :param pheno: A single phenotype: A 'pheno dictionary' contains an ndarray on the 'vals' key and a iid list on the 'iid' key.
      If you give a string, it should be the file name of a PLINK phenotype-formatted file.
    :type pheno: a 'pheno dictionary' or a string

    :param G0: SNPs from which to construct a similarity matrix.
          If you give a string, it should be the base name of a set of PLINK Bed-formatted files.
    :type G0: a :class:`.SnpReader` or a string
    #!!!cmk update
    :param G1: SNPs from which to construct a second similarity kernel, optional. Also, see 'mixing').
          If you give a string, it should be the base name of a set of PLINK Bed-formatted files.
    :type G1: a :class:`.SnpReader` or a string

    :param mixing: Weight between 0.0 (inclusive, default) and 1.0 (inclusive) given to G1 relative to G0.
            If you give no mixing number and a G1 is given, the best weight will be learned.
    :type mixing: number

    :param covar: covariate information, optional: A 'pheno dictionary' contains an ndarray on the 'vals' key and a iid list on the 'iid' key.
      If you give a string, it should be the file name of a PLINK phenotype-formatted file.
    :type covar: a 'pheno dictionary' or a string

    :param output_file_name: Name of file to write results to, optional. If not given, no output file will be created.
    :type output_file_name: file name

    :param h2: A parameter to LMM learning, optional
            If not given will search for best value.
            If mixing is unspecified, then h2 must also be unspecified.
    :type h2: number

    :param log_delta: a re-parameterization of h2 provided for backwards compatibility.
    :type log_delta: number


    :param cache_file: Name of  file to read or write cached precomputation values to, optional.
                If not given, no cache file will be used.
                If given and file does not exists, will write precomputation values to file.
                If given and file does exists, will read precomputation values from file.
                The file contains the U and S matrix from the decomposition of the training matrix. It is in Python's np.savez (*.npz) format.
                Calls using the same cache file should have the same 'G0' and 'G1'
                If given and the file does exist then G0 and G1 need not be given.
    :type cache_file: file name

    :param interact_with_snp: index of a covariate to perform an interaction test with. 
            Allows for interaction testing (interact_with_snp x snp will be tested)
            default: None

    :rtype: Pandas dataframe with one row per test SNP. Columns include "PValue"

    :Example:

    >>> import logging
    >>> import numpy as np
    >>> from fastlmm.association import single_snp
    >>> from pysnptools.snpreader import Bed
    >>> logging.basicConfig(level=logging.INFO)
    >>> snpreader = Bed("../feature_selection/examples/toydata")
    >>> pheno_fn = "../feature_selection/examples/toydata.phe"
    >>> results_dataframe = single_snp(test_snps=snpreader[:,5000:10000],pheno=pheno_fn,G0=snpreader[:,0:5000],h2=.2,mixing=0)
    >>> print results_dataframe.iloc[0].SNP,round(results_dataframe.iloc[0].PValue,7),len(results_dataframe)
    null_7487 3.4e-06 5000

    """
    t0 = time.time()

    if runner is None:
        runner = Local()
    if force_full_rank and force_low_rank:
        raise Exception("Can't force both full rank and low rank")

    from fastlmm.association.fastlmmmodel import _snps_fixup, _pheno_fixup, _kernel_fixup
    test_snps = _snps_fixup(test_snps)
    pheno = _pheno_fixup(pheno).read()
    assert pheno.sid_count == 1, "Expect pheno to be just one variable"
    pheno = pheno[(pheno.val==pheno.val)[:,0],:] #!!!cmk is this a good idea?: remove NaN's from pheno before intersections (this means that the inputs will be standardized according to this pheno)

    covar = _pheno_fixup(covar, iid_if_none=pheno.iid)
    K0 = _kernel_fixup(K0 or G0, iid_if_none=test_snps.iid, standardizer=Unit()) #!!!cmk document that will use test_snps if K0 (and G0) are not given
    K1 = _kernel_fixup(K1 or G1, iid_if_none=test_snps.iid, standardizer=Unit())

    K0, K1, test_snps, pheno, covar  = pstutil.intersect_apply([K0, K1, test_snps, pheno, covar]) #!!!cmk fix up util's intersect_apply and then use it instead
    logging.debug("# of iids now {0}".format(K0.iid_count))

    frame =  _internal_single(K0_standardized=K0, test_snps=test_snps, pheno=pheno,
                                covar=covar, K1_standardized=K1,
                                mixing=mixing, h2=h2, log_delta=log_delta,
                                cache_file = cache_file, force_full_rank=force_full_rank,force_low_rank=force_low_rank,
                                output_file_name=output_file_name,batch_size=batch_size, interact_with_snp=interact_with_snp,
                                runner=runner)

    from pysnptools.util.intrangeset import IntRangeSet
    sid_index_range = IntRangeSet(frame['sid_index'])
    assert sid_index_range == (0,test_snps.sid_count), "Some SNP rows are missing from the output"

    return frame

#!!might one need to pre-compute h2 for each chrom?
#!!clusterize????
def single_snp_leave_out_one_chrom(test_snps, pheno,
                 K0=None,
                 K1=None, mixing=None, #!!!cmk c update comments, etc for G0->G0_or_K0
                 covar=None,covar_by_chrom=None,
                 output_file_name=None, h2=None, log_delta=None,
                 cache_pattern = None, G0=None, G1=None, force_full_rank=False, force_low_rank=False, batch_size1=None, batch_size2=None, interact_with_snp=None, runner1=None, runner2=None):
    """
    Function performing single SNP GWAS via cross validation over the chromosomes with REML

    :param test_snps: SNPs to test and to construct similarity matrix.
          If you give a string, it should be the base name of a set of PLINK Bed-formatted files.
    :type test_snps: a :class:`.SnpReader` or a string

    :param pheno: A single phenotype: A 'pheno dictionary' contains an ndarray on the 'vals' key and a iid list on the 'iid' key.
      If you give a string, it should be the file name of a PLINK phenotype-formatted file.
    :type pheno: a 'pheno dictionary' or a string


    :param G1: SNPs from which to construct a second similarity matrix, optional. Also, see 'mixing').
          If you give a string, it should be the base name of a set of PLINK Bed-formatted files.
    :type G1: a :class:`.SnpReader` or a string

    :param mixing: Weight between 0.0 (inclusive, default) and 1.0 (inclusive) given to G1 relative to G0.
            If you give no mixing number, G0 will get all the weight and G1 will be ignored.
    :type mixing: number

    :param covar: covariate information, optional: A 'pheno dictionary' contains an ndarray on the 'vals' key and a iid list on the 'iid' key.
      If you give a string, it should be the file name of a PLINK phenotype-formatted file.
    :type covar: a 'pheno dictionary' or a string

    :param covar_by_chrom: covariate information, optional: A way to give different covariate information for each chromosome.
            It is a dictionary from chromosome number to a 'pheno dictionary' or a string
    :type covar_by_chrom: A dictionary from chromosome number to a 'pheno dictionary' or a string

    :param output_file_name: Name of file to write results to, optional. If not given, no output file will be created.
    :type output_file_name: file name

    :param h2: A parameter to LMM learning, optional
            If not given will search for best value.
            If mixing is unspecified, then h2 must also be unspecified.
    :type h2: number

    :param log_delta: a re-parameterization of h2 provided for backwards compatibility.
    :type log_delta: number

    :param interact_with_snp: index of a covariate to perform an interaction test with. 
            Allows for interaction testing (interact_with_snp x snp will be tested)
            default: None
    :rtype: Pandas dataframe with one row per test SNP. Columns include "PValue"

    :Example:

    >>> import logging
    >>> import numpy as np
    >>> from fastlmm.association import single_snp_leave_out_one_chrom
    >>> from pysnptools.snpreader import Bed
    >>> logging.basicConfig(level=logging.INFO)
    >>> pheno_fn = "../feature_selection/examples/toydata.phe"
    >>> results_dataframe = single_snp_leave_out_one_chrom(test_snps="../feature_selection/examples/toydata.5chrom", pheno=pheno_fn, h2=.2)
    >>> print results_dataframe.iloc[0].SNP,round(results_dataframe.iloc[0].PValue,7),len(results_dataframe)
    null_576 1e-07 10000

    """
    t0 = time.time()

    runner1 = runner1 or runner2 or Local()
    runner2 = runner2 or runner1 or Local()

    assert (K0 is None) or (G0 is None), "Expect at least of one of K0 and G0 to be none"
    assert (K1 is None) or (G1 is None), "Expect at least of one of K1 and G1 to be none"

    from fastlmm.association.fastlmmmodel import _snps_fixup, _pheno_fixup, _kernel_fixup
    test_snps = _snps_fixup(test_snps)
    pheno = _pheno_fixup(pheno).read()
    assert pheno.sid_count == 1, "Expect pheno to be just one variable"
    pheno = pheno[(pheno.val==pheno.val)[:,0],:] #!!!cmk is this a good idea?: remove NaN's from pheno before intersections (this means that the inputs will be standardized according to this pheno)
    covar = _pheno_fixup(covar, iid_if_none=pheno.iid)

    chrom_list = list(set(test_snps.pos[:,0])) # find the set of all chroms mentioned in test_snps, the main testing data
    assert len(chrom_list) > 1, "single_leave_out_one_chrom requires more than one chromosome"

    input_files1 = [test_snps, pheno, K0, K1, covar ,covar_by_chrom, G0, G1]
    chrom_to_cache_file = {chrom:(None if cache_pattern is None else cache_pattern.format(chrom)) for chrom in chrom_list}

    def files_for_chrom(chrom):
        logging.info("Working on chrom {0}".format(chrom))
        cache_file_chrom = chrom_to_cache_file[chrom]
        K0_chrom = _K_per_chrom(K0 or G0 or test_snps, chrom, test_snps.iid,block_size=batch_size1) #!!!cmk why is it called "batch_size" in some places and "block_size" in others.
        K1_chrom = _K_per_chrom(K1 or G1, chrom, test_snps.iid,block_size=batch_size1)
        test_snps_chrom = test_snps[:,test_snps.pos[:,0]==chrom]
        covar_chrom = _create_covar_chrom(covar, covar_by_chrom, chrom)
        K0_chrom, K1_chrom, test_snps_chrom, pheno_chrom, covar_chrom  = pstutil.intersect_apply([K0_chrom, K1_chrom, test_snps_chrom, pheno, covar_chrom])
        logging.debug("# of iids now {0}".format(K0_chrom.iid_count))
        return K0_chrom, K1_chrom, test_snps_chrom, pheno_chrom, covar_chrom, cache_file_chrom


    def nested_closure1(chrom):
        K0_chrom, K1_chrom, test_snps_chrom, pheno_chrom, covar_chrom, cache_file_chrom = files_for_chrom(chrom)

        _internal_single(cache_and_quit=True, K0_standardized=K0_chrom, test_snps=test_snps_chrom, pheno=pheno_chrom,
                                    covar=covar_chrom, K1_standardized=K1_chrom,
                                    mixing=mixing, h2=h2, log_delta=log_delta,
                                    cache_file = cache_file_chrom, force_full_rank=force_full_rank,force_low_rank=force_low_rank,
                                    output_file_name=None, batch_size=batch_size1, interact_with_snp=interact_with_snp) #!!!cmk should output_file_name be set here optionally?

        return chrom, cache_file_chrom

    if cache_pattern is not None:
        chrom_list_needed = [chrom for chrom in chrom_list if not os.path.exists(chrom_to_cache_file[chrom])]
        if len(chrom_list_needed) > 0:
            map_reduce(chrom_list_needed,
                            mapper=nested_closure1,
                            reducer=lambda l : {chrom : filename for (chrom, filename) in l},
                            input_files = input_files1, #!!!cmk covar_by_chrom needs some code, to pull the values from any dictionary
                            output_files = chrom_to_cache_file.values(),
                            name = "single_snp_leave_out_one_chrom, part 1, out='{0}'".format(cache_pattern), #!!!cmk test this on output none
                            runner = runner1)

    def nested_closure2(chrom):
        K0_chrom, K1_chrom, test_snps_chrom, pheno_chrom, covar_chrom, cache_file_chrom = files_for_chrom(chrom)
        #!!!cmk doesn't seem like cache_file and output_file_name should not be the same across all chroms
        distributable = _internal_single(K0_standardized=K0_chrom, test_snps=test_snps_chrom, pheno=pheno_chrom,
                                    covar=covar_chrom, K1_standardized=K1_chrom,
                                    mixing=mixing, h2=h2, log_delta=log_delta,
                                    cache_file = cache_file_chrom, force_full_rank=force_full_rank,force_low_rank=force_low_rank,
                                    output_file_name=None, batch_size=batch_size2, interact_with_snp=interact_with_snp) #!!!cmk should output_file_name be set here optionally?
            
        return distributable

    def reducer_closure2(frame_sequence):
        frame = pd.concat(frame_sequence)
        frame.sort_values(by="PValue", inplace=True)
        frame.index = np.arange(len(frame))
        if output_file_name is not None:
            frame.to_csv(output_file_name, sep="\t", index=False)
        logging.info("PhenotypeName\t{0}".format(pheno.sid[0]))
        logging.info("SampleSize\t{0}".format(test_snps.iid_count))
        logging.info("SNPCount\t{0}".format(test_snps.sid_count))
        logging.info("Runtime\t{0}".format(time.time()-t0))

        return frame

    input_files2 = input_files1+chrom_to_cache_file.values()
    frame = map_reduce(chrom_list,
               nested = nested_closure2,
               reducer = reducer_closure2, ###!!!cmk rename input_files and output_files to inputs and outputs
               input_files = input_files2, #!!!cmk covar_by_chrom needs some code, to pull the values from any dictionary
               output_files = [output_file_name],
               name = "single_snp_leave_out_one_chrom, part 2, out='{0}'".format(output_file_name), #!!!cmk test this on output none
               runner = runner2)

    return frame

def _K_per_chrom(K, chrom, iid, block_size): #!!!cmk is does regular single_snp use 'block_size'?
    from fastlmm.association.fastlmmmodel import _snps_fixup, _pheno_fixup, _kernel_fixup

    if isinstance(K,dict):
        K = _kernel_fixup(K[chrom], iid_if_none=iid, standardizer=Unit())
    elif K is None:
        return None
    else:
        K_all = _kernel_fixup(K, iid_if_none=iid, standardizer=Unit(),block_size=block_size) #!!!move this out of loop !!!cmk
        if isinstance(K_all,SnpKernel):
            return SnpKernel(K_all.snpreader[:,K_all.pos[:,0] != chrom],Unit(),block_size=block_size)
        else:
            raise Exception("Don't know how to make '{0}' work per chrom".format(K_all))

#!!!cmk figureout when read's and read(view_ok=True) should be used
#!!!cmk add code to test force_full_rank=False, force_low_rank=False
def _combine_the_best_way(K0_standardized,K1_standardized,covar,y,mixing,h2,force_full_rank=False, force_low_rank=False):
    if force_full_rank and force_low_rank:
        raise Exception("Can't force both full rank and low rank")

    ##########################
    # A special case: both kernels are the Identity so just return the first one
    ##########################
    if isinstance(K0_standardized,KernelIdentity) and isinstance(K1_standardized,KernelIdentity):
        return K0_standardized, mixing or 0, h2 or 0, False

    ##########################
    # Special cases: mixing says to use just one kernel or the other kernel is just identity, so just return one kernel
    ##########################
    if mixing == 0.0 or isinstance(K1_standardized,KernelIdentity):
        return K0_standardized, mixing or 0.0, h2, False #!!!cmk is "read" needed because "standardize" will be inplace?

    if mixing == 1.0 or isinstance(K0_standardized,KernelIdentity):
        return K1_standardized, mixing or 1.0, h2, False #!!!cmk is "read" needed because "standardize" will be inplace?

    ##########################
    # A special case: Treat the kernels as collections of snps (i.e. low-rank)
    ##########################
    #!!!cmk define sid_count
    if (isinstance(K0_standardized,SnpKernel) and isinstance(K1_standardized,SnpKernel) and not force_full_rank
        and (force_low_rank or K0_standardized.sid_count + K1_standardized.sid_count < K0_standardized.iid_count)):

            G = np.empty((K0_standardized.iid_count, K0_standardized.sid_count + K1_standardized.sid_count))
            # These two need to allocate their own memory so that they can be copied into G over and over again quickly
            G0_standardized_val = K0_standardized.read_snps().standardize(DiagKtoN()).val
            G1_standardized_val = K1_standardized.read_snps().standardize(DiagKtoN()).val

            if mixing is None:
                mixing, h2 = _find_mixing_from_Gs(G, covar, G0_standardized_val, G1_standardized_val, h2, y)
            _mix_from_Gs(G, G0_standardized_val,G1_standardized_val,mixing)
        
            snpdata = SnpData(iid=K0_standardized.iid,
                              sid=np.concatenate((K0_standardized.sid,K1_standardized.sid),axis=0),
                              val=G,parent_string="{0}&{1}".format(G0_standardized_val,G1_standardized_val),
                              pos=np.concatenate((K0_standardized.pos,K1_standardized.pos),axis=0)
                              )
            return SnpKernel(snpdata,StandardizerIdentity()), mixing, h2, True

    ##########################
    # The most general case, treat the new kernels as kernels (i.e.. full rank)
    ##########################
    # These two need to allocate their own memory so that they can be copied into K over and over again quickly (otherwise they might be reading from file over and over again or changing memory used for something else)
    K0_data = K0_standardized.read().standardize() #!!!cmk if K0_standardized is 'standardized' why does it need to be done again?
    if K1_standardized is None:
        K = K0_data.val
        mixing = 0
    else:
        K1_data = K1_standardized.read().standardize()
        K = np.empty(K0_data.val.shape)
        if mixing is None:
            mixing, h2 = _find_mixing_from_Ks(K, covar, K0_data.val, K1_data.val, h2, y)
        _mix_from_Ks(K, K0_data.val, K1_data.val, mixing)
        assert K.shape[0] == K.shape[1] and abs(np.diag(K).sum() - K.shape[0]) < 1e-7, "Expect mixed K to be standardized"
    from pysnptools.kernelreader import KernelData #!!!cmk what is this needed here and not just at the top?
    return KernelData(val=K,iid=K0_data.iid), mixing, h2, True

def _create_dataframe(row_count):
    dataframe = pd.DataFrame(
        index=np.arange(row_count),
        columns=('sid_index', 'SNP', 'Chr', 'GenDist', 'ChrPos', 'PValue', 'SnpWeight', 'SnpWeightSE','SnpFractVarExpl','Mixing', 'Nullh2')
        )
    #!!Is this the only way to set types in a dataframe?
    dataframe['sid_index'] = dataframe['sid_index'].astype(np.float)
    dataframe['Chr'] = dataframe['Chr'].astype(np.float)
    dataframe['GenDist'] = dataframe['GenDist'].astype(np.float)
    dataframe['ChrPos'] = dataframe['ChrPos'].astype(np.float)
    dataframe['ChrPos'] = dataframe['ChrPos'].astype(np.float)
    dataframe['SnpWeight'] = dataframe['SnpWeight'].astype(np.float)
    dataframe['SnpWeightSE'] = dataframe['SnpWeightSE'].astype(np.float)
    dataframe['SnpFractVarExpl'] = dataframe['SnpFractVarExpl'].astype(np.float)
    dataframe['Mixing'] = dataframe['Mixing'].astype(np.float)
    dataframe['Nullh2'] = dataframe['Nullh2'].astype(np.float)

    return dataframe

def _internal_single(K0_standardized, test_snps, pheno, covar, K1_standardized,
                 mixing, #!!test mixing and G1
                 h2, log_delta,
                 cache_file, force_full_rank,force_low_rank,
                 output_file_name, batch_size, interact_with_snp, cache_and_quit=False, runner=None):
    assert mixing is None or 0.0 <= mixing <= 1.0
    if force_full_rank and force_low_rank:
        raise Exception("Can't force both full rank and low rank")
    if cache_and_quit:
        assert cache_file is not None, "if 'cache_and_quit' option is given, 'cache_file' must be given, too"

    assert h2 is None or log_delta is None, "if h2 is specified, log_delta may not be specified"
    if log_delta is not None:
        h2 = 1.0/(np.exp(log_delta)+1)

    covar = np.hstack((covar.read(view_ok=True).val,np.ones((test_snps.iid_count, 1))))  #We always add 1's to the end.

    y =  pheno.read(view_ok=True).val

    if cache_file is not None and os.path.exists(cache_file):
        lmm = fastLMM(X=covar, Y=y, G=None, K=None)
        with np.load(cache_file) as data: #!! similar code in epistasis
            lmm.U = data['arr_0']
            lmm.S = data['arr_1']
            h2 = data['arr_2'][0]
            mixing = data['arr_2'][1]
    else: #!!!need to best regular single_snp on K1=None and confirm that no mixing happens
        assert K0_standardized.iid0 is K0_standardized.iid1, "Expect K0 to be square"
        assert K1_standardized is None or K1_standardized.iid0 is K1_standardized.iid1, "Expect K0 and K1 to be square"
         #!!!cmk if this is slow then having it before the outer loop will require it being processed over and over and over again
        K, mixing, h2, in_place_ok = _combine_the_best_way(K0_standardized,K1_standardized,covar,y,mixing,h2,force_full_rank=force_full_rank,force_low_rank=force_low_rank)
        #!!!cmk if both kernels are None (or identity) should just call linear regression
        if (isinstance(K,SnpKernel) and not force_full_rank and (force_low_rank or K.sid_count < K.iid_count)):
            if not in_place_ok:
                G = K.read_snps().standardize(DiagKtoN())
            else:
                G = K.read_snps(view_ok=True)
            lmm = fastLMM(X=covar, Y=y, K=None, G=G.val, inplace=True)
        else:
            if not in_place_ok:
                K = K.read().standardize()
            lmm = fastLMM(X=covar, Y=y, K=K.val, G=None, inplace=True)

        if h2 is None:
            result = lmm.findH2()  #!!!cmk if this is slow then having it before the outter loop will require it being processed over and over and over again
            h2 = result['h2']
        logging.info("h2={0}".format(h2))

        if cache_file is not None and not os.path.exists(cache_file):
            pstutil.create_directory_if_necessary(cache_file)
            lmm.getSU()
            np.savez(cache_file, lmm.U,lmm.S,np.array([h2,mixing])) #using np.savez instead of pickle because it seems to be faster to read and write

    if cache_and_quit:
        return

    if interact_with_snp is not None:
        logging.info("interaction with %i" % interact_with_snp)
        assert 0 <= interact_with_snp and interact_with_snp < covar.shape[1]-1, "interact_with_snp is out of range"
        interact = covar[:,interact_with_snp].copy()
        interact -=interact.mean()
        interact /= interact.std()
    else:
        interact = None

    if batch_size is None:
       batch_size = test_snps.sid_count #!!!cmk what's the point in having an inner loop is everything is done in one match?
    work_count = -(test_snps.sid_count // -batch_size) #Find the work count based on batch size (rounding up)

    # We define three closures, that is, functions define inside function so that the inner function has access to the local variables of the outer function.
    def debatch_closure(work_index):
        return test_snps.sid_count * work_index // work_count

    def mapper_closure(work_index):
        logging.info("Working on part {0} of {1}".format(work_index,work_count))
        do_work_time = time.time()
        start = debatch_closure(work_index)
        end = debatch_closure(work_index+1)

        snps_read = test_snps[:,start:end].read().standardize() #!!!could it be better to standardize train (if available) and test together?
        if interact_with_snp is not None:
            variables_to_test = snps_read.val * interact[:,np.newaxis]
        else:
            variables_to_test = snps_read.val
        res = lmm.nLLeval(h2=h2, dof=None, scale=1.0, penalty=0.0, snps=variables_to_test) #!!!the code assumes that this is the slowest bit. Is it?

        beta = res['beta']
        
        chi2stats = beta*beta/res['variance_beta']
        #p_values = stats.chi2.sf(chi2stats,1)[:,0]
        assert test_snps.iid_count == lmm.U.shape[0]
        p_values = stats.f.sf(chi2stats,1,lmm.U.shape[0]-3)[:,0]#note that G.shape is the number of individuals and 3 is the number of fixed effects (covariates+SNP)

        dataframe = _create_dataframe(snps_read.sid_count)
        dataframe['sid_index'] = np.arange(start,end)
        dataframe['SNP'] = snps_read.sid
        dataframe['Chr'] = snps_read.pos[:,0]
        dataframe['GenDist'] = snps_read.pos[:,1]
        dataframe['ChrPos'] = snps_read.pos[:,2] 
        dataframe['PValue'] = p_values
        dataframe['SnpWeight'] = beta[:,0]
        dataframe['SnpWeightSE'] = np.sqrt(res['variance_beta'][:,0])
        dataframe['SnpFractVarExpl'] = np.sqrt(res['fraction_variance_explained_beta'][:,0])
        dataframe['Mixing'] = np.zeros((snps_read.sid_count)) + mixing
        dataframe['Nullh2'] = np.zeros((snps_read.sid_count)) + h2

        logging.info("time={0}".format(time.time()-do_work_time))

        #logging.info(dataframe)
        return dataframe

    def reducer_closure(result_sequence):
        if output_file_name is not None:
            create_directory_if_necessary(output_file_name)

        frame = pd.concat(result_sequence)
        frame.sort_values(by="PValue", inplace=True)
        frame.index = np.arange(len(frame))

        if output_file_name is not None:
            frame.to_csv(output_file_name, sep="\t", index=False)

        return frame

    frame = map_reduce(xrange(work_count),
                       mapper=mapper_closure,reducer=reducer_closure,
                       input_files=[test_snps],output_files=[output_file_name],
                       name="single_snp(output_file={0})".format(output_file_name),
                       runner=runner)
    return frame



def _create_covar_chrom(covar, covar_by_chrom, chrom):
    from fastlmm.association.fastlmmmodel import _snps_fixup, _pheno_fixup, _kernel_fixup
    if covar_by_chrom is not None:
        covar_by_chrom_chrom = covar_by_chrom[chrom]
        covar_by_chrom_chrom = _pheno_fixup(covar_by_chrom_chrom, iid_if_none=covar)
        covar_after,  covar_by_chrom_chrom = pstutil.intersect_apply([covar,  covar_by_chrom_chrom])
        ret = SnpData(iid=covar_after.iid,sid=np.r_[covar_after.sid,covar_by_chrom_chrom.sid],val=np.c_[covar_after.read(order='A',view_ok=True).val,covar_by_chrom_chrom.read(order='A',view_ok=True).val])
        return ret
    else:
        return covar


def _find_mixing_from_Gs(G, covar, G0_standardized_val, G1_standardized_val, h2, y):
    logging.info("starting _find_mixing_from_Gs")
    import fastlmm.util.mingrid as mingrid
    assert h2 is None, "if mixing is None, expect h2 to also be None"
    resmin=[None]
    def f(mixing,G0_standardized_val=G0_standardized_val,G1_standardized_val=G1_standardized_val,covar=covar,y=y,**kwargs):

        if not isinstance(mixing, (int, long, float, complex)): #!!!cmk
            assert mixing.ndim == 1 and mixing.shape[0] == 1
            mixing = mixing[0]

        _mix_from_Gs(G, G0_standardized_val,G1_standardized_val,mixing)
        lmm = fastLMM(X=covar, Y=y, G=G, K=None, inplace=True)
        result = lmm.findH2()
        if (resmin[0] is None) or (result['nLL']<resmin[0]['nLL']):
            resmin[0]=result
        logging.info("mixing_from_Gs\t{0}\th2\t{1}\tnLL\t{2}".format(mixing,result['h2'],result['nLL']))
        #logging.info("reporter:counter:single_snp,find_mixing_from_Gs_count,1")
        assert not np.isnan(result['nLL']), "nLL should be a number (not a NaN)"
        return result['nLL'] #!!!cmk what does switching to -log likelihood to to single_snp?
    mixing,nLL = mingrid.minimize1D(f=f, nGrid=10, minval=0.0, maxval=1.0,verbose=False)

    if not isinstance(mixing, (int, long, float, complex)): #!!!cmk
        assert mixing.ndim == 1 and mixing.shape[0] == 1
        mixing = mixing[0]

    h2 = resmin[0]['h2']
    return mixing, h2

def _find_mixing_from_Ks(K, covar, K0_val, K1_val, h2, y):
    logging.debug("starting _find_mixing_from_Ks")
    import fastlmm.util.mingrid as mingrid
    assert h2 is None, "if mixing is None, expect h2 to also be None"
    resmin=[None]
    def f(mixing,K0_val=K0_val,K1_val=K1_val,covar=covar,y=y,**kwargs):

        if not isinstance(mixing, (int, long, float, complex)): #!!!cmk
            assert mixing.ndim == 1 and mixing.shape[0] == 1
            mixing = mixing[0]

        _mix_from_Ks(K, K0_val,K1_val,mixing)
        lmm = fastLMM(X=covar, Y=y, G=None, K=K, inplace=True)
        result = lmm.findH2()
        if (resmin[0] is None) or (result['nLL']<resmin[0]['nLL']):
            resmin[0]=result
        logging.debug("mixing_from_Ks\t{0}\th2\t{1}\tnLL\t{2}".format(mixing,result['h2'],result['nLL']))
        #logging.info("reporter:counter:single_snp,find_mixing_from_Ks_count,1")
        assert not np.isnan(result['nLL']), "nLL should be a number (not a NaN)"
        return result['nLL'] #!!!cmk what does switching to -log likelihood to to single_snp?
    mixing,nLL = mingrid.minimize1D(f=f, nGrid=10, minval=0.0, maxval=1.0,verbose=False)

    if not isinstance(mixing, (int, long, float, complex)): #!!!cmk
        assert mixing.ndim == 1 and mixing.shape[0] == 1
        mixing = mixing[0]

    h2 = resmin[0]['h2']
    return mixing, h2

def _mix_from_Gs(G, G0_standardized_val, G1_standardized_val, mixing):
    #logging.info("concat G1, mixing {0}".format(mixing))
    G[:,0:G0_standardized_val.shape[1]] = G0_standardized_val
    G[:,0:G0_standardized_val.shape[1]] *= (np.sqrt(1.0-mixing))
    G[:,G0_standardized_val.shape[1]:] = G1_standardized_val
    G[:,G0_standardized_val.shape[1]:] *= np.sqrt(mixing)

def _mix_from_Ks(K, K0_val, K1_val, mixing):
    K[:,:] = K0_val * (1.0-mixing) + K1_val * mixing #!!!cmk does this avoid memory allocation? Is there a way to avoid memory allocation?

if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)

    import doctest
    doctest.testmod()

