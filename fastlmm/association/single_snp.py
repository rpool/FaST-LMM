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
from pysnptools.snpreader import wrap_matrix_subset
from pysnptools.util.intrangeset import IntRangeSet
from fastlmm.association.fastlmmmodel import _snps_fixup, _pheno_fixup, _kernel_fixup
            

def single_snp(test_snps, pheno, K0=None,
                 K1=None, mixing=None,
                 covar=None, output_file_name=None, h2=None, log_delta=None,
                 cache_file = None, GB_goal=None, interact_with_snp=None, force_full_rank=False, force_low_rank=False, G0=None, G1=None, runner=None):
    """
    Function performing single SNP GWAS with REML. Will reorder and intersect IIDs as needed.

    :param test_snps: SNPs to test. Can be any :class:`.SnpReader`. If you give a string, it should be the base name of a set of PLINK Bed-formatted files.
           (For backwards compatibility can also be dictionary with keys 'vals', 'iid', 'header')
    :type test_snps: a :class:`.SnpReader` or a string

    :param pheno: A single phenotype: Can be any :class:`.SnpReader`, for example, :class:`.Pheno` or :class:`.SnpData`.
           If you give a string, it should be the file name of a PLINK phenotype-formatted file.
           Any IIDs with missing values will be removed.
           (For backwards compatibility can also be dictionary with keys 'vals', 'iid', 'header')
    :type pheno: a :class:`.SnpReader` or a string

    :param K0: A similarity matrix. If not given, will use test_snps.
          If a :class:`.SnpReader` is given, the similarity matrix will be created from the SNP values.
          If you give a string, it should be the name of a :class:`.KernelNpz` file or base name of a set of PLINK Bed-formatted files.
    :type K0: a :class:`.KernelReader` or :class:`.SnpReader` or a string

    :param K1: A second similarity kernel, optional. (Also, see 'mixing').
          If a :class:`.SnpReader` is given, the similarity matrix will be created from the SNP values.
          If you give a string, it should be the name of a :class:`.KernelNpz` file or base name of a set of PLINK Bed-formatted files.
    :type K1: a :class:`.KernelReader` or :class:`.SnpReader` or a string

    :param mixing: Weight between 0.0 (inclusive, default) and 1.0 (inclusive) given to K1 relative to K0.
            If you give no mixing number and a K1 is given, the best weight will be learned.
    :type mixing: number

    :param covar: covariate information, optional: Can be any :class:`.SnpReader`, for example, :class:`.Pheno` or :class:`.SnpData`.
           If you give a string, it should be the file name of a PLINK phenotype-formatted file.
           (For backwards compatibility can also be dictionary with keys 'vals', 'iid', 'header')
    :type covar: a :class:`.SnpReader` or a string

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
                Calls using the same cache file should have the same 'K0' and 'K1'
                If given and the file does exist then K0 and K1 need not be given.
    :type cache_file: file name

    :param GB_goal: gigabytes of memory the run should use, optional. If not given, will read the test_snps in blocks the same size as the kernel,
        which is memory efficient with little overhead on computation time.
    :type GB_goal: number

    :param interact_with_snp: index of a covariate to perform an interaction test with. 
            Allows for interaction testing (interact_with_snp x snp will be tested)
            default: None

    :param force_full_rank: Even if kernels are defined with fewer SNPs than IIDs, create an explicit iid_count x iid_count kernel. Cannot be True if force_low_rank is True.
    :type force_full_rank: Boolean

    :param force_low_rank: Even if kernels are defined with fewer IIDs than SNPs, create a low-rank iid_count x sid_count kernel. Cannot be True if force_full_rank is True.
    :type force_low_rank: Boolean

    :param G0: Same as K0. Provided for backwards compatibility. Cannot be given if K0 is given.
    :type G0: a :class:`.KernelReader` or :class:`.SnpReader` or a string

    :param G1: Same as K1. Provided for backwards compatibility. Cannot be given if K1 is given.
    :type G1: a :class:`.KernelReader` or :class:`.SnpReader` or a string

    :param runner: a runner, optional: Tells how to run locally, multi-processor, or on a cluster.
        If not given, the function is run locally.
    :type runner: a runner.

    :rtype: Pandas dataframe with one row per test SNP. Columns include "PValue"

    :Example:

    >>> import logging
    >>> import numpy as np
    >>> from fastlmm.association import single_snp
    >>> from pysnptools.snpreader import Bed
    >>> logging.basicConfig(level=logging.INFO)
    >>> snpreader = Bed("../feature_selection/examples/toydata")
    >>> pheno_fn = "../feature_selection/examples/toydata.phe"
    >>> results_dataframe = single_snp(test_snps=snpreader[:,5000:10000],pheno=pheno_fn,K0=snpreader[:,0:5000],h2=.2,mixing=0)
    >>> print results_dataframe.iloc[0].SNP,round(results_dataframe.iloc[0].PValue,7),len(results_dataframe)
    null_7487 3.4e-06 5000

    """
    t0 = time.time()
    runner = runner or Local()
    if force_full_rank and force_low_rank:
        raise Exception("Can't force both full rank and low rank")

    test_snps, pheno, covar = _fixup_three(test_snps, pheno, covar)

    K0 = _kernel_fixup(K0 or G0 or test_snps, iid_if_none=test_snps.iid, standardizer=Unit())
    K1 = _kernel_fixup(K1 or G1, iid_if_none=test_snps.iid, standardizer=Unit())

    K0, K1, test_snps, pheno, covar  = pstutil.intersect_apply([K0, K1, test_snps, pheno, covar])
    logging.debug("# of iids now {0}".format(K0.iid_count))
    K0, K1, block_size = _set_block_size(test_snps, K0, K1, mixing, GB_goal, force_full_rank, force_low_rank)

    frame =  _internal_single(K0=K0, test_snps=test_snps, pheno=pheno,
                                covar=covar, K1=K1,
                                mixing=mixing, h2=h2, log_delta=log_delta,
                                cache_file = cache_file, force_full_rank=force_full_rank,force_low_rank=force_low_rank,
                                output_file_name=output_file_name,block_size=block_size, interact_with_snp=interact_with_snp,
                                runner=runner)

    sid_index_range = IntRangeSet(frame['sid_index'])
    assert sid_index_range == (0,test_snps.sid_count), "Some SNP rows are missing from the output"

    return frame

overhead_gig = .127
factor = 8.5  # found via trial and error

def _fixup_three(test_snps, pheno, covar):
    assert test_snps is not None, "test_snps must be given as input"
    test_snps = _snps_fixup(test_snps)
    pheno = _pheno_fixup(pheno).read()
    assert pheno.sid_count == 1, "Expect pheno to be just one variable"
    pheno = pheno[(pheno.val==pheno.val)[:,0],:]
    covar = _pheno_fixup(covar, iid_if_none=pheno.iid)
    return     test_snps, pheno, covar

def _GB_goal_from_block_size(block_size, iid_count, kernel_gig):
    left_bytes = block_size * (iid_count * 8.0 *factor)
    left_gig = left_bytes / 1024.0**3
    GB_goal = left_gig + overhead_gig + kernel_gig
    return GB_goal

def _block_size_from_GB_goal(GB_goal, iid_count, min_count):
    kernel_bytes = iid_count * min_count * 8
    kernel_gig = kernel_bytes / (1024.0**3)

    if GB_goal is None:
        GB_goal = _GB_goal_from_block_size(min_count, iid_count, kernel_gig)
        logging.info("Setting GB_goal to {0} GB".format(GB_goal))
        return min_count

    left_gig = GB_goal - overhead_gig - kernel_gig
    if left_gig <= 0:
        warnings.warn("The full kernel and related operations will likely not fit in the goal_memory")
    left_bytes = left_gig * 1024.0**3
    snps_at_once = left_bytes / (iid_count * 8.0 * factor)
    block_size = int(snps_at_once)

    if block_size < min_count:
        block_size = min_count
        GB_goal = _GB_goal_from_block_size(block_size, iid_count, kernel_gig)
        warnings.warn("Can't meet goal_memory without loading too few snps at once. Resetting GB_goal to {0} GB".format(GB_goal))

    return block_size

def single_snp_leave_out_one_chrom(test_snps, pheno,
                 K0=None, K1=None, mixing=None,
                 covar=None,covar_by_chrom=None,
                 output_file_name=None, h2=None, log_delta=None,
                 interact_with_snp=None, force_full_rank=False, force_low_rank=False, GB_goal=None, G0=None, G1=None, runner=None):
    """
    Function performing single SNP GWAS via cross validation over the chromosomes with REML. Will reorder and intersect IIDs as needed.

    :param test_snps: SNPs to test. Can be any :class:`.SnpReader`. If you give a string, it should be the base name of a set of PLINK Bed-formatted files.
           (For backwards compatibility can also be dictionary with keys 'vals', 'iid', 'header')
    :type test_snps: a :class:`.SnpReader` or a string

    :param pheno: A single phenotype: Can be any :class:`.SnpReader`, for example, :class:`.Pheno` or :class:`.SnpData`.
           If you give a string, it should be the file name of a PLINK phenotype-formatted file.
           Any IIDs missing values will be removed.
           (For backwards compatibility can also be dictionary with keys 'vals', 'iid', 'header')
    :type pheno: a :class:`.SnpReader` or a string

    :param K0: SNP data for creating a similarity matrix with each chromosome removed. If not given, will use test_snps.
          If you give a string, it should be the base name of a set of PLINK Bed-formatted files.
    :type K0: a class:`.SnpReader` or a string

    :param K1: SNP data for creating a similarity matrix with each chromosome removed. (Also, see 'mixing').
          If you give a string, it should be the base name of a set of PLINK Bed-formatted files.
    :type K1: a class:`.SnpReader` or a string

    :param mixing: Weight between 0.0 (inclusive, default) and 1.0 (inclusive) given to K1 relative to K0.
            If you give no mixing number and a K1 is given, the best weight will be learned.
    :type mixing: number

    :param covar: covariate information, optional: Can be any :class:`.SnpReader`, for example, :class:`.Pheno` or :class:`.SnpData`.
           If you give a string, it should be the file name of a PLINK phenotype-formatted file.
           (For backwards compatibility can also be dictionary with keys 'vals', 'iid', 'header')
    :type covar: a :class:`.SnpReader` or a string

    :param output_file_name: Name of file to write results to, optional. If not given, no output file will be created.
    :type output_file_name: file name

    :param h2: A parameter to LMM learning, optional
            If not given will search for best value.
            If mixing is unspecified, then h2 must also be unspecified.
    :type h2: number

    :param log_delta: a re-parameterization of h2 provided for backwards compatibility.
    :type log_delta: number

    :param GB_goal: gigabytes of memory the run should use, optional. If not given, will read the test_snps in blocks the same size as the kernel,
        which is memory efficient with little overhead on computation time.
    :type GB_goal: number

    :param interact_with_snp: index of a covariate to perform an interaction test with. 
            Allows for interaction testing (interact_with_snp x snp will be tested)
            default: None

    :param force_full_rank: Even if kernels are defined with fewer SNPs than IIDs, create an explicit iid_count x iid_count kernel. Cannot be True if force_low_rank is True.
    :type force_full_rank: Boolean

    :param force_low_rank: Even if kernels are defined with fewer IIDs than SNPs, create a low-rank iid_count x sid_count kernel. Cannot be True if force_full_rank is True.
    :type force_low_rank: Boolean

    :param G0: Same as K0. Provided for backwards compatibility. Cannot be given if K0 is given.
    :type G0: a :class:`.SnpReader` or a string

    :param G1: Same as K1. Provided for backwards compatibility. Cannot be given if K1 is given.
    :type G1: a :class:`.SnpReader` or a string

    :param runner: a runner, optional: Tells how to run locally, multi-processor, or on a cluster.
        If not given, the function is run locally.
    :type runner: a runner.

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
    runner = runner or Local()
    if force_full_rank and force_low_rank:
        raise Exception("Can't force both full rank and low rank")

    test_snps, pheno, covar = _fixup_three(test_snps, pheno, covar)

    chrom_list = list(set(test_snps.pos[:,0])) # find the set of all chroms mentioned in test_snps, the main testing data
    assert len(chrom_list) > 1, "single_leave_out_one_chrom requires more than one chromosome"

    input_files = [test_snps, pheno, K0, G0, K1, G1, covar] + ([] if covar_by_chrom is None else covar_by_chrom.values())

    def nested_closure(chrom):
        test_snps_chrom = test_snps[:,test_snps.pos[:,0]==chrom]
        covar_chrom = _create_covar_chrom(covar, covar_by_chrom, chrom)

        K0_chrom = _K_per_chrom(K0 or G0 or test_snps, chrom, test_snps.iid)
        K1_chrom = _K_per_chrom(K1 or G1, chrom, test_snps.iid)

        K0_chrom, K1_chrom, test_snps_chrom, pheno_chrom, covar_chrom  = pstutil.intersect_apply([K0_chrom, K1_chrom, test_snps_chrom, pheno, covar_chrom])
        logging.debug("# of iids now {0}".format(K0_chrom.iid_count))
        K0_chrom, K1_chrom, block_size = _set_block_size(test_snps_chrom, K0_chrom, K1_chrom, mixing, GB_goal, force_full_rank, force_low_rank)

        distributable = _internal_single(K0=K0_chrom, test_snps=test_snps_chrom, pheno=pheno_chrom,
                                    covar=covar_chrom, K1=K1_chrom,
                                    mixing=mixing, h2=h2, log_delta=log_delta, cache_file=None,
                                    force_full_rank=force_full_rank,force_low_rank=force_low_rank,
                                    output_file_name=None, block_size=block_size, interact_with_snp=interact_with_snp,
                                    runner=Local())
            
        return distributable

    def reducer_closure(frame_sequence):
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

    frame = map_reduce(chrom_list,
               mapper = nested_closure,
               reducer = reducer_closure,
               input_files = input_files,
               output_files = [output_file_name],
               name = "single_snp_leave_out_one_chrom, out='{0}'".format(output_file_name),
               runner = runner)

    return frame

def _K_per_chrom(K, chrom, iid):
    if K is None:
        return KernelIdentity(iid)
    else:
        K_all = _kernel_fixup(K, iid_if_none=iid, standardizer=Unit()) 
        if isinstance(K_all,SnpKernel):
            return SnpKernel(K_all.snpreader[:,K_all.pos[:,0] != chrom],Unit())
        else:
            raise Exception("Don't know how to make '{0}' work per chrom".format(K_all))

def _combine_the_best_way(K0, K1, covar, y, mixing, h2, force_full_rank=False, force_low_rank=False):
    #last return value tells if is_already_standardized is OK

    assert not(force_full_rank and force_low_rank), "real assert"

    if isinstance(K0,SnpKernel) and K0.sid_count == 0:
        warnings.warn("K0 is based on zero SNPs. Will treat as an identity kernel")
        K0 = KernelIdentity(K0.iid)
    if isinstance(K1,SnpKernel) and K1.sid_count == 0:
        warnings.warn("K1 is based on zero SNPs. Will treat as an identity kernel")
        K1 = KernelIdentity(K1.iid)


    ##########################
    # A special case: both kernels are the Identity so just return the first one
    ##########################
    if isinstance(K0,KernelIdentity) and isinstance(K1,KernelIdentity):
        warnings.warn("Both K0 and K1 are identity. Consider using linear regression instead.")
        return K0.read(), mixing or 0, h2 or 0, True 

    ##########################
    # Special cases: mixing says to use just one kernel or the other kernel is just identity, so just return one kernel
    ##########################
    if mixing == 0.0 or isinstance(K1,KernelIdentity):
        return K0, mixing or 0.0, h2, False

    if mixing == 1.0 or isinstance(K0,KernelIdentity):
        return K1, mixing or 1.0, h2, False

    ##########################
    # A special case: Treat the kernels as collections of snps (i.e. low-rank)
    ##########################
    if (isinstance(K0,SnpKernel) and isinstance(K1,SnpKernel) and not force_full_rank
        and (force_low_rank or K0.sid_count + K1.sid_count < K0.iid_count)):

        G = np.empty((K0.iid_count, K0.sid_count + K1.sid_count))
        # These two need to allocate their own memory so that they can be copied into G over and over again quickly
        G0_standardized_val = K0.read_snps().standardize(DiagKtoN()).val
        G1_standardized_val = K1.read_snps().standardize(DiagKtoN()).val

        if mixing is None:
            mixing, h2 = _find_mixing_from_Gs(G, covar, G0_standardized_val, G1_standardized_val, h2, y)
        _mix_from_Gs(G, G0_standardized_val,G1_standardized_val,mixing)
        
        snpdata = SnpData(iid=K0.iid,
                            sid=np.concatenate((K0.sid,K1.sid),axis=0),
                            val=G,parent_string="{0}&{1}".format(G0_standardized_val,G1_standardized_val),
                            pos=np.concatenate((K0.pos,K1.pos),axis=0)
                            )
        return SnpKernel(snpdata,StandardizerIdentity()), mixing, h2, True

    ##########################
    # The most general case, treat the new kernels as kernels (i.e.. full rank)
    ##########################
    # These two need to allocate their own memory so that they can be copied into K over and over again quickly (otherwise they might be reading from file over and over again or changing memory used for something else)
    K0_data = K0.read().standardize()
    K1_data = K1.read().standardize()
    K = np.empty(K0_data.val.shape)
    if mixing is None:
        mixing, h2 = _find_mixing_from_Ks(K, covar, K0_data.val, K1_data.val, h2, y)
    _mix_from_Ks(K, K0_data.val, K1_data.val, mixing)
    assert K.shape[0] == K.shape[1] and abs(np.diag(K).sum() - K.shape[0]) < 1e-7, "Expect mixed K to be standardized"
    return KernelData(val=K,iid=K0_data.iid), mixing, h2, True

def _set_block_size(test_snps, K0, K1, mixing, GB_goal, force_full_rank, force_low_rank):
    min_count = _internal_determine_block_size(K0, K1, mixing, force_full_rank, force_low_rank)
    block_size = _block_size_from_GB_goal(GB_goal, test_snps.iid_count, min_count)
    logging.info("Dividing SNPs by {0}".format(-(test_snps.sid_count//-block_size)))

    if isinstance(K0,SnpKernel):
        K0 = SnpKernel(K0.snpreader,K0.standardizer,block_size=block_size)
    if isinstance(K1,SnpKernel):
        K1 = SnpKernel(K1.snpreader,K1.standardizer,block_size=block_size)
    return K0, K1, block_size


def _internal_determine_block_size(K0, K1, mixing, force_full_rank, force_low_rank):
    assert not(force_full_rank and force_low_rank), "real assert"

    if isinstance(K0,SnpKernel) and K0.sid_count == 0:
        K0 = KernelIdentity(K0.iid)
    if isinstance(K1,SnpKernel) and K1.sid_count == 0:
        K1 = KernelIdentity(K1.iid)


    ##########################
    # A special case: both kernels are the Identity so just return the first one
    ##########################
    if isinstance(K0,KernelIdentity) and isinstance(K1,KernelIdentity):
        return K0.iid_count

    ##########################
    # Special cases: mixing says to use just one kernel or the other kernel is just identity, so just return one kernel
    ##########################
    if mixing == 0.0 or isinstance(K1,KernelIdentity):
        if isinstance(K0,SnpKernel) and not force_full_rank and (force_low_rank or K0.sid_count < K0.iid_count):
            return K0.sid_count
        else:
            return K0.iid_count

    if mixing == 1.0 or isinstance(K0,KernelIdentity):
        if isinstance(K1,SnpKernel) and not force_full_rank and (force_low_rank or K1.sid_count < K1.iid_count):
            return K1.sid_count
        else:
            return K1.iid_count

    ##########################
    # A special case: Treat the kernels as collections of snps (i.e. low-rank)
    ##########################
    if (isinstance(K0,SnpKernel) and isinstance(K1,SnpKernel) and not force_full_rank
        and (force_low_rank or K0.sid_count + K1.sid_count < K0.iid_count)):
        return K0.sid_count + K1.sid_count 

    ##########################
    # The most general case, treat the new kernels as kernels (i.e.. full rank)
    ##########################
    return K0.iid_count

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

def _internal_single(K0, test_snps, pheno, covar, K1,
                 mixing, h2, log_delta,
                 cache_file, force_full_rank,force_low_rank,
                 output_file_name, block_size, interact_with_snp, runner):
    assert K0 is not None, "real assert"
    assert K1 is not None, "real assert"
    assert block_size is not None, "real assert"
    assert mixing is None or 0.0 <= mixing <= 1.0
    if force_full_rank and force_low_rank:
        raise Exception("Can't force both full rank and low rank")

    assert h2 is None or log_delta is None, "if h2 is specified, log_delta may not be specified"
    if log_delta is not None:
        h2 = 1.0/(np.exp(log_delta)+1)

    covar = np.c_[covar.read(view_ok=True,order='A').val,np.ones((test_snps.iid_count, 1))]  #view_ok because np.c_ will allocation new memory

    y =  pheno.read(view_ok=True,order='A').val #view_ok because this code already did a fresh read to look for any missing values 

    if cache_file is not None and os.path.exists(cache_file):
        lmm = fastLMM(X=covar, Y=y, G=None, K=None)
        with np.load(cache_file) as data: #!! similar code in epistasis
            lmm.U = data['arr_0']
            lmm.S = data['arr_1']
            h2 = data['arr_2'][0]
            mixing = data['arr_2'][1]
    else:
        assert K0.iid0 is K0.iid1, "Expect K0 to be square"
        assert K1 is None or K1.iid0 is K1.iid1, "Expect K0 and K1 to be square"
        K, mixing, h2, is_already_standardized = _combine_the_best_way(K0,K1,covar,y,mixing,h2,force_full_rank=force_full_rank,force_low_rank=force_low_rank)
        if (isinstance(K,SnpKernel) and not force_full_rank and (force_low_rank or K.sid_count < K.iid_count)):
            if not is_already_standardized:
                G = K.read_snps().standardize(DiagKtoN())
            else:
                G = K.snpreader
            lmm = fastLMM(X=covar, Y=y, K=None, G=G.val, inplace=True)
        else:
            if not is_already_standardized:
                K = K.read().standardize()
            lmm = fastLMM(X=covar, Y=y, K=K.val, G=None, inplace=True)

        if h2 is None:
            result = lmm.findH2()
            h2 = result['h2']
        logging.info("h2={0}".format(h2))

        if cache_file is not None and not os.path.exists(cache_file):
            pstutil.create_directory_if_necessary(cache_file)
            lmm.getSU()
            np.savez(cache_file, lmm.U,lmm.S,np.array([h2,mixing])) #using np.savez instead of pickle because it seems to be faster to read and write

    if interact_with_snp is not None:
        logging.info("interaction with %i" % interact_with_snp)
        assert 0 <= interact_with_snp and interact_with_snp < covar.shape[1]-1, "interact_with_snp is out of range"
        interact = covar[:,interact_with_snp].copy()
        interact -=interact.mean()
        interact /= interact.std()
    else:
        interact = None

    work_count = -(test_snps.sid_count // -block_size) #Find the work count based on batch size (rounding up)

    # We define three closures, that is, functions define inside function so that the inner function has access to the local variables of the outer function.
    def debatch_closure(work_index):
        return test_snps.sid_count * work_index // work_count

    def mapper_closure(work_index):
        logging.info("Working on part {0} of {1}".format(work_index,work_count))
        do_work_time = time.time()
        start = debatch_closure(work_index)
        end = debatch_closure(work_index+1)

        snps_read = test_snps[:,start:end].read().standardize()
        if interact_with_snp is not None:
            variables_to_test = snps_read.val * interact[:,np.newaxis]
        else:
            variables_to_test = snps_read.val
        res = lmm.nLLeval(h2=h2, dof=None, scale=1.0, penalty=0.0, snps=variables_to_test)

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
    if covar_by_chrom is not None:
        covar_by_chrom_chrom = covar_by_chrom[chrom]
        covar_by_chrom_chrom = _pheno_fixup(covar_by_chrom_chrom, iid_if_none=covar)
        covar_after,  covar_by_chrom_chrom = pstutil.intersect_apply([covar,  covar_by_chrom_chrom])
        ret = SnpData(iid=covar_after.iid,sid=np.r_[covar_after.sid,covar_by_chrom_chrom.sid],
                      val=np.c_[covar_after.read(order='A',view_ok=True).val,
                                covar_by_chrom_chrom.read(order='A',view_ok=True).val]) #view_ok because np.c_ will allocate new memory.
        return ret
    else:
        return covar


def _find_mixing_from_Gs(G, covar, G0_standardized_val, G1_standardized_val, h2, y):
    logging.info("starting _find_mixing_from_Gs")
    import fastlmm.util.mingrid as mingrid
    assert h2 is None, "if mixing is None, expect h2 to also be None"
    resmin=[None]
    def f(mixing,G0_standardized_val=G0_standardized_val,G1_standardized_val=G1_standardized_val,covar=covar,y=y,**kwargs):

        if not isinstance(mixing, (int, long, float, complex)):
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
        return result['nLL']
    mixing,nLL = mingrid.minimize1D(f=f, nGrid=10, minval=0.0, maxval=1.0,verbose=False)

    if not isinstance(mixing, (int, long, float, complex)):
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

        if not isinstance(mixing, (int, long, float, complex)):
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
        return result['nLL']
    mixing,nLL = mingrid.minimize1D(f=f, nGrid=10, minval=0.0, maxval=1.0,verbose=False)

    if not isinstance(mixing, (int, long, float, complex)):
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
    K[:,:] = K0_val * (1.0-mixing) + K1_val * mixing

if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)

    import doctest
    doctest.testmod()

