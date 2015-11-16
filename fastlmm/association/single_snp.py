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
                 cache_file = None, G0=None, G1=None, force_full_rank=False, force_low_rank=False, batch_size=None, interact_with_snp=None, runner=None):
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

    if runner is None:
        runner = Local()

    assert (K0 is None) or (G0 is None), "Expect at least of one of K0 and G0 to be none"
    assert (K1 is None) or (G1 is None), "Expect at least of one of K1 and G1 to be none"

    from fastlmm.association.fastlmmmodel import _snps_fixup, _pheno_fixup, _kernel_fixup
    test_snps = _snps_fixup(test_snps)
    pheno = _pheno_fixup(pheno).read()
    assert pheno.sid_count == 1, "Expect pheno to be just one variable"
    pheno = pheno[(pheno.val==pheno.val)[:,0],:] #!!!cmk is this a good idea?: remove NaN's from pheno before intersections (this means that the inputs will be standardized according to this pheno)
    covar = _pheno_fixup(covar, iid_if_none=pheno.iid)

    chrom_set = set(test_snps.pos[:,0]) # find the set of all chroms mentioned in test_snps, the main testing data
    assert len(chrom_set) > 1, "single_leave_out_one_chrom requires more than one chromosome"
    frame_list = []
    for chrom in chrom_set: #!!!cmk use map_reduce here
        K0_chrom = _K_per_chrom(K0 or G0 or test_snps, chrom, test_snps.iid)
        K1_chrom = _K_per_chrom(K1 or G1, chrom, test_snps.iid)
        test_snps_chrom = test_snps[:,test_snps.pos[:,0]==chrom]
        covar_chrom = _create_covar_chrom(covar, covar_by_chrom, chrom)
        K0_chrom, K1_chrom, test_snps_chrom, pheno_chrom, covar_chrom  = pstutil.intersect_apply([K0_chrom, K1_chrom, test_snps_chrom, pheno, covar_chrom])
        logging.debug("# of iids now {0}".format(K0_chrom.iid_count))


        frame_chrom =  _internal_single(K0_standardized=K0_chrom, test_snps=test_snps_chrom, pheno=pheno_chrom,
                                    covar=covar_chrom, K1_standardized=K1_chrom,
                                    mixing=mixing, h2=h2, log_delta=log_delta,
                                    cache_file = cache_file, force_full_rank=force_full_rank,force_low_rank=force_low_rank,
                                    output_file_name=output_file_name,batch_size=batch_size, interact_with_snp=interact_with_snp,
                                    runner=runner)
        frame_list.append(frame_chrom)

    frame = pd.concat(frame_list)
    frame.sort("PValue", inplace=True)
    frame.index = np.arange(len(frame))

    if output_file_name is not None:
        frame.to_csv(output_file_name, sep="\t", index=False)

    logging.info("PhenotypeName\t{0}".format(pheno.sid[0]))
    logging.info("SampleSize\t{0}".format(test_snps.iid_count))
    logging.info("SNPCount\t{0}".format(test_snps.sid_count))
    logging.info("Runtime\t{0}".format(time.time()-t0))

    return frame

def _K_per_chrom(K, chrom, iid):
    from fastlmm.association.fastlmmmodel import _snps_fixup, _pheno_fixup, _kernel_fixup

    if isinstance(K,dict):
        K = _kernel_fixup(K[chrom], iid_if_none=iid, standardizer=Unit())
    elif K is None:
        return KernelIdentity(iid)
    else:
        K_all = _kernel_fixup(K, iid_if_none=iid, standardizer=Unit()) #!!!move this out of loop !!!cmk
        if isinstance(K_all,SnpKernel):
            return SnpKernel(K_all.snpreader[:,K_all.pos[:,0] != chrom],Unit())
        else:
            raise Exception("Don't know how to make '{0}' work per chrom".format(K_all))

#def _old_find_mixing(G, covar, G0_standardized_val, G1_standardized_val, h2, y):
#    import fastlmm.util.mingrid as mingrid
#    assert h2 is None, "if mixing is None, expect h2 to also be None"
#    resmin=[None]
#    def f(mixing,G0_standardized_val=G0_standardized_val,G1_standardized_val=G1_standardized_val,covar=covar,y=y,**kwargs):
#        _mix(G, G0_standardized_val,G1_standardized_val,mixing)
#        lmm = fastLMM(X=covar, Y=y, G=G, K=None, inplace=True)
#        result = lmm.findH2()
#        if (resmin[0] is None) or (result['nLL']<resmin[0]['nLL']):
#            resmin[0]=result
#        return result['nLL']
#    mixing,nLL = mingrid.minimize1D(f=f, nGrid=10, minval=0.0, maxval=1.0,verbose=False)
#    h2 = resmin[0]['h2']
#    return mixing, h2

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
    K0_data = K0_standardized.read().standardize() 
    K1_data = K1_standardized.read().standardize()
    K = np.empty(K0_data.val.shape)
    if mixing is None:
        mixing, h2 = _find_mixing_from_Ks(K, covar, K0_data.val, K1_data.val, h2, y)
    _mix_from_Ks(K, K0_data.val, K1_data.val, mixing)
    assert K.shape[0] == K.shape[1] and abs(np.diag(K).sum() - K.shape[0]) < 1e-7, "Expect mixed K to be standardized"
    from pysnptools.kernelreader import KernelData
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
                 output_file_name, batch_size, interact_with_snp, runner):
    assert mixing is None or 0.0 <= mixing <= 1.0
    if force_full_rank and force_low_rank:
        raise Exception("Can't force both full rank and low rank")

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
            mixing = data['arr_2'][1] if len(data['arr_2'])==2 else np.nan #!!!cmk remove this "if". It's here for temporary compatibility the with cache file
    else:
        assert K0_standardized.iid0 is K0_standardized.iid1 and K1_standardized.iid0 is K1_standardized.iid1, "Expect K0 and K1 to be square"
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

    if batch_size is None:
       batch_size = test_snps.sid_count
    work_count = -(test_snps.sid_count // -batch_size) #Find the work count based on batch size (rounding up)

    def debatch(work_index):
        return test_snps.sid_count * work_index // work_count

    def mapper(work_index):
        logging.info("Working on part {0} of {1}".format(work_index,work_count))
        do_work_time = time.time()
        start = debatch(work_index)
        end = debatch(work_index+1)

        snps_read = test_snps[:,start:end].read().standardize() #!!!could it be better to standardize train (if available) and test together?
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

        logging.info(dataframe)
        return dataframe

    def reducer(result_sequence):
        if output_file_name is not None:
            create_directory_if_necessary(output_file_name)

        frame = pd.concat(result_sequence)
        frame.sort("PValue", inplace=True)
        frame.index = np.arange(len(frame))

        if output_file_name is not None:
            frame.to_csv(output_file_name, sep="\t", index=False)

        return frame

    frame = map_reduce(xrange(work_count),
                       mapper=mapper,reducer=reducer,
                       input_files=[test_snps],output_files=[output_file_name],
                       name="single_snp(output_file={0})".format(output_file_name),
                       runner=runner)
    return frame


#def _old_internal_single(G0_standardized, test_snps, pheno,covar, G1_standardized,
#                 mixing, #!!test mixing and G1
#                 h2, log_delta,
#                 cache_file, interact_with_snp=None):

#    assert h2 is None or log_delta is None, "if h2 is specified, log_delta may not be specified"
#    if log_delta is not None:
#        h2 = 1.0/(np.exp(log_delta)+1)

#    covar = np.hstack((covar['vals'],np.ones((test_snps.iid_count, 1))))  #We always add 1's to the end.
#    y =  pheno['vals']

#    from pysnptools.standardizer import DiagKtoN

#    assert mixing is None or 0.0 <= mixing <= 1.0

#    if cache_file is not None and os.path.exists(cache_file):
#        lmm = fastLMM(X=covar, Y=y, G=None, K=None)
#        with np.load(cache_file) as data: #!! similar code in epistasis
#            lmm.U = data['arr_0']
#            lmm.S = data['arr_1']
#    else:
#        # combine two kernels (normalize kernels to diag(K)=N
#        G0_standardized_val = DiagKtoN(G0_standardized.val.shape[0]).standardize(G0_standardized.val)
#        G1_standardized_val = DiagKtoN(G1_standardized.val.shape[0]).standardize(G1_standardized.val)

#        if mixing == 0.0 or G1_standardized.sid_count == 0:
#            G = G0_standardized.val
#        elif mixing == 1.0 or G0_standardized.sid_count == 0:
#            G = G1_standardized.val
#        else:
#            G = np.empty((G0_standardized.iid_count,G0_standardized.sid_count+G1_standardized.sid_count))
#            if mixing is None:
#                mixing, h2 = _old_find_mixing(G, covar, G0_standardized_val, G1_standardized_val, h2, y)
#            _old_mix(G, G0_standardized_val,G1_standardized_val,mixing)
        
#        #TODO: make sure low-rank case is handled correctly
#        lmm = fastLMM(X=covar, Y=y, G=G, K=None, inplace=True)


#    if h2 is None:
#        result = lmm.findH2()
#        h2 = result['h2']
#    logging.info("h2={0}".format(h2))

#    snps_read = test_snps.read().standardize()
    
#    if interact_with_snp is not None:
#        logging.info("interaction with %i" % interact_with_snp)
#        assert 0 <= interact_with_snp and interact_with_snp < covar.shape[1]-1, "interact_with_snp is out of range"
#        interact = covar[:,interact_with_snp]
#        interact -=interact.mean()
#        interact /= interact.std()
#        variables_to_test = snps_read.val * interact[:,np.newaxis]
#    else:
#        variables_to_test = snps_read.val
#    res = lmm.nLLeval(h2=h2, dof=None, scale=1.0, penalty=0.0, snps=variables_to_test)

#    if cache_file is not None and not os.path.exists(cache_file):
#        pstutil.create_directory_if_necessary(cache_file)
#        np.savez(cache_file, lmm.U,lmm.S) #using np.savez instead of pickle because it seems to be faster to read and write


#    beta = res['beta']
        
#    chi2stats = beta*beta/res['variance_beta']
#    #p_values = stats.chi2.sf(chi2stats,1)[:,0]
#    if G0_standardized is not None:
#        assert G.shape[0] == lmm.U.shape[0]
#    p_values = stats.f.sf(chi2stats,1,lmm.U.shape[0]-3)[:,0]#note that G.shape is the number of individuals and 3 is the number of fixed effects (covariates+SNP)


#    items = [
#        ('SNP', snps_read.sid),
#        ('Chr', snps_read.pos[:,0]), 
#        ('GenDist', snps_read.pos[:,1]),
#        ('ChrPos', snps_read.pos[:,2]), 
#        ('PValue', p_values),
#        ('SnpWeight', beta[:,0]),
#        ('SnpWeightSE', np.sqrt(res['variance_beta'][:,0])),
#        ('SnpFractVarExpl', np.sqrt(res['fraction_variance_explained_beta'][:,0])),
#        ('Nullh2', np.zeros((snps_read.sid_count)) + h2)
#    ]
#    frame = pd.DataFrame.from_items(items)

#    return frame

#def _old_mix(G, G0_standardized_val, G1_standardized_val, mixing):
#    #logging.info("concat G1, mixing {0}".format(mixing))
#    G[:,0:G0_standardized_val.shape[1]] = G0_standardized_val
#    G[:,0:G0_standardized_val.shape[1]] *= (np.sqrt(1.0-mixing))
#    G[:,G0_standardized_val.shape[1]:] = G1_standardized_val
#    G[:,G0_standardized_val.shape[1]:] *= np.sqrt(mixing)


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


##!!!cmk
#def _old_snp_fixup(snp_input, iid_source_if_none=None):
#    if isinstance(snp_input, str):
#        return Bed(snp_input)
#    elif snp_input is None:
#        return iid_source_if_none[:,0:0] #return snpreader with no snps
#    else:
#        return snp_input

#def _old_pheno_fixup(pheno_input, iid_source_if_none=None):
#    import pysnptools.util.pheno as pstpheno
#    if isinstance(pheno_input, str):
#        return pstpheno.loadPhen(pheno_input) #!!what about missing=-9?

#    if pheno_input is None:
#        ret = {
#        'header':[],
#        'vals': np.empty((iid_source_if_none['vals'].shape[0], 0)),
#        'iid':iid_source_if_none['iid']
#        }
#        return ret

#    if len(pheno_input['vals'].shape) == 1:
#        ret = {
#        'header' : pheno_input['header'],
#        'vals' : np.reshape(pheno_input['vals'],(-1,1)),
#        'iid' : pheno_input['iid']
#        }
#        return ret

#    return pheno_input

# could this be written without the inside-out of IDistributable?
#class _SingleSnp(object) : #implements IDistributable

#    def create_output_dir_if_needed(self):
#        if self.output_file_or_none is not None:
#            from pysnptools.util import create_directory_if_necessary
#            create_directory_if_necessary(self.output_file_or_none)

#    def __init__(self,test_snps,mixing,h2,lmm,batch_size,output_file_or_none,interact_with_snp_covar):
#        self.test_snps = test_snps
#        self.mixing = mixing
#        self.h2 = h2
#        self.lmm = lmm
#        self.batch_size = batch_size
#        self.output_file_or_none=output_file_or_none
#        self.create_output_dir_if_needed()
#        self._str = "{0}(output_file={1})".format(self.__class__.__name__, output_file_or_none)
#        self._ran_once = False
#        self.interact_with_snp_covar = interact_with_snp_covar


#    def _run_once(self):
#        if self._ran_once:
#            return
#        self._ran_once = None

#        if self.batch_size is None:
#            self.batch_size = self.test_snps.sid_count

#        self._work_count = -(self.test_snps.sid_count // -self.batch_size) #Find the work count based on batch size (rounding up)

#        if self.output_file_or_none is None:
#            self.__tempdirectory = ".working"
#        else:
#            self.__tempdirectory = self.output_file_or_none + ".working"
        

# #start of IDistributable interface--------------------------------------
#    @property
#    def work_count(self):
#        self._run_once()
#        return self._work_count

#    #def work_sequence_range(self, start, end): #implement this efficiently

#    def work_sequence(self):
#        self._run_once()

#        start = 0
#        for work_index in xrange(self._work_count):
#            end = self.test_snps.sid_count * (1 + work_index) // self._work_count
#            yield lambda work_index=work_index,start=start,end=end : self.do_work(work_index,start,end)  # the 'start=start,...' is need to get around a strangeness in Python
#            start = end

#    #!!!cmk update
#    def reduce(self, result_sequence):
#        #doesn't need "run_once()"

#        self.create_output_dir_if_needed()

#        frame = pd.concat(result_sequence)

#        frame.sort("PValue", inplace=True)
#        frame.index = np.arange(len(frame))

#        if self.output_file_or_none is not None:
#            frame.to_csv(self.output_file_or_none, sep="\t", index=False)

#        #logging.info("PhenotypeName\t{0}".format(",".join(pheno.sid)))
#        #if K0 is not None:
#        #    logging.info("SampleSize\t{0}".format(len(K0.iid)))
#        #    logging.info("K0 \t{0}".format(K0))

#        #logging.info("Runtime\t{0}".format(time.time()-t0))



#        return frame

#    @property
#    def tempdirectory(self):
#        self._run_once()
#        return self.__tempdirectory

#    #optional override -- the str name of the instance is used by the cluster as the job name
#    def __str__(self):
#        #Doesn't need run_once
#        return self._str


#    def copyinputs(self, copier):
#        #Doesn't need run_once
#        copier.input(self.test_snps)

#    def copyoutputs(self,copier):
#        #Doesn't need run_once
#        copier.output(self.output_file_or_none)

# #end of IDistributable interface---------------------------------------

#    #!!!cmk what are these for?
#    do_pheno_count = 0
#    do_pheno_time = time.time()

#    def create_dataframe(self,row_count):
#        dataframe = pd.DataFrame(
#            index=np.arange(row_count),
#            columns=('sid_index', 'SNP', 'Chr', 'GenDist', 'ChrPos', 'PValue', 'SnpWeight', 'SnpWeightSE','SnpFractVarExpl','Mixing', 'Nullh2')
#            )
#        #!!Is this the only way to set types in a dataframe?
#        dataframe['sid_index'] = dataframe['sid_index'].astype(np.float)
#        dataframe['Chr'] = dataframe['Chr'].astype(np.float)
#        dataframe['GenDist'] = dataframe['GenDist'].astype(np.float)
#        dataframe['ChrPos'] = dataframe['ChrPos'].astype(np.float)
#        dataframe['ChrPos'] = dataframe['ChrPos'].astype(np.float)
#        dataframe['SnpWeight'] = dataframe['SnpWeight'].astype(np.float)
#        dataframe['SnpWeightSE'] = dataframe['SnpWeightSE'].astype(np.float)
#        dataframe['SnpFractVarExpl'] = dataframe['SnpFractVarExpl'].astype(np.float)
#        dataframe['Mixing'] = dataframe['Mixing'].astype(np.float)
#        dataframe['Nullh2'] = dataframe['Nullh2'].astype(np.float)

#        return dataframe

#    do_work_count = 0
#    do_work_time = time.time()

#    def do_work(self, work_index, start,end):
#        logging.info("Working on part {0} of {1}".format(work_index,self.work_count))

#        #!!!cmk need to test this
#        snps_read = self.test_snps[:,start:end].read().standardize() #!!!could it be better to standardize train (if available) and test together?
#        if self.interact_with_snp_covar is not None:
#            interact = self.interact_with_snp_covar.copy()
#            interact -=interact.mean()
#            interact /= interact.std()
#            variables_to_test = snps_read.val * interact[:,np.newaxis]
#        else:
#            variables_to_test = snps_read.val
#        #was res = self.lmm.nLLeval(h2=self.h2, dof=None, scale=1.0, penalty=0.0, snps=snps_read.val)
#        res = self.lmm.nLLeval(h2=self.h2, dof=None, scale=1.0, penalty=0.0, snps=variables_to_test)




#        beta = res['beta']
        
#        chi2stats = beta*beta/res['variance_beta']
#        #p_values = stats.chi2.sf(chi2stats,1)[:,0]
#        assert self.test_snps.iid_count == self.lmm.U.shape[0]
#        p_values = stats.f.sf(chi2stats,1,self.lmm.U.shape[0]-3)[:,0]#note that G.shape is the number of individuals and 3 is the number of fixed effects (covariates+SNP)

#        dataframe = self.create_dataframe(snps_read.sid_count)
#        dataframe['sid_index'] = np.arange(start,end)
#        dataframe['SNP'] = snps_read.sid
#        dataframe['Chr'] = snps_read.pos[:,0]
#        dataframe['GenDist'] = snps_read.pos[:,1]
#        dataframe['ChrPos'] = snps_read.pos[:,2] 
#        dataframe['PValue'] = p_values
#        dataframe['SnpWeight'] = beta[:,0]
#        dataframe['SnpWeightSE'] = np.sqrt(res['variance_beta'][:,0])
#        dataframe['SnpFractVarExpl'] = np.sqrt(res['fraction_variance_explained_beta'][:,0])
#        dataframe['Mixing'] = np.zeros((snps_read.sid_count)) + self.mixing
#        dataframe['Nullh2'] = np.zeros((snps_read.sid_count)) + self.h2

#        self.do_work_count += 1
#        if self.do_work_count % 1 == 0:
#            start = self.do_work_time
#            self.do_work_time = time.time()
#            logging.info("do_work_count={0}, time={1}".format(self.do_work_count,self.do_work_time-start))

#        logging.info(dataframe)
#        return dataframe

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

