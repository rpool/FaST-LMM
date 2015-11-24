import numpy as np
import logging
import unittest
import os
#from fastlmm.feature_selection import FeatureSelectionStrategy, load_snp_data
from pysnptools.snpreader import Bed,Pheno
from pysnptools.kernelreader import SnpKernel
from pysnptools.kernelreader import Identity as KernelIdentity
import pysnptools.util as pstutil
#from fastlmm.feature_selection.feature_selection_two_kernel import FeatureSelectionInSample
#from fastlmm.association import single_snp
#from pysnptools.standardizer import DiagKtoN,UnitTrained
#from fastlmm.inference.lmm import LMM
from pysnptools.util import intersect_apply
from pysnptools.snpreader import SnpData,SnpReader
from pysnptools.standardizer import Unit
from pysnptools.kernelreader import KernelNpz

def _snps_fixup(snp_input, iid_if_none=None):
    if isinstance(snp_input, str):
        return Bed(snp_input)

    if isinstance(snp_input, dict):
        return SnpData(iid=snp_input['iid'],sid=snp_input['header'],val=snp_input['vals'])

    if snp_input is None:
        assert iid_if_none is not None, "snp_input cannot be None here"
        return SnpData(iid_if_none, sid=np.empty((0),dtype='str'), val=np.empty((len(iid_if_none),0)),pos=np.empty((0,3)),parent_string="") #todo: make a static factory method on SnpData

    return snp_input

def _pheno_fixup(pheno_input, iid_if_none=None, missing ='-9'):

    try:
        ret = Pheno(pheno_input, iid_if_none, missing=missing)
        ret.iid #doing this just to force file load
        return ret
    except:
        return _snps_fixup(pheno_input, iid_if_none=iid_if_none)


    return pheno_input

def _kernel_fixup(input, iid_if_none, standardizer, test=None, test_iid_if_none=None, block_size=None):
    if test is not None and input is None:
        input = test
        test = None

    if isinstance(input, str) and input.endswith(".npz"):
        return KernelNpz(input)

    if isinstance(input, str):
        input = Bed(input)     #Note that we don't return here. Processing continues
    if isinstance(test, str):
        test = Bed(test)      #Note that we don't return here. Processing continues

    if isinstance(input,SnpReader):
        return SnpKernel(input,standardizer=standardizer,test=test,block_size=block_size)

    if input is None:
        return KernelIdentity(iid=iid_if_none,test=test_iid_if_none)

    return input

