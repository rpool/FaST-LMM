import numpy as np
import scipy as sp
import logging
import unittest
import os.path
import time
import sys
import doctest
import pandas as pd
import fastlmm.util.util as ut
from fastlmm.association.heritability_spatial_correction import heritability_spatial_correction
from fastlmm.util.runner import Local, HPC, LocalMultiProc
from pysnptools.snpreader import Dat, Bed, Pheno, SnpData
from fastlmm.feature_selection.test import TestFeatureSelection

tolerance = 1e-4


class TestHeritabilitySpatialCorrection(unittest.TestCase): #!!!cmk run these

    @classmethod
    def setUpClass(self):
        from fastlmm.util.util import create_directory_if_necessary
        create_directory_if_necessary(self.tempout_dir, isfile=False)
        self.pythonpath = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)),"..","..",".."))
        self.snpreader_whole = Bed(self.pythonpath + "/tests/datasets/synth/all")
        self.pheno_whole = Pheno(self.pythonpath + "/tests/datasets/synth/pheno_10_causals.txt")

    tempout_dir = "tempout/heritability_spatial_correction"

    def file_name(self,testcase_name):
        temp_fn = os.path.join(self.tempout_dir,testcase_name)
        if os.path.exists(temp_fn):
            os.remove(temp_fn)
        return temp_fn

    def test_one(self):
        '''
        Lock in results on arbitrary data -- because meaningful runs take too long to run.
        '''
        fn = "one.txt"
        logging.info(fn)
        tmpOutfile = self.file_name(fn)

        spatial_coor = [[i,-i] for i in xrange(self.snpreader_whole.iid_count)]
        alpha_list = alpha_list_big=[int(v) for v in np.logspace(2,np.log10(4000), 5)]
        dataframe = heritability_spatial_correction(self.snpreader_whole,spatial_coor,self.snpreader_whole.iid,alpha_list,self.pheno_whole,permute_plus_count=2,jackknife_count=2,permute_times_count=2,just_testing=True)

        dataframe.to_csv(tmpOutfile,sep="\t",index=False)
        referenceOutfile = TestFeatureSelection.reference_file("heritability_spatial_correction/"+fn)
        out,msg=ut.compare_files(tmpOutfile, referenceOutfile, tolerance)                
        self.assertTrue(out, "msg='{0}', ref='{1}', tmp='{2}'".format(msg, referenceOutfile, tmpOutfile))



    def test_doctest(self):
        old_dir = os.getcwd()
        os.chdir(os.path.dirname(os.path.realpath(__file__))+"/..")
        result = doctest.testfile("../heritability_spatial_correction.py")
        os.chdir(old_dir)
        assert result.failed == 0, "failed doc test: " + __file__



def getTestSuite():
    
    #!!!cmk add these to fastlmm main testing code
    suite1 = unittest.TestLoader().loadTestsFromTestCase(TestHeritabilitySpatialCorrection)
    return unittest.TestSuite([suite1])



if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)

    # this import is needed for the runner
    from fastlmm.association.tests.test_heritability_spatical_correction import TestHeritabilitySpatialCorrection
    suites = unittest.TestSuite([getTestSuite()])

    if True: #Standard test run #!!!cmk
        r = unittest.TextTestRunner(failfast=True) #!!!cmk fail fast=False
        r.run(suites)
    else: #Cluster test run
        from fastlmm.util.distributabletest import DistributableTest

        runner = HPC(10, 'RR1-N13-09-H44',r'\\msr-arrays\Scratch\msr-pool\Scratch_Storage4\Redmond',
                     remote_python_parent=r"\\msr-arrays\Scratch\msr-pool\Scratch_Storage4\REDMOND\carlk\Source\carlk\july_7_14\tests\runs\2014-07-24_15_02_02_554725991686\pythonpath",
                     update_remote_python_parent=True,
                     priority="AboveNormal",mkl_num_threads=1)
        runner = Local()
        #runner = LocalMultiProc(taskcount=20,mkl_num_threads=5)
        #runner = LocalInParts(1,2,mkl_num_threads=1) # For debugging the cluster runs
        #runner = Hadoop(100, mapmemory=8*1024, reducememory=8*1024, mkl_num_threads=1, queue="default")
        distributable_test = DistributableTest(suites,"temp_test")
        print runner.run(distributable_test)


    logging.info("done with testing")
