import matplotlib #!!!cmk comment out
matplotlib.use('TkAgg')  #!!!cmk comment out
import pylab

import numpy as np
import logging
import unittest
import os.path
import doctest
import pandas as pd
import pysnptools.util as pstutil

from fastlmm.association.fastlmmmodel import FastLmmModel #!!!cmk put in __init__
from fastlmm.util.runner import Local, HPC, LocalMultiProc
from pysnptools.snpreader import Dat, Bed, Pheno
from fastlmm.feature_selection.test import TestFeatureSelection
from pysnptools.standardizer import Unit

class TestFastLmmModel(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        from fastlmm.util.util import create_directory_if_necessary
        create_directory_if_necessary(self.tempout_dir, isfile=False)
        self.pythonpath = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)),"..","..",".."))

        self.snpreader_whole = Bed(self.pythonpath + "/tests/datasets/synth/all")
        self.covariate_whole = Pheno(self.pythonpath + "/tests/datasets/synth/cov.txt") #!!!cmk be sure this file is in source control
        self.pheno_whole = Pheno(self.pythonpath + "/tests/datasets/synth/pheno_10_causals.txt")

    tempout_dir = "tempout/fastlmmmodel"

    def file_name(self,testcase_name):
        temp_fn = os.path.join(self.tempout_dir,testcase_name+".dat")
        if os.path.exists(temp_fn):
            os.remove(temp_fn)
        return temp_fn
    #!!!cmk add reference files to source control

    def test_one(self):
        logging.info("TestFastLmmModel test_one")

        train_idx = np.r_[10:self.snpreader_whole.iid_count] # iids 10 and on
        test_idx  = np.r_[0:10] # the first 10 iids

        G0_train = self.snpreader_whole[train_idx,:]
        covar_train = self.covariate_whole[train_idx,:]
        pheno_train = self.pheno_whole[train_idx,:]

        fastlmm_model1 = FastLmmModel.learn(G0_train, covar_train, pheno_train)
        filename = self.tempout_dir + "/model_one.flm.npz"
        pstutil.create_directory_if_necessary(filename)
        fastlmm_model1.save(filename)
        fastlmm_model2 = FastLmmModel.load(filename)
                
        # predict on test set
        G0_test = self.snpreader_whole[test_idx,:]
        covar_test = self.covariate_whole[test_idx,:]

        predicted_pheno = fastlmm_model2.predict(G0_test, covar_test)

        output_file = self.file_name("one")
        Dat.write(output_file,predicted_pheno)

        pheno_actual = self.pheno_whole[test_idx,:].read().val[:,0]

        #pylab.plot(pheno_actual, predicted_pheno.val,".")
        #pylab.show()


        self.compare_files(predicted_pheno,"one")

    def test_lr(self):
        logging.info("TestFastLmmModel test_lr")

        train_idx = np.r_[10:self.snpreader_whole.iid_count] # iids 10 and on
        test_idx  = np.r_[0:10] # the first 10 iids

        G0_train = self.snpreader_whole[train_idx,:]
        covar_train3 = self.covariate_whole[train_idx,:].read()
        covar_train3.val = np.array([[float(num)] for num in xrange(covar_train3.iid_count)])
        pheno_train3 = self.pheno_whole[train_idx,:].read()
        np.random.seed(0)
        pheno_train3.val = covar_train3.val * 2.0 + 100 + np.random.normal(size=covar_train3.val.shape) # y = 2*x+100+normal(0,1)

        ##Plot training x and y
        #pylab.plot(covar_train3.val, pheno_train3.val,".")
        #pylab.show()

        #Learn model, save, load
        fastlmm_model3x = FastLmmModel.learn(K0_train=G0_train, covar_train=covar_train3, pheno_train=pheno_train3)
        filename = self.tempout_dir + "/model_lr.flm.npz"
        pstutil.create_directory_if_necessary(filename)
        fastlmm_model3x.save(filename)
        fastlmm_model3 = FastLmmModel.load(filename)

        #Predict with model (test on train)
        predicted_pheno = fastlmm_model3.predict(K0_test=G0_train, covar_test=covar_train3) #test on train
        output_file = self.file_name("lr")
        Dat.write(output_file,predicted_pheno)

        ## Plot training x and y, and training x with predicted y
        #pylab.plot(covar_train3.val, pheno_train3.val,covar_train3.val,predicted_pheno.val,".")
        #pylab.show()

        ## Plot y and predicted y (test on train)
        #pheno_actual = pheno_train3.val[:,0]
        #pylab.plot(pheno_actual,predicted_pheno.val,".")
        #pylab.show()


        self.compare_files(predicted_pheno,"lr")

    #!!!cmk get this working
    def xtest_lr_no_K0(self):
        logging.info("TestFastLmmModel test_lr_no_k0")

        train_idx = np.r_[10:self.snpreader_whole.iid_count] # iids 10 and on
        test_idx  = np.r_[0:10] # the first 10 iids

        covar_train3 = self.covariate_whole[train_idx,:].read()
        covar_train3.val = np.array([[float(num)] for num in xrange(covar_train3.iid_count)])
        pheno_train3 = self.pheno_whole[train_idx,:].read()
        np.random.seed(0)
        pheno_train3.val = covar_train3.val * 2.0 + 100 + np.random.normal(size=covar_train3.val.shape) # y = 2*x+100+normal(0,1)

        #Learn model, save, load
        fastlmm_model3x = FastLmmModel.learn(covar_train=covar_train3, pheno_train=pheno_train3)
        filename = self.tempout_dir + "/model3.flm.npz"
        pstutil.create_directory_if_necessary(filename)
        fastlmm_model3x.save(filename)
        fastlmm_model3 = FastLmmModel.load(filename)

        #Predict with model (test on train)
        predicted_pheno = fastlmm_model3.predict(covar_test=covar_train3) #test on train
        output_file = self.file_name("lr_no_k0")
        Dat.write(output_file,predicted_pheno)

        self.compare_files(predicted_pheno,"lr_no_k0")

    def test_snps(self):
        logging.info("TestFastLmmModel test_snps")

        train_idx = np.r_[10:self.snpreader_whole.iid_count] # iids 10 and on
        test_idx  = np.r_[0:10] # the first 10 iids

        # Show it using the snps
        G0_train = self.snpreader_whole[train_idx,:]
        covar_train3 = self.covariate_whole[train_idx,:].read()
        pheno_train3 = self.pheno_whole[train_idx,:].read()
        pheno_train3.val = G0_train[:,0:1].read().val*2

        #pylab.plot(G0_train[:,0:1].read().val[:,0], pheno_train3.val[:,0],".")
        #pylab.show()

        #Learn model, save, load
        fastlmm_model3x = FastLmmModel.learn(K0_train=G0_train, covar_train=covar_train3, pheno_train=pheno_train3)
        filename = self.tempout_dir + "/model_snps.flm.npz"
        pstutil.create_directory_if_necessary(filename)
        fastlmm_model3x.save(filename)
        fastlmm_model3 = FastLmmModel.load(filename)

        #Predict with model (test on train)
        predicted_pheno = fastlmm_model3.predict(K0_test=G0_train, covar_test=covar_train3) #test on train
        output_file = self.file_name("snps")
        Dat.write(output_file,predicted_pheno)

        ### Plot training x and y, and training x with predicted y
        #pylab.plot(G0_train[:,0:1].read().val[:,0], pheno_train3.val,".",G0_train[:,0:1].read().val[:,0],predicted_pheno.val,".")
        #pylab.show()

        ### Plot y and predicted y (test on train)
        #pheno_actual = pheno_train3.val[:,0]
        #pylab.plot(pheno_actual,predicted_pheno.val,".")
        #pylab.show()


        self.compare_files(predicted_pheno,"snps")

    def test_kernel(self):
        logging.info("TestFastLmmModel test_kernel")

        train_idx = np.r_[10:self.snpreader_whole.iid_count] # iids 10 and on
        test_idx  = np.r_[0:10] # the first 10 iids

        # Show it using the snps
        K0_train = self.snpreader_whole[train_idx,:].read_kernel(Unit())
        covar_train3 = self.covariate_whole[train_idx,:].read()
        pheno_train3 = self.pheno_whole[train_idx,:].read()
        pheno_train3.val = self.snpreader_whole[train_idx,0:1].read().val*2
        assert np.array_equal(K0_train.iid,covar_train3.iid), "Expect iids to be the same (so that early and late Unit standardization will give the same result)"
        assert np.array_equal(K0_train.iid,pheno_train3.iid), "Expect iids to be the same (so that early and late Unit standardization will give the same result)"

        #pylab.plot(G0_train[:,0:1].read().val[:,0], pheno_train3.val[:,0],".")
        #pylab.show()

        #Learn model, save, load
        fastlmm_model3x = FastLmmModel.learn(K0_train=K0_train, covar_train=covar_train3, pheno_train=pheno_train3)
        filename = self.tempout_dir + "/model_snps.flm.npz"
        pstutil.create_directory_if_necessary(filename)
        fastlmm_model3x.save(filename)
        fastlmm_model3 = FastLmmModel.load(filename)

        #!!!cmk this is test on train. We also need test on test to show what ktest* construction
        #Predict with model (test on train)
        predicted_pheno = fastlmm_model3.predict(K0_test=K0_train, covar_test=covar_train3) #test on train
        output_file = self.file_name("kernel")
        Dat.write(output_file,predicted_pheno)

        #### Plot training x and y, and training x with predicted y
        #pylab.plot(self.snpreader_whole[train_idx,0:1].read().val[:,0], pheno_train3.val,".",self.snpreader_whole[train_idx,0:1].read().val[:,0],predicted_pheno.val,".")
        #pylab.show()

        #### Plot y and predicted y (test on train)
        #pheno_actual = pheno_train3.val[:,0]
        #pylab.plot(pheno_actual,predicted_pheno.val,".")
        #pylab.show()


        self.compare_files(predicted_pheno,"snps") #"kernel" and "snps" test cases should give the same results

    def test_kernel_one(self):
        logging.info("TestFastLmmModel test_kernel_one")

        train_idx = np.r_[10:self.snpreader_whole.iid_count] # iids 10 and on
        test_idx  = np.r_[0:10] # the first 10 iids

        K0_train = self.snpreader_whole[train_idx,:].read_kernel(Unit())
        covar_train = self.covariate_whole[train_idx,:]
        pheno_train = self.pheno_whole[train_idx,:]
        assert np.array_equal(K0_train.iid,covar_train.iid), "Expect iids to be the same (so that early and late Unit standardization will give the same result)"
        assert np.array_equal(K0_train.iid,pheno_train.iid), "Expect iids to be the same (so that early and late Unit standardization will give the same result)"

        fastlmm_model1 = FastLmmModel.learn(K0_train, covar_train, pheno_train)
        filename = self.tempout_dir + "/model_kernel_one.flm.npz"
        pstutil.create_directory_if_necessary(filename)
        fastlmm_model1.save(filename)
        fastlmm_model2 = FastLmmModel.load(filename)
                
        # predict on test set
        G0_test = self.snpreader_whole[test_idx,:]
        covar_test = self.covariate_whole[test_idx,:]

        K0_test = self.snpreader_whole[train_idx,:].read_kernel(Unit(),test=self.snpreader_whole[test_idx,:])
        predicted_pheno = fastlmm_model2.predict(K0_test, covar_test)

        output_file = self.file_name("kernel_one")
        Dat.write(output_file,predicted_pheno)

        pheno_actual = self.pheno_whole[test_idx,:].read().val[:,0]

        #pylab.plot(pheno_actual, predicted_pheno.val,".")
        #pylab.show()


        self.compare_files(predicted_pheno,"one") #Expect same results as SNPs "one"

    def compare_files(self,answer,ref_base):
        reffile = TestFeatureSelection.reference_file("fastlmmmodel/"+ref_base+".dat")
        reference=Dat(reffile).read()
        assert np.array_equal(answer.sid,reference.sid), "sid differs. File '{0}'".format(reffile)
        assert np.array_equal(answer.iid,reference.iid), "iid differs. File '{0}'".format(reffile)
        for iid_index in xrange(reference.iid_count):
            for sid_index in xrange(reference.sid_count):
                a_v = answer.val[iid_index,sid_index]
                r_v = reference.val[iid_index,sid_index]
                assert abs(a_v - r_v) < 1e-5, "Value at {0},{1} differs too much from file '{2}'".format(iid_index,sid_index,reffile)

    def test_doctest(self):
        old_dir = os.getcwd()
        os.chdir(os.path.dirname(os.path.realpath(__file__))+"/..")
        result = doctest.testfile("../fastlmmmodel.py")
        os.chdir(old_dir)
        assert result.failed == 0, "failed doc test: " + __file__


        


def getTestSuite():
    
    #!!!cmk add these to fastlmm main testing code
    suite1 = unittest.TestLoader().loadTestsFromTestCase(TestFastLmmModel)
    return unittest.TestSuite([suite1])



if __name__ == '__main__':
    # this import is needed for the runner
    from fastlmm.association.tests.test_fastlmmmodel import TestFastLmmModel
    suites = unittest.TestSuite([getTestSuite()])

    if False: #Standard test run #!!!cmk
        r = unittest.TextTestRunner(failfast=True) #!!!cmk fail fast=False
        r.run(suites)
    else: #Cluster test run
        logging.basicConfig(level=logging.INFO)

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
