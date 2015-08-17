import logging

import matplotlib #!!!cmk comment out
matplotlib.use('TkAgg')  #!!!cmk comment out
import pylab

import numpy as np
import unittest
import os.path
import doctest
import pandas as pd
import pysnptools.util as pstutil

from fastlmm.association.fastlmmmodel import FastLmmModel #!!!cmk put in __init__
from fastlmm.association.fastlmmmodel import LinearRegressionModel #!!!cmk put in __init__
from fastlmm.util.runner import Local, HPC, LocalMultiProc
from pysnptools.snpreader import Dat, Bed, Pheno, SnpData
from fastlmm.feature_selection.test import TestFeatureSelection
from pysnptools.standardizer import Unit
from pysnptools.kernelreader import Identity as KernelIdentity
from pysnptools.kernelreader import KernelData

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

    #!!!cmk add a test case that passes K0=None, has a cov and that predicts on new data
    #        the code be able to both 1. Create a Kernel Identity that works with rectangeles and 2. create a SnpKernel with zero snp count

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

        predicted_pheno, covar = fastlmm_model2.predict(G0_test, G0_test, covar_test)

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
        predicted_pheno, covar = fastlmm_model3.predict(K0_test=G0_train, K0_test_test=G0_train, covar_test=covar_train3) #test on train
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

    def test_lmm(self):
        do_plot = False
        iid_count = 500
        seed = 0


        import pylab
        logging.info("TestFastLmmModel test_lmm")

        iid = [["cid{0}P{1}".format(iid_index,iid_index//250)]*2 for iid_index in xrange(iid_count)]
        train_idx = np.r_[10:iid_count] # iids 10 and on
        test_idx  = np.r_[0:10] # the first 10 iids


        #Every person is 100% related to everyone in one of 5 families
        K0a = KernelData(iid=iid,val=np.empty([iid_count,iid_count]),parent_string="related by distance")
        for iid_index0 in xrange(iid_count):
            for iid_index1 in xrange(iid_count):
                K0a.val[iid_index0,iid_index1] = 1 if iid_index0 % 5 == iid_index1 % 5 else 0
                if iid_index1 < iid_index0:
                    assert K0a.val[iid_index0,iid_index1] == K0a.val[iid_index1,iid_index0]

        #every person lives on a line from 0 to 1
        # They are related to every other person as a function of distance on the line
        np.random.seed(seed)
        home = np.random.random([iid_count])
        K0b = KernelData(iid=iid,val=np.empty([iid_count,iid_count]),parent_string="related by distance")
        for iid_index in xrange(iid_count):
            K0b.val[iid_index,:] = 1 - np.abs(home-home[iid_index])**.1

        #make covar just numbers 0,1,...
        covar = SnpData(iid=iid,sid=["x"],val=np.array([[float(num)] for num in xrange(iid_count)]))
        covar_train = covar[train_idx,:].read()
        covar_test = covar[test_idx,:].read()

        for name, h2, K0 in [("clones", 1, K0a),("line_world",.75,K0b)]:

            sigma2x = 100
            varg = sigma2x * h2
            vare = sigma2x * (1-h2)

            #make pheno  # pheno = 2*covar+100+normal(0,1)*2.5+normal(0,K)*7.5
            np.random.seed(seed)
            p1 = covar.val * 2.0 + 100
            p2 = np.random.normal(size=covar.val.shape)*np.sqrt(vare)
            p3 = (np.random.multivariate_normal(np.zeros(iid_count),K0.val)*np.sqrt(varg)).reshape(-1,1)
            pheno = SnpData(iid=iid,sid=["pheno0"],val= p1 + p2 + p3)

            pheno_train = pheno[train_idx,:].read()
            pheno_test = pheno[test_idx,:].read()

            if do_plot:
                #Plot training x and y, testing x and y
                pylab.plot(covar_train.val, pheno_train.val,".",covar_test.val, pheno_test.val,".")
                pylab.suptitle(name + ": Plot training x and y, testing x and y")
                pylab.show()

            Xtrain = np.c_[covar_train.val,np.ones((covar_train.iid_count,1))]
            Xtest = np.c_[covar_test.val,np.ones((covar_test.iid_count,1))]
            lsqSol = np.linalg.lstsq(Xtrain, pheno_train.val[:,0])
            bs=lsqSol[0] #weights
            r2=lsqSol[1] #squared residuals
            D=lsqSol[2]  #rank of design matrix
            N=pheno_train.iid_count
            REML = False
            if not REML:
                sigma2 = float(r2/N)
                nLL =  N*0.5*np.log(2*np.pi*sigma2) + N*0.5
            else:
                sigma2 = float(r2 / (N-D))
                nLL = N*0.5*np.log(2*np.pi*sigma2) + 0.5/sigma2*r2;
                nLL -= 0.5*D*np.log(2*np.pi*sigma2);#REML term

            predicted = Xtest.dot(bs)
            yerr = [np.sqrt(sigma2)] * len(predicted)
            if do_plot:
                pylab.plot(covar_test.val, pheno_test.val,"g.",covar_test.val, predicted,"r.")
                pylab.xlim([-1, 10])
                pylab.errorbar(covar_test.val, predicted,yerr,linestyle='None')
                pylab.suptitle(name + ": real linear regression: actual to prediction")
                pylab.show()

            for factor in [1,100,.02]:
                K0 = K0.read()
                K0.val *= factor

                K0_train = K0[train_idx]
                K0_test = K0[train_idx,test_idx]
                K0_test_test = K0[test_idx]

                #Learn model, save, load
                fastlmm_modelx = FastLmmModel.learn(K0_train=K0_train, covar_train=covar_train, pheno_train=pheno_train)
                v2 = np.var(p2)
                v3 = np.var(p3)
                logging.debug("Original h2 of {0}. Generated h2 of {1}. Learned h2 of {2}".format(h2, v3/(v2+v3), fastlmm_modelx.h2))
                
                
                filename = self.tempout_dir + "/model_lmm.flm.npz"
                pstutil.create_directory_if_necessary(filename)
                fastlmm_modelx.save(filename)
                fastlmm_model = FastLmmModel.load(filename)

                do_test_on_train = True
                if do_test_on_train:
                    #Predict with model (test on train)
                    predicted_pheno, covar_pheno = fastlmm_model.predict(K0_test=K0_train, K0_test_test=K0_train, covar_test=covar_train) #test on train
                    output_file = self.file_name("lmma_"+name)
                    Dat.write(output_file,predicted_pheno)
                    covar2 = SnpData(iid=covar_pheno.row,sid=covar_pheno.col[:,1],val=covar_pheno.val) #kludge to write kernel to text format
                    output_file = self.file_name("lmma.cov_"+name)
                    Dat.write(output_file,covar2)

                    yerr = np.sqrt(np.diag(covar_pheno.val))
                    predicted = predicted_pheno.val
                    if do_plot:
                        pylab.plot(covar_train.val, pheno_train.val,"g.",covar_train.val, predicted,"r.")
                        pylab.xlim([0, 50])
                        pylab.ylim([100, 200])
                        pylab.errorbar(covar_train.val, predicted,yerr,linestyle='None')
                        pylab.suptitle(name+": test on train: train X to true target (green) and prediction (red)")
                        pylab.show()

                    self.compare_files(predicted_pheno,"lmma_"+name)
                    self.compare_files(covar2,"lmma.cov_"+name)

                    predicted_pheno0, covar_pheno0 = fastlmm_model.predict(K0_test=K0_train[:,0], K0_test_test=K0_train[0], covar_test=covar_train[0,:]) #test on train #0
                    assert np.abs(predicted_pheno0.val[0,0] - predicted_pheno.val[0,0]) < 1e-6, "Expect a single case to get the same prediction as a set of cases"
                    assert np.abs(covar_pheno0.val[0,0] - covar_pheno.val[0,0]) < 1e-6, "Expect a single case to get the same prediction as a set of cases"


                #Predict with model (test on test)
                predicted_pheno, covar_pheno  = fastlmm_model.predict(K0_test=K0_test, K0_test_test=K0_test_test, covar_test=covar_test) #test on test
                output_file = self.file_name("lmmb_"+name)
                Dat.write(output_file,predicted_pheno)
                covar2 = SnpData(iid=covar_pheno.row,sid=covar_pheno.col[:,1],val=covar_pheno.val) #kludge to write kernel to text format
                output_file = self.file_name("lmmb.cov_"+name)
                Dat.write(output_file,covar2)

                yerr = np.sqrt(np.diag(covar_pheno.val))
                predicted = predicted_pheno.val
                if do_plot:
                    pylab.plot(covar_test.val, pheno_test.val,"g.",covar_test.val, predicted,"r.")
                    pylab.xlim([-1, 10])
                    pylab.errorbar(covar_test.val, predicted,yerr,linestyle='None')
                    pylab.suptitle(name+": test on test: test X to true target (green) and prediction (red)")
                    pylab.show()

                self.compare_files(predicted_pheno,"lmmb_"+name)
                self.compare_files(covar2,"lmmb.cov_"+name)

                predicted_pheno0, covar_pheno0  = fastlmm_model.predict(K0_test=K0_test[:,0], K0_test_test=K0_test_test[0], covar_test=covar_test[0,:]) #test on test
                assert np.abs(predicted_pheno0.val[0,0] - predicted_pheno.val[0,0]) < 1e-6, "Expect a single case to get the same prediction as a set of cases"
                assert np.abs(covar_pheno0.val[0,0] - covar_pheno.val[0,0]) < 1e-6, "Expect a single case to get the same prediction as a set of cases"

    def test_lr_as_lmm(self):
            do_plot = False

            import pylab
            logging.info("TestFastLmmModel test_lr_as_lmm")

            train_idx = np.r_[10:self.snpreader_whole.iid_count] # iids 10 and on
            test_idx  = np.r_[0:10] # the first 10 iids

            #make covar just numbers 0,1,...
            covar = self.covariate_whole.read()
            covar.val = np.array([[float(num)] for num in xrange(covar.iid_count)])
            covar_train = covar[train_idx,:].read()
            covar_test = covar[test_idx,:].read()

            name = ""
            temp = covar_train.read()
            unit_trained = Unit()._train_standardizer(temp,apply_in_place=True)
            from pysnptools.kernelreader import SnpKernel
            K0_train = SnpKernel(temp, standardizer=Unit()).read()
            K0_test = SnpKernel(covar_train, test=covar_test, standardizer=Unit()).read() #!!!cmk how do we know that both covar_train and covar_test will be standardized?
            K0_test_test = SnpKernel(covar_test, standardizer=unit_trained).read()
            #!!!cmk test this on SnpKernel (without reading it to memory yet)
            #!!!cmk can we make this work with just passing the snp files???


            #make pheno  # pheno = 2*covar+100+normal(0,1)*10
            pheno = self.pheno_whole.read()
            np.random.seed(0)
            pheno.val = covar.val * 2.0 + 100 + np.random.normal(size=covar.val.shape)*10

            #!!!cmk standardize it
            #!!!don't pheno = pheno.standardize()
            pheno_train = pheno[train_idx,:].read()
            pheno_test = pheno[test_idx,:].read()

            if do_plot:
                #Plot training x and y, testing x and y
                pylab.plot(covar_train.val, pheno_train.val,".",covar_test.val, pheno_test.val,".")
                pylab.suptitle("Plot training x and y, testing x and y")
                pylab.show()

            Xtrain = np.c_[covar_train.val,np.ones((covar_train.iid_count,1))]
            Xtest = np.c_[covar_test.val,np.ones((covar_test.iid_count,1))]
            lsqSol = np.linalg.lstsq(Xtrain, pheno_train.val[:,0])
            bs=lsqSol[0] #weights
            r2=lsqSol[1] #squared residuals
            D=lsqSol[2]  #rank of design matrix
            N=pheno_train.iid_count
            REML = False
            if not REML:
                sigma2 = float(r2/N)
                nLL =  N*0.5*np.log(2*np.pi*sigma2) + N*0.5
            else:
                sigma2 = float(r2 / (N-D))
                nLL = N*0.5*np.log(2*np.pi*sigma2) + 0.5/sigma2*r2;
                nLL -= 0.5*D*np.log(2*np.pi*sigma2);#REML term

            predicted = Xtest.dot(bs)
            yerr = [np.sqrt(sigma2)] * len(predicted)
            if do_plot:
                pylab.plot(covar_test.val, pheno_test.val,"g.",covar_test.val, predicted,"r.")
                pylab.xlim([-1, 10])
                pylab.errorbar(covar_test.val, predicted,yerr,linestyle='None')
                pylab.suptitle("real linear regression: actual to prediction")
                pylab.show()

            #Learn model, save, load
            fastlmm_modelx = FastLmmModel.learn(K0_train=K0_train, covar_train=None, pheno_train=pheno_train)
                
                
            filename = self.tempout_dir + "/model_lr_as_lmm.flm.npz"
            pstutil.create_directory_if_necessary(filename)
            fastlmm_modelx.save(filename)
            fastlmm_model = FastLmmModel.load(filename)

            do_test_on_train = True
            if do_test_on_train:
                #Predict with model (test on train)
                predicted_pheno, covar = fastlmm_model.predict(K0_test=K0_train, K0_test_test=K0_train, covar_test=None) #test on train
                output_file = self.file_name("lr_as_lmma_"+name)
                Dat.write(output_file,predicted_pheno)
                covar2 = SnpData(iid=covar.row,sid=covar.col[:,1],val=covar.val) #kludge to write kernel to text format
                output_file = self.file_name("lr_as_lmma.cov_"+name)
                Dat.write(output_file,covar2)

                yerr = np.sqrt(np.diag(covar.val))
                predicted = predicted_pheno.val
                if do_plot:
                    pylab.plot(covar_train.val, pheno_train.val,"g.",covar_train.val, predicted,"r.")
                    pylab.xlim([0, 50])
                    pylab.ylim([100, 200])
                    pylab.errorbar(covar_train.val, predicted,yerr,linestyle='None')
                    pylab.suptitle(name+": test on train: train X to true target (green) and prediction (red)")
                    pylab.show()

                self.compare_files(predicted_pheno,"lr_as_lmma_"+name)
                self.compare_files(covar2,"lr_as_lmma.cov_"+name)

            #Predict with model (test on test)
            predicted_pheno, covar  = fastlmm_model.predict(K0_test=K0_test, K0_test_test=K0_test_test, covar_test=None) #test on train
            output_file = self.file_name("lr_as_lmmb_"+name)
            Dat.write(output_file,predicted_pheno)
            covar2 = SnpData(iid=covar.row,sid=covar.col[:,1],val=covar.val) #kludge to write kernel to text format
            output_file = self.file_name("lr_as_lmmb.cov_"+name)
            Dat.write(output_file,covar2)

            yerr = np.sqrt(np.diag(covar.val))
            predicted = predicted_pheno.val
            if do_plot:
                pylab.plot(covar_test.val, pheno_test.val,"g.",covar_test.val, predicted,"r.")
                pylab.xlim([-1, 10])
                pylab.errorbar(covar_test.val, predicted,yerr,linestyle='None')
                pylab.suptitle(name+": test on test: test X to true target (green) and prediction (red)")
                pylab.show()
                ## Plot y and predicted y (test on train)
                #pylab.plot(pheno_test.val,predicted_pheno.val,".")
                #pylab.suptitle(name+": test on test: true target to prediction")
                #pylab.show()

            self.compare_files(predicted_pheno,"lr_as_lmmb_"+name)
            self.compare_files(covar2,"lr_as_lmmb.cov_"+name)

    def test_lr2(self):
        do_plot = False

        import pylab
        logging.info("TestFastLmmModel test_lr2")

        train_idx = np.r_[10:self.snpreader_whole.iid_count] # iids 10 and on
        test_idx  = np.r_[0:10] # the first 10 iids

        #make covar just numbers 0,1,...
        covar = self.covariate_whole.read()
        covar.val = np.array([[float(num)] for num in xrange(covar.iid_count)])
        covar_train = covar[train_idx,:].read()
        covar_test = covar[test_idx,:].read()
        K0_test_test = KernelIdentity(covar_test.iid)

        #make pheno  # pheno = 2*covar+100+normal(0,1)*10
        pheno = self.pheno_whole.read()
        np.random.seed(0)
        pheno.val = covar.val * 2.0 + 100 + np.random.normal(size=covar.val.shape)*10

        #!!!cmk standardize it
        #!!!don't pheno = pheno.standardize()
        pheno_train = pheno[train_idx,:].read()
        pheno_test = pheno[test_idx,:].read()

        if do_plot:
            #Plot training x and y, testing x and y
            pylab.plot(covar_train.val, pheno_train.val,".",covar_test.val, pheno_test.val,".")
            pylab.suptitle("Plot training x and y, testing x and y")
            pylab.show()

        Xtrain = np.c_[covar_train.val,np.ones((covar_train.iid_count,1))]
        Xtest = np.c_[covar_test.val,np.ones((covar_test.iid_count,1))]
        lsqSol = np.linalg.lstsq(Xtrain, pheno_train.val[:,0])
        bs=lsqSol[0] #weights
        r2=lsqSol[1] #squared residuals
        D=lsqSol[2]  #rank of design matrix
        N=pheno_train.iid_count
        REML = False
        if not REML:
            sigma2 = float(r2/N)
            nLL =  N*0.5*np.log(2*np.pi*sigma2) + N*0.5
        else:
            sigma2 = float(r2 / (N-D))
            nLL = N*0.5*np.log(2*np.pi*sigma2) + 0.5/sigma2*r2;
            nLL -= 0.5*D*np.log(2*np.pi*sigma2);#REML term

        predicted = Xtest.dot(bs)
        yerr = [np.sqrt(sigma2)] * len(predicted)
        if do_plot:
            pylab.plot(covar_test.val, pheno_test.val,"g.",covar_test.val, predicted,"r.")
            pylab.xlim([-1, 10])
            pylab.errorbar(covar_test.val, predicted,yerr,linestyle='None')
            pylab.suptitle("real linear regression: actual to prediction")
            pylab.show()

        #These should all give the same result
        first_name = None
        for name,K0_train,K0_test in [("Identity Kernel",KernelIdentity(self.snpreader_whole.iid[train_idx]),KernelIdentity(self.snpreader_whole.iid[train_idx],test=self.snpreader_whole.iid[test_idx])),
                                    ("sid_count=0", self.snpreader_whole[train_idx,[]],self.snpreader_whole[test_idx,[]])]:

            first_name = first_name or name
            #Learn model, save, load
            fastlmm_modelx = FastLmmModel.learn(K0_train=K0_train, covar_train=covar_train, pheno_train=pheno_train)
                
                
            filename = self.tempout_dir + "/model_lr2.flm.npz"
            pstutil.create_directory_if_necessary(filename)
            fastlmm_modelx.save(filename)
            fastlmm_model = FastLmmModel.load(filename)

            do_test_on_train = True
            if do_test_on_train:
                #Predict with model (test on train)
                predicted_pheno, covar = fastlmm_model.predict(K0_test=K0_train, K0_test_test=K0_train, covar_test=covar_train) #test on train
                output_file = self.file_name("lr2a_"+name)
                Dat.write(output_file,predicted_pheno)
                covar2 = SnpData(iid=covar.row,sid=covar.col[:,1],val=covar.val) #kludge to write kernel to text format
                output_file = self.file_name("lr2a.cov_"+name)
                Dat.write(output_file,covar2)

                yerr = np.sqrt(np.diag(covar.val))
                predicted = predicted_pheno.val
                if do_plot:
                    pylab.plot(covar_train.val, pheno_train.val,"g.",covar_train.val, predicted,"r.")
                    pylab.xlim([0, 50])
                    pylab.ylim([100, 200])
                    pylab.errorbar(covar_train.val, predicted,yerr,linestyle='None')
                    pylab.suptitle(name+": test on train: train X to true target (green) and prediction (red)")
                    pylab.show()

                self.compare_files(predicted_pheno,"lr2a_"+first_name)
                self.compare_files(covar2,"lr2a.cov_"+first_name)

            #Predict with model (test on test)
            predicted_pheno, covar  = fastlmm_model.predict(K0_test=K0_test, K0_test_test=K0_test_test, covar_test=covar_test) #test on train
            output_file = self.file_name("lr2b_"+name)
            Dat.write(output_file,predicted_pheno)
            covar2 = SnpData(iid=covar.row,sid=covar.col[:,1],val=covar.val) #kludge to write kernel to text format
            output_file = self.file_name("lr2b.cov_"+name)
            Dat.write(output_file,covar2)

            yerr = np.sqrt(np.diag(covar.val))
            predicted = predicted_pheno.val
            if do_plot:
                pylab.plot(covar_test.val, pheno_test.val,"g.",covar_test.val, predicted,"r.")
                pylab.xlim([-1, 10])
                pylab.errorbar(covar_test.val, predicted,yerr,linestyle='None')
                pylab.suptitle(name+": test on test: test X to true target (green) and prediction (red)")
                pylab.show()
                ## Plot y and predicted y (test on train)
                #pylab.plot(pheno_test.val,predicted_pheno.val,".")
                #pylab.suptitle(name+": test on test: true target to prediction")
                #pylab.show()

            self.compare_files(predicted_pheno,"lr2b_"+first_name)
            self.compare_files(covar2,"lr2b.cov_"+first_name)

    def test_lr_real(self):
        do_plot = False

        import pylab
        logging.info("TestFastLmmModel test_lr_real")

        train_idx = np.r_[10:self.snpreader_whole.iid_count] # iids 10 and on
        test_idx  = np.r_[0:10] # the first 10 iids

        #make covar just numbers 0,1,...
        covar = self.covariate_whole.read()
        covar.val = np.array([[float(num)] for num in xrange(covar.iid_count)])
        covar_train = covar[train_idx,:].read()
        covar_test = covar[test_idx,:].read()
        K0_test_test = KernelIdentity(covar_test.iid)

        #make pheno  # pheno = 2*covar+100+normal(0,1)*10
        pheno = self.pheno_whole.read()
        np.random.seed(0)
        pheno.val = covar.val * 2.0 + 100 + np.random.normal(size=covar.val.shape)*10

        pheno_train = pheno[train_idx,:].read()
        pheno_test = pheno[test_idx,:].read()

        if do_plot:
            #Plot training x and y, testing x and y
            pylab.plot(covar_train.val, pheno_train.val,".",covar_test.val, pheno_test.val,".")
            pylab.suptitle("Plot training x and y, testing x and y")
            pylab.show()

        Xtrain = np.c_[covar_train.val,np.ones((covar_train.iid_count,1))]
        Xtest = np.c_[covar_test.val,np.ones((covar_test.iid_count,1))]
        lsqSol = np.linalg.lstsq(Xtrain, pheno_train.val[:,0])
        bs=lsqSol[0] #weights
        r2=lsqSol[1] #squared residuals
        D=lsqSol[2]  #rank of design matrix
        N=pheno_train.iid_count
        REML = False
        if not REML:
            sigma2 = float(r2/N)
            nLL =  N*0.5*np.log(2*np.pi*sigma2) + N*0.5
        else:
            sigma2 = float(r2 / (N-D))
            nLL = N*0.5*np.log(2*np.pi*sigma2) + 0.5/sigma2*r2;
            nLL -= 0.5*D*np.log(2*np.pi*sigma2);#REML term

        predicted = Xtest.dot(bs)
        yerr = [np.sqrt(sigma2)] * len(predicted)
        if do_plot:
            pylab.plot(covar_test.val, pheno_test.val,"g.",covar_test.val, predicted,"r.")
            pylab.xlim([-1, 10])
            pylab.errorbar(covar_test.val, predicted,yerr,linestyle='None')
            pylab.suptitle("real linear regression: actual to prediction")
            pylab.show()

        #These should all give the same result
        first_name = None
        for name,K0_train,K0_test in [("Identity Kernel",None,None)]:

            first_name = first_name or name
            #Learn model, save, load
            modelx = LinearRegressionModel.learn(K0_train=K0_train, covar_train=covar_train, pheno_train=pheno_train)
                
                
            filename = self.tempout_dir + "/model_lr_real.flm.npz"
            pstutil.create_directory_if_necessary(filename)
            modelx.save(filename)
            model = LinearRegressionModel.load(filename)

            do_test_on_train = True
            if do_test_on_train:
                #Predict with model (test on train)
                predicted_pheno, covar = model.predict(K0_test=K0_train, K0_test_test=K0_train, covar_test=covar_train) #test on train
                output_file = self.file_name("lr_reala_"+name)
                Dat.write(output_file,predicted_pheno)
                covar2 = SnpData(iid=covar.row,sid=covar.col[:,1],val=covar.val) #kludge to write kernel to text format
                output_file = self.file_name("lr_reala.cov_"+name)
                Dat.write(output_file,covar2)

                yerr = np.sqrt(np.diag(covar.val))
                predicted = predicted_pheno.val
                if do_plot:
                    pylab.plot(covar_train.val, pheno_train.val,"g.",covar_train.val, predicted,"r.")
                    pylab.xlim([0, 50])
                    pylab.ylim([100, 200])
                    pylab.errorbar(covar_train.val, predicted,yerr,linestyle='None')
                    pylab.suptitle(name+": test on train: train X to true target (green) and prediction (red)")
                    pylab.show()

                self.compare_files(predicted_pheno,"lr2a_"+first_name)
                self.compare_files(covar2,"lr2a.cov_"+first_name)

            #Predict with model (test on test)
            predicted_pheno, covar  = model.predict(K0_test=K0_test, K0_test_test=K0_test_test, covar_test=covar_test) #test on train
            output_file = self.file_name("lr_realb_"+name)
            Dat.write(output_file,predicted_pheno)
            covar2 = SnpData(iid=covar.row,sid=covar.col[:,1],val=covar.val) #kludge to write kernel to text format
            output_file = self.file_name("lr_realb.cov_"+name)
            Dat.write(output_file,covar2)

            yerr = np.sqrt(np.diag(covar.val))
            predicted = predicted_pheno.val
            if do_plot:
                pylab.plot(covar_test.val, pheno_test.val,"g.",covar_test.val, predicted,"r.")
                pylab.xlim([-1, 10])
                pylab.errorbar(covar_test.val, predicted,yerr,linestyle='None')
                pylab.suptitle(name+": test on test: test X to true target (green) and prediction (red)")
                pylab.show()
                ## Plot y and predicted y (test on train)
                #pylab.plot(pheno_test.val,predicted_pheno.val,".")
                #pylab.suptitle(name+": test on test: true target to prediction")
                #pylab.show()

            self.compare_files(predicted_pheno,"lr2b_"+first_name)
            self.compare_files(covar2,"lr2b.cov_"+first_name)

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
        predicted_pheno, covar = fastlmm_model3.predict(K0_test=G0_train, K0_test_test=G0_train, covar_test=covar_train3) #test on train
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
        predicted_pheno, covar = fastlmm_model3.predict(K0_test=K0_train, K0_test_test=K0_train, covar_test=covar_train3) #test on train
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
        K0_test_test = KernelIdentity(iid=K0_test.iid1)
        predicted_pheno, covar = fastlmm_model2.predict(K0_test, K0_test_test, covar_test)

        output_file = self.file_name("kernel_one")
        Dat.write(output_file,predicted_pheno)

        pheno_actual = self.pheno_whole[test_idx,:].read().val[:,0]

        #pylab.plot(pheno_actual, predicted_pheno.val,".")
        #pylab.show()


        self.compare_files(predicted_pheno,"one") #Expect same results as SNPs "one"

    def compare_files(self,answer,ref_base):
        reffile = TestFeatureSelection.reference_file("fastlmmmodel/"+ref_base+".dat")
        reference=Dat(reffile).read()
        assert np.array_equal(answer.col,reference.col), "sid differs. File '{0}'".format(reffile)
        assert np.array_equal(answer.row,reference.row), "iid differs. File '{0}'".format(reffile)
        for iid_index in xrange(reference.row_count):
            for sid_index in xrange(reference.col_count):
                a_v = answer.val[iid_index,sid_index]
                r_v = reference.val[iid_index,sid_index]
                try:
                    assert abs(a_v - r_v) < 1e-4, "Value at {0},{1} differs too much from file '{2}'".format(iid_index,sid_index,reffile)
                except:
                    raise Exception("!!!cmk")

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
    logging.basicConfig(level=logging.DEBUG)

    # this import is needed for the runner
    from fastlmm.association.tests.test_fastlmmmodel import TestFastLmmModel
    suites = unittest.TestSuite([getTestSuite()])

    if False: #Standard test run #!!!cmk
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
