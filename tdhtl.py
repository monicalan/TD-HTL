# -*- coding: utf-8 -*-
"""
Created on Sun Jul 28 12:57:22 2024

@author: yanglan
"""

import math
import numpy as np
from scipy.spatial.distance import pdist
from sklearn.model_selection import train_test_split
from sklearn import linear_model 
from sklearn.metrics import accuracy_score
from numpy.linalg import norm

class DataLoad(object):
    
    def __init__(self, wl, sl):
    
        self.wl=wl
        self.sl=sl
        
    def data_load(self, sig_data, labels):
        
        sample_n=np.size(sig_data,0)
        node_num=np.size(sig_data,1)
        time_num=np.size(sig_data,2)
        sigdata=np.zeros((node_num,time_num,sample_n)) 
        T=math.ceil((time_num-self.wl)/self.sl)+1
        adjdata=np.zeros((math.comb(node_num, 2),T,sample_n))
        
        for n in range(sample_n):
            sigdata[:,:,n]=sig_data[n,:,:]
            
        for n in range(sample_n):
            for t in range(T):
               if t!=T:
                   adjdata[:,t,n]=1-pdist(sig_data[n,:,((t-0)*self.sl+0):((t-0)*self.sl+self.wl)],'correlation')
               else:
                   adjdata[:,t,n]=1-pdist(sig_data[n,:,((t-0)*self.sl+0):time_num],'correlation')
            
        return sigdata, adjdata, labels 
 

class TLDL(object):
    
    def __init__(self, n_components, max_iter=30, tol=1e-6, nonzero_level_s=None, nonzero_level_t=None, disp=True):
        """
        MODEL: 
            Sn=DVn   n=1,2,...,N
            Sn is a matrix with p rows and T columns, p is the features number, T is the time points.
            D is a dictionary matrix with p rows and n_components columns.
            Vn is a sparse matrix with n_components rows and T columns.        
        param n_components: The number of dictionary atoms.
        param max_iter: The maximum iterations. 
        param tol: The tolerance of error between the original data and the reconstruction data. 
        param nonzero_level_s: The ratio of non-zeros elements in each column of sparse matrix for source data.
        param nonzero_level_t: The ratio of non-zeros elements in each column of sparse matrix for target data.
        disp: Whether to show the iter information.
        """
        self.Source_dict = None
        self.Source_scoe = None
        self.Target_dict = None
        self.Target_scoe = None
        self.max_iter = max_iter
        self.tol = tol
        self.n_components = n_components
        self.nonzero_level_s = nonzero_level_s
        self.nonzero_level_t = nonzero_level_t
        self.disp=disp
    
    def _initialize(self, y_source):
        """
        Initialize dictionary
        """       
        (p,T,N) = np.shape(y_source)
        y_source = y_source.reshape(p,N*T,order='F')
        
        #   ksvd_initialize
        u, _, _ = np.linalg.svd(y_source)
        d = u[:, :self.n_components]
        
        #   random_initialize
        # d = np.random.rand(p, self.n_components)
        # d = d / np.linalg.norm(d, axis=0) 
        
        x = linear_model.orthogonal_mp(d, y_source, n_nonzero_coefs=int(self.nonzero_level_s*self.n_components))
        x = x.reshape(self.n_components,T,N,order='F')
        return d, x
        
    def _update_dict(self, y_source, d, x):
        """
        Update each dictionary atoms by K-SVD
        """
        (p,T,N)=np.shape(y_source)
        y_source=y_source.reshape(p,N*T,order='F')
        x=x.reshape(self.n_components,N*T,order='F')       
        for i in range(self.n_components):
            index = np.nonzero(x[i, :])[0]
            if len(index) == 0:
                continue
            d[:, i] = 0
            r = (y_source - np.dot(d, x))[:, index]
            u, s, v = np.linalg.svd(r, full_matrices=False)
            d[:, i] = u[:, 0].T
            x[i, index] = s[0] * v[0, :]
        return d
    
    def _update_scoe(self, y_source, d):
        """
        Update sparse matrix using OMP
        """
        (p,T,N) = np.shape(y_source)
        y_source = y_source.reshape(p,N*T,order='F')
        x = linear_model.orthogonal_mp(d, y_source, n_nonzero_coefs=int(self.nonzero_level_s*self.n_components))
        x = x.reshape(self.n_components,T,N,order='F')
        return x    
    
    def source_fit(self, y_source, label):
        """
        Training the model
        INPUT: 
            y_source, which is a three-way arrays with size (p,T,N).
            label, which is the true label of samples with size [N,1]
        OUTPUT: 
            D, which is a list, it contains C matrices with size (p,n_components).
            X, which is a list, it contains C three-way arrays with size (n_components,T,N).
        """
        classnum = np.size(np.unique(label),0)
        D = []
        X = []
        if self.disp==True:
            for c in range(classnum):              
                print(
                   "Class {:05d}".format(
                            c))
                y_class = y_source[:,:,label==c]
                d, x = self._initialize(y_class)
                for i in range(self.max_iter):
                    print(
                        "Epoch {:05d}".format(
                            i))
                    e = np.mean(norm(y_class.transpose(2,0,1) - d@x.transpose(2,0,1), 'fro', (1,2)))
                    if e < self.tol:
                       break
                    d = self._update_dict(y_class, d, x)
                    x = self._update_scoe(y_class, d)
                D.append(d)
                X.append(x)
        else:
            for c in range(classnum):              
                y_class = y_source[:,:,label==c]
                d, x = self._initialize(y_class)
                for i in range(self.max_iter):
                    e = np.mean(norm(y_class.transpose(2,0,1) - d@x.transpose(2,0,1), 'fro', (1,2)))
                    if e < self.tol:
                       break
                    d = self._update_dict(y_class, d, x)
                    x = self._update_scoe(y_class, d)
                D.append(d)
                X.append(x)               
        self.Source_dict = D
        self.Source_scoe = X
    
    def _transfer_Dict(self, d_s):
        """
        Transfer the learned dictionary from source data into dictionary for target data
        """
        p = np.size(d_s,0)
        d_t = np.zeros((int(p*(p-1)/2),0)) 
        for i in range(self.n_components): 
            for j in range(i,self.n_components): 
                if i==j:
                   tmp = np.triu(d_s[:,i,None]@d_s[:,j,None].T,k=1).ravel()  
                else:
                   tmp = np.triu(d_s[:,i,None]@d_s[:,j,None].T+d_s[:,j,None]@d_s[:,i,None].T,k=1).ravel()            
                d_t = np.hstack((d_t,tmp[tmp!=0][:,None]))
        return d_t
    
    def _cal_scoe(self, y_target, d_t):
        """
        Calculate the sparse matrix for target data
        """
        (p,T,N) = np.shape(y_target)
        y_target = y_target.reshape(p,N*T,order='F')
        x = linear_model.orthogonal_mp(d_t, y_target, n_nonzero_coefs=int(self.nonzero_level_t*np.size(d_t,1)))
        x = x.reshape(np.size(d_t,1),T,N,order='F')
        return x
    
    def target_fit(self, y_target, label, D_s):
        """
        Parameters
        ----------
        y_target : three-way arrays
            The input target data.
        label : N-dimensional vector
            The true label of samples
        D_s : list
            The dictionary learned from source data.

        Returns
        -------
        None.
        """
        classnum = np.size(np.unique(label),0)
        D = []
        X = []
        for c in range(classnum):
            y_class = y_target[:,:,label==c]
            d = self._transfer_Dict(D_s[c])
            x = self._cal_scoe(y_class, d)
            D.append(d)
            X.append(x)
        self.Target_dict = D
        self.Target_scoe = X        
        
    def predict(self, y, D, label, Domain):
        """
        Estimate the labels of samples
        INPUT:
            y: The samples with size [p,T,N].
            D: The dictionaries learned from each labels, which is a list.
            label: The true label.
        OUTPUT: 
            labelest: The estimation of the labels.
            acc: The accuracy of classification.
        """
        classnum = np.size(D,0)
        loss = np.zeros((classnum,np.size(y,2)))
        for i in range(classnum):
            if Domain=="S":         
               x = self._update_scoe(y, D[i])
            else:
               x = self._cal_scoe(y, D[i])   
            for n in range(np.size(y,2)): 
                loss[i,n] = norm(y[:,:,n]-D[i].dot(x[:,:,n]),'fro')
        labelest = np.argmin(loss,0)
        acc = accuracy_score(label,labelest) 
        return labelest, acc
    
    def eva_snr(self, y, D, X, label):
        """
        SNR: to evaluate the reconstruction performance of tensor dictionary learning
        """
        classnum = np.size(D,0)
        snr = np.zeros((1,classnum))
        for i in range(classnum):
            error=y[:,:,label==i]-(D[i]@X[i].transpose(2,0,1)).transpose(1,2,0)
            snr[:,i]=10*np.log10(np.sum(norm(y[:,:,label==i],'fro',(0,1))**2)/np.sum(norm(error,'fro',(0,1))**2))
        return snr

class TDHTLExperiment:
    def __init__(self, num_repeats=10, test_size=0.2, dicnum=50, max_iter=30, 
                 tol=1e-6,
                 nonzero_ratio_s=0.8, nonzero_ratio_t=0.3, random_state=None):
        self.num_repeats = num_repeats
        self.test_size = test_size
        self.random_state = random_state
        self.dicnum=dicnum
        self.nonzero_ratio_s=nonzero_ratio_s
        self.nonzero_ratio_t=nonzero_ratio_t
        self.max_iter=max_iter
        self.tol=tol
        
        self.accuracies_s = []
        self.accuracies_t = []
        
    def tdhtl_experiment(self, trains, traint, label_train, tests, testt, label_test):
        
        mymodel = TLDL(n_components=self.dicnum, max_iter=self.max_iter, 
                       tol=self.tol, nonzero_level_s=self.nonzero_ratio_s, 
                       nonzero_level_t=self.nonzero_ratio_t
                       )    
        mymodel.source_fit(trains, label_train)
        mymodel.target_fit(traint, label_train, mymodel.Source_dict)
        
        # Prediction
        _, acc_s = mymodel.predict(tests, mymodel.Source_dict, label_test, Domain="S")
        _, acc_t = mymodel.predict(testt, mymodel.Target_dict, label_test, Domain="T")
        
        return acc_s, acc_t
        
    def run_experiments(self, datas, datat, labels):
        
        for i in range(self.num_repeats):
            train_idx, test_idx, label_train, label_test = train_test_split(
                np.arange(labels.shape[0]), labels, test_size=self.test_size, random_state=i)

            trains = datas[:, :, train_idx]
            tests = datas[:, :, test_idx]
            
            traint = datat[:, :, train_idx]
            testt = datat[:, :, test_idx]

            accuracys, accuracyt = self.tdhtl_experiment(trains, traint, 
                                        label_train, tests, testt, label_test)
            
            self.accuracies_s.append(accuracys)
            self.accuracies_t.append(accuracyt)
            
            
        self.calculate_statistics(self.accuracies_s)
        self.calculate_statistics(self.accuracies_t)

    def calculate_statistics(self, accuracies):
        self.mean_accuracy = np.mean(accuracies)
        self.std_accuracy = np.std(accuracies)
        
        print(f"Mean accuracy: {self.mean_accuracy:.4f}")
        print(f"Standard deviation of accuracy: {self.std_accuracy:.4f}")


# 使用示例
if __name__ == "__main__":
    
    sig_data = np.load('multivariate_time_series_data.npy', 
                       allow_pickle=True)
    labels = np.load('multivariate_time_series_labels.npy', 
                     allow_pickle=True)
    
    RD = DataLoad(20,10)  
    sigdata, adjdata, labels = RD.data_load(sig_data, labels)
        
    experiment = TDHTLExperiment(num_repeats=10, test_size=0.2, dicnum=10, max_iter=30, 
                  tol=1e-6,
                  nonzero_ratio_s=0.8, nonzero_ratio_t=0.9, random_state=None)
    experiment.run_experiments(sigdata, adjdata, labels)
    
    np.save('STDL_source_Results.npy', experiment.accuracies_s)
    
    np.save('TDHTL_target_Results.npy', experiment.accuracies_t)
    
    
