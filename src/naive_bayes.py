import numpy as np
import pdb
import warnings

from src.utils import softmax
from src.sparse_practice import flip_bits_sparse_matrix
from src.sparse_practice import sparse_to_numpy

class NaiveBayes:
    """
    A Naive Bayes classifier for binary data.
    """

    def __init__(self, smoothing=1):
        """
        Args:
            smoothing: controls the smoothing behavior when calculating beta
        """
        self.smoothing = smoothing

    def predict(self, X):
        """
        Return the most probable label for each row x of X.
        You should not need to edit this function.
        """
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)

    def predict_proba(self, X): #softmax
        """
        Using self.alpha and self.beta, compute the probability p(y | X[i, :])
            for each row X[i, :] of X.  The returned array should be
            probabilities, not log probabilities. If you use log probabilities
            in any calculations, you can use src.utils.softmax to convert those
            into probabilities that sum to 1 for each row.

        Don't worry about divide-by-zero RuntimeWarnings.

        Args:
            X: a sparse matrix of shape `[n_documents, vocab_size]` on which to
               predict p(y | x)

        Returns 
            probs: an array of shape `[n_documents, n_labels]` where probs[i, j] contains
                the probability `p(y=j | X[i, :])`. Thus, for a given row of this array,
                np.sum(probs[i, :]) == 1.
        """
        n_docs, vocab_size = X.shape
        n_labels = 2

        assert hasattr(self, "alpha") and hasattr(self, "beta"), "Model not fit!"
        assert vocab_size == self.vocab_size, "Vocab size mismatch"

        ret_mat = np.zeros((X.shape[0],n_labels)) #what do I remove Nan from
        
        for i in range(X.shape[0]): #take these for loops away and just do toarray math
            for j in range(n_labels): #use log
                p = self.alpha[j]
                val = 1
                for k in range(self.beta.shape[0]): #double check this if its rows or cols
                    if X[i,k] == 0:
                        val *= (1-self.beta[k,j]**(1-X[i,k]))
                    else:
                       val *=  (self.beta[k,j]**X[i,k]) 

                ret_mat[i,j] = p*val
            ret_mat[i,:] = ret_mat[i,:]/sum(ret_mat[i,:])
            
        return ret_mat                





        

    def fit(self, X, y):
        """
        Compute self.alpha and self.beta using the training data.
        This function *should not* use unlabeled data. Wherever y is NaN, that
        label and the corresponding row of X should be ignored.

        self.alpha should be set to contain the marginal probability of each class label.

        self.beta is an array of shape [n_vocab, n_labels]. self.beta[j, k]
            is the probability of seeing the word j in a document with label k.
            Remember to use self.smoothing. If there are M documents with label
            k, and the `j`th word shows up in L of them, then `self.beta[j, k]`.

        Note: all tests will provide X to you as a *sparse array* which will
            make calculations with large datasets much more efficient.  We
            encourage you to use sparse arrays whenever possible, but it can be
            easier to debug with dense arrays (e.g., it is easier to print out
            the contents of an array by first converting it to a dense array).

        Args: X, a sparse matrix of word counts; Y, an array of labels
        Returns: None; sets self.alpha and self.beta
        """
        
        n_docs, vocab_size = X.shape
        n_labels = 2
        self.vocab_size = vocab_size

        uniq = np.unique(y,return_counts= True)

        lbd =1
        b = np.zeros((X.shape[1],n_labels))
        



        uniq_labels,labels_count = np.unique(y,return_counts= True)
        tt_count = np.where(y != np.nan)[0].shape[0]

         # ignore nan
        self.alpha = labels_count/tt_count
 

        for i in range(X.shape[1]):
            iind = X[:,i].toarray()
            for j in range(n_labels):
                fltten = X[:,i].toarray().flatten()
                sm = fltten[(np.where((fltten ==1) & (y == j)))].shape[0]
                b[i,j] = (self.smoothing+sm)/(2*self.smoothing+labels_count[j])
        self.beta = b
        return None

    def likelihood(self, X, y): #logsum
        r"""
        Using the self.alpha and self.beta that were already computed in
            `self.fit`, compute the LOG likelihood of the data.  You should use
            logs to avoid underflow.  This function should not use unlabeled
            data. Wherever y is NaN, that label and the corresponding row of X
            should be ignored.

        Don't worry about divide-by-zero RuntimeWarnings.

        Args: X, a sparse matrix of binary word counts; Y, an array of labels
        Returns: the log likelihood of the data
        """
        assert hasattr(self, "alpha") and hasattr(self, "beta"), "Model not fit!"

        n_docs, vocab_size = X.shape
        n_labels = 2
        #how to ignore y
        p = 0
        for i in range(X.shape[0]):
            if ~(np.isnan(y[i])):    #y[i] != np.nan:"""
                p1 = np.log(self.alpha[y[i].astype(int)])
                p += p1
            p2 = 0
            for j in range(X.shape[1]):
                if ~(np.isnan(y[i])):#if y[i] != np.nan:
                    p2 += (X[i,j]*np.log(self.beta[j,y[i].astype(int)]) + (1-X[i,j])*np.log(1-self.beta[j,y[i].astype(int)]))

            p += (p2)
            #sum accross the correct l
        
        return (p)

"""

        y_non_nans  = y[~np.isnan(y)]
        pdb.set_trace()
        X_non_nans= X[~np.isnan(y),:]


        p1 = X@(np.log(self.beta))
        


        p2 = (1-X.toarray())@(np.log(1-self.beta))
        p3  = p2 + p1
        pdb.set_trace()

        z = y_non_nans[y==0].sum()
        o = y_non_nans[y==1].sum()

        tt = 0 
        tt += z*self.alpha[0]
        tt += o*self.alpha[1]



        for i in range(X.shape[0]):
            if y[i] != np.nan:
                tt += ( p3[i,y[i].astype(int)])
        return tt"""
 


        #y_no+nans[y==0].sum()
        #y_no+nans[y==1].sum() *self.alp









