import numpy as np
import sklearn


def lda_usual(class1,class2,finaldim):
    from sklearn.decomposition import PCA
    from scipy import sparse
    import numpy
    if isinstance(class1, numpy.ndarray) == False:
        input = class1.todense()
    if isinstance(class2, numpy.ndarray) == False:
        input = class2.todense()
    pca = PCA(n_components=class1.shape[1],copy=True, iterated_power='auto', random_state=None,svd_solver='auto', tol=0.0, whiten=False)
    zeromeanclass1=class1- np.mean(class1,0)
    zeromeanclass2=class2- np.mean(class2,0)
    withinclass_cov=np.asmatrix(zeromeanclass1.transpose())*np.asmatrix(zeromeanclass1)+ np.asmatrix(zeromeanclass2.transpose())*np.asmatrix(zeromeanclass2)
    meandifs= np.abs(np.mean(class2,0)-np.mean(class1,0))
    # Within-class subspace extraction
    betweenclass_cov= np.asmatrix(meandifs).T*(np.asmatrix(meandifs))
    pca.fit(withinclass_cov)
    eigs=pca.explained_variance_
    inveigsqrt= 1/(eigs**0.5)
    inveigsqrt[inveigsqrt>10000]=10000
    inveigsqrt[np.isnan(inveigsqrt)]=0
    # Between-class subspace extraction
    betweenclass_cov_transformed =  pca.components_.dot(np.diag(inveigsqrt)).dot( betweenclass_cov).dot((pca.components_.dot(np.diag(inveigsqrt))).T)
    pca1 = PCA(n_components=finaldim,copy=True, iterated_power='auto', random_state=None,svd_solver='auto', tol=0.0, whiten=False)
    pca1.fit(betweenclass_cov_transformed)
    class1_transformed= class1.dot(pca1.components_.T)
    class2_transformed= class2.dot(pca1.components_.T)
    return pca1.components_.transpose(), class1_transformed,class2_transformed


def lda_fullrank(class1,class2,finaldim):
    from sklearn.decomposition import PCA
    from scipy import sparse
    import numpy
    if isinstance(class1, numpy.ndarray) == False:
        input = class1.todense()
    if isinstance(class2, numpy.ndarray) == False:
        input = class2.todense()
    pca = PCA(n_components=class1.shape[1],copy=True, iterated_power='auto', random_state=None,svd_solver='auto', tol=0.0, whiten=False)
    zeromeanclass1=class1- np.mean(class1,0)
    zeromeanclass2=class2- np.mean(class2,0)
    withinclass_cov=np.asmatrix(zeromeanclass1.transpose())*np.asmatrix(zeromeanclass1)+ np.asmatrix(zeromeanclass2.transpose())*np.asmatrix(zeromeanclass2)
    autocorr=lambda M : M.T *M
    betweenclass_cov= autocorr(np.kron(np.asmatrix(class1),np.ones((class2.shape[0],1))) - np.kron(np.ones((class1.shape[0],1)), np.asmatrix(class2) ))
    # Within-class subspace extraction
    pca.fit(withinclass_cov)
    eigs=pca.explained_variance_
    inveigsqrt= 1/(eigs**0.5)
    inveigsqrt[inveigsqrt>100000]=100000
    inveigsqrt[np.isnan(inveigsqrt)]=0
    betweenclass_cov_transformed =  pca.components_.dot(np.diag(inveigsqrt)).dot( betweenclass_cov).dot((pca.components_.dot(np.diag(inveigsqrt))).T)
    # Between-class subspace extraction
    pca1 = PCA(n_components=finaldim,copy=True, iterated_power='auto', random_state=None,svd_solver='auto', tol=0.0, whiten=False)
    pca1.fit(betweenclass_cov_transformed)
    class1_transformed= class1.dot(pca1.components_.T)
    class2_transformed= class2.dot(pca1.components_.T)
    return pca1.components_.transpose(), class1_transformed,class2_transformed

def lda_fukunaga(class1,class2,finaldim):
    from sklearn.decomposition import PCA
    from scipy import sparse
    import numpy
    if isinstance(class1, numpy.ndarray) == False:
        input = class1.todense()
    if isinstance(class2, numpy.ndarray) == False:
        input = class2.todense()
    pca = PCA(n_components=class1.shape[1],copy=True, iterated_power='auto', random_state=None,svd_solver='auto', tol=0.0, whiten=False)
    zeromeanclass1=class1- np.mean(class1,0)
    zeromeanclass2=class2- np.mean(class2,0)
    withinclass_cov=np.asmatrix(zeromeanclass1.transpose())*np.asmatrix(zeromeanclass1)+ np.asmatrix(zeromeanclass2.transpose())*np.asmatrix(zeromeanclass2)
    autocorr=lambda M : M.T *M
    pca.fit(withinclass_cov)
    # find least eigenvalues
    inds= np.argsort(pca.explained_variance_[-2:])[:finaldim]
    # get pcacomponent to project betweenclass upon
    components= pca.components_[inds]
    components_first=components.copy()
    transformed_class1= np.asmatrix(class1)* np.asmatrix( components.T)
    transformed_class2= np.asmatrix(class2)* np.asmatrix( components.T)
    betweenclass_cov= autocorr(np.kron(np.asmatrix(transformed_class1),np.ones((class2.shape[0],1))) - np.kron(np.ones((class1.shape[0],1)), np.asmatrix(transformed_class2) ))
    pca1 = PCA(n_components=finaldim)
    res=pca1.fit(betweenclass_cov)
    class1_transformed= class1.dot(components_first.T).dot(pca1.components_.T)
    class2_transformed= class2.dot(components_first.T).dot(pca1.components_.T)
    return pca1.components_.transpose(), class1_transformed,class2_transformed




def csp(class1,class2,finaldim):
    from sklearn.decomposition import PCA
    from scipy import sparse
    import numpy
    if isinstance(class1, numpy.ndarray) == False:
        input = class1.todense()
    if isinstance(class2, numpy.ndarray) == False:
        input = class2.todense()
    pca = PCA(n_components=class1.shape[1])
    zeromeanclass1=class1- np.mean(class1,0)
    zeromeanclass2=class2- np.mean(class2,0)
    withinclass_cov=np.asmatrix(zeromeanclass1.transpose())*np.asmatrix(zeromeanclass1)+ np.asmatrix(zeromeanclass2.transpose())*np.asmatrix(zeromeanclass2)
    autocorr=lambda M : M.T *M
    betweenclass_cov= np.asmatrix(zeromeanclass1.transpose())*np.asmatrix(zeromeanclass1)
    pca.fit(withinclass_cov)
    # Denominator subspace extraction
    eigs=pca.explained_variance_
    inveigsqrt= 1/(eigs**0.5)
    inveigsqrt[inveigsqrt>100000]=100000
    inveigsqrt[np.isnan(inveigsqrt)]=0
    # Nominator subspace extraction
    betweenclass_cov_transformed =  pca.components_.dot(np.diag(inveigsqrt)).dot( betweenclass_cov).dot((pca.components_.dot(np.diag(inveigsqrt))).T)
    pca1 = PCA(n_components=finaldim,copy=True, iterated_power='auto', random_state=None,svd_solver='auto', tol=0.0, whiten=False)
    pca1.fit(betweenclass_cov_transformed)
    class1_transformed= class1.dot(pca1.components_.T)
    class2_transformed= class2.dot(pca1.components_.T)

    return pca1.components_.transpose(), class1_transformed,class2_transformed



def lda_sklearn(class1,class2,finaldim):
    import numpy as np
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    # Data
    X = np.concatenate([class1,class2],axis=0)
    y=np.concatenate([np.ones((len(class1),1)), 2*np.ones((len(class2),1))],axis=0)[:,0]
    # Model
    clf = LinearDiscriminantAnalysis(n_components=3)
    Xnew=clf.fit(X, y).transform(X)

    return [],Xnew[:(len(class1)),:].T,Xnew[(len(class1)):,:].T


