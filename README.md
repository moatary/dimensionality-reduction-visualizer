# dimensionality-reduction-visualizer
Collection of supervised and unsupervised dimensionality reduction techniques used for preprocessing, mass-fine-tuning, visualized educational comparisons purposes.

#Supervised and Unsupervised Dimensionality Reduction techniques.
To visualize and compare different unsupervised dimensionality reduction techniques, I used UCI cancer dataset and reduced it to dimension 2, and finally visualized it by colourizing each instance in matplot.

#Unupervised Methods are listed in the following lines:
-LLE, Locally Linear Embedding
-SVD, Singular Value Decomposition
-Remove dims with more zero value
-PCA, Principle Component Analysis
-T-SNE
-Cube-IndependentComponentAnalysis
-exponential-Independent-Component-Analysis
-Factor-Analysis
-Isomap
-Remove-Lower-Variance-Dims
-Umap
-Multi Dimensional Scaling

It must be noted that, before using SVD function, sparsify your data using scipy.sparse.lil_matrix or other scipy sparse functions.

#The supervised part of the codes, provide 4 different variations of Linear Discriminant Analysis (LDA), each of which works well in its approprate context. By checking out visualization results of classification of following  LDA variations, you see each next one works more accurately than the previous one : 
- The function named "lda_sklearn" (as Fisher LDA), does not function well in datas having nullspace in their within-class covariance matrix. But generally they require less computational resources versus other variations.
- Function with name "lda_usual", prevents effects of within-class covariance-inverse ill-posed-ness by removing NAN or high eigenvalues of that. 
- Function with name "lda_fukunaga",  seeks into subspace of lowest eigenvalues of within-class for higher-between-class-eigenvalues-subspaces. This approach even results in better discrimination versus "lda_usual"
- And finally, Function "lda_fullrank" even goes further and computes all within-class interactions of instances rather than merely interactions of average.
