import sklearn
import numpy
import scipy
from dim_reductions import *


def initialize_DimRed_biomedicaldata(database,databasename='diabetes_', path='',dim=2,type=['pca'],inputdimensions=[2,3,4]):
    import pickle
    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    from matplotlib import animation
    from utils import plotbycolor# , animate
    from mpl_toolkits.mplot3d import Axes3D
    methodnames=['lle','svd','missingvalueratio','pca','tsne','fastica_cube','fastica_exp','fastica_logcosh','factoranal','isomap','lowvarfilter','umap','spectralembed']
    methodcaptions=[u"LLE, Locally Linear Embedding" , u"SVD, Singular Value Decomposition", u"Remove dims with more zero values", u"PCA, Principle Component Analysis",u"T-SNE", u"Cube-IndependentComponentAnalysis", u"exponential-Independent-Component-Analysis", u"  (}", u"Factor-Analysis", u"Isomap", u"Remove-Lower-Variance-Dims" , u"umap", u"full-proximity-spectral"]
    functionnames=[lle,svd,missingvalueratio,pca,tsne,fastica_cube,fastica_exp,fastica_logcosh,factoranal,isomap,lowvarfilter,umap,spectralembed]
    outputstate=[]
    transformed_data = [[] for _ in range(len(type))]
    if not path is '':
        w=open(path, 'rb')
        database = pickle.load(w)
        w.close()
    else:
        # How many dimensions should be used?
        database= database.data[:,inputdimensions]
        colors= diabetes.target/np.max(diabetes.target)
    # save source and transform in image file
    plt.clf()
    plotspecs = ['or']
    fig, ax = plt.subplots(nrows=2, ncols=2)
    datadims=[[0,1],[0,2],[2,1],[2,2]]
    numerator=-1
    for row in ax:
        for col in row:
            numerator+=1
            col = plotbycolor(database[:, datadims[numerator][0]], database[:, datadims[numerator][1]], plotspecs, colors,plot=col,title="dim %1d versus dim%2d"%tuple(datadims[numerator]))
            col.xaxis.set_visible(False)
    fig.savefig('results/' + databasename + '_source.png')
    fig = plt.figure()
    ax = Axes3D(fig)
    def init():
        ax.scatter(database[:, 0], database[:, 1], database[:, 2], c=matplotlib.cm.rainbow(colors), cmap=plt.cm.Spectral)# process dimensionality reductions
        return fig
    def animate(i):
        ax.view_init(elev=10., azim=i)
        return fig,
    # ax = fig.add_subplot(111, projection='3d')#
    # anim = animation.FuncAnimation(fig, animate, init_func=init, frames=360, interval=20, blit=True)
    # anim.save('results/' + databasename + '_source.mp4', fps=30, extra_args=['-vcodec', 'libx264'])
    for i,typ in enumerate(type):
        print('training%dth:%s'%(i,typ))
        ind=methodnames.index(typ)
        transformed_data[i]=functionnames[ind](database,dim)

        plt.clf()
        plotspecs=['ob']
        plt=plotbycolor(transformed_data[i][1][:,0],transformed_data[i][1][:,1],plotspecs,colors,title=methodcaptions[i])
        plt.gca().xaxis.set_visible(False)
        plt.savefig('results/'+databasename+typ+'_transf.png')

        print('finished %dth:%s\n' % (i, typ))
    outputstate.append(transformed_data)

    return outputstate



if __name__=='__main__':
    from sklearn import datasets, linear_model
    import pickle
    diabetes = datasets.load_diabetes()
    siz= diabetes.data.shape
    types=['lle','svd','missingvalueratio','pca','tsne','fastica_cube','fastica_logcosh','fastica_exp','factoranal','isomap','lowvarfilter','spectralembed']
    #types=['isomap']
    transforms=initialize_DimRed_biomedicaldata(diabetes,dim=2,type=types)
    with open('transforms_diabetes.pkl','wb') as ww:
        pickle.dump(transforms,ww)
