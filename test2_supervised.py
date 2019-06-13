import sklearn
import numpy
import scipy
from dim_reductions import *
from dim_reductions_supervised import *


def initialize_supervisedlearn_biomedicaldata(database,databasename='diabetes_', path='',dim=2,type=['pca'],inputdimensions=[2,3,4]):
    import pickle
    import matplotlib
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    from matplotlib import animation
    from utils import plotbycolor# , animate
    from mpl_toolkits.mplot3d import Axes3D


    methodnames=['lda_sklearn','lda_usual','lda_fullrank','lda_fukunaga','csp']
    functionnames=[lda_sklearn,lda_usual,lda_fullrank,lda_fukunaga,csp]
    outputstate=[]
    transformed_data = [[] for _ in range(len(type))]
    if not path is '':
        w=open(path, 'rb')
        class1,class2,targets,colors = pickle.load(w)
        w.close()
    else:
        # How many of dimensions should be used?
        targets = (database.target/np.max(database.target))>0.5
        database= database.data[:,inputdimensions]
        # prune data to suit vc limits of linear classifier
        # database,targets=database[:100,:], targets[:100]
        class1= database[targets==False,:]
        class2= database[targets==True,:]
        colors= targets
        # prune data to suit vc limits of linear classifier
        class1, class2 = class1[:40, :], class2[:40, :]
        colors=np.concatenate([np.ones((len(class1))),2*np.ones((len(class2)))],axis=0)
        colors=colors/np.max(colors)
        pass
    # save source and transform in image file
    plt.clf()
    plotspecs = ['or']
    fig, ax = plt.subplots(nrows=2, ncols=2)
    datadims=[[3,4],[3,5],[6,5],[6,4]]
    numerator=-1
    for row in ax:
        for col in row:
            numerator+=1
            col = plotbycolor(database[:, datadims[numerator][0]], database[:, datadims[numerator][1]], colors, plotspecs_list=plotspecs,plot=col,title='Dim %1d versus Dim %2d'%tuple(datadims[numerator]))
            # col.xaxis.set_visible(False)
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
        transformed_data[i]=functionnames[ind](class1,class2,2)
        wholedata= np.concatenate([transformed_data[i][1],transformed_data[i][2]],axis=0)
        plt.clf()
        plotspecs_list=['ob']
        plt=plotbycolor(wholedata[:,0],wholedata[:,1],colors,plotspecs_list=plotspecs_list)
        # plot axises over dim_means to be more salient:
        plotspecs_dict=dict(marker = 'o')
        # plt=plotbycolor([np.mean(wholedata[:,0])]*2,[np.min(wholedata[:,1]), np.max(wholedata[:,1])],colors,plotspecs_dict=plotspecs_dict,plot=plt,dontclear=True)
        # plt=plotbycolor([np.min(wholedata[:,0]), np.max(wholedata[:,0])],[np.mean(wholedata[:,1])]*2,colors,plotspecs_dict=plotspecs_dict,plot=plt,dontclear=True)

        plt.plot([np.mean(wholedata[:,0])]*2,[np.min(wholedata[:,1]), np.max(wholedata[:,1])],**plotspecs_dict)
        plt.plot( [np.min(wholedata[:,0]), np.max(wholedata[:,0])],[np.mean(wholedata[:,1])]*2,**plotspecs_dict)
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
    types=['lda_usual','lda_fullrank','lda_fukunaga','csp']#'lda_sklearn'
    #types=['isomap']
    transforms=initialize_supervisedlearn_biomedicaldata(diabetes,dim=2,type=types,inputdimensions=slice(-1))
    with open('transforms_diabetes_supervised.pkl','wb') as ww:
        pickle.dump(transforms,ww)
