
def plotbycolor(x,y,c='auto',plotspecs_list=list(),plotspecs_dict=dict(),plot=None,title='',dontclear=False):
    import numpy as np
    if plot is not None:
        plt=plot
        out=plt
        titlefunc=plt.set_title
    else:
        import matplotlib.pyplot as plt
        titlefunc=plt.suptitle
        out=plt#.gcf()
    titlefunc = plt.set_title if 'set_title' in dir(plt) else plt.suptitle
    import matplotlib.cm as cm
    if isinstance(c,str):
        c=np.linspace(0, 1, len(x))
    try:
        colors = cm.rainbow(c.astype(float))
    except:
        pass
    if dontclear==False:
        plt.cla()
    titlefunc(title)
    for x0,y0,c0 in zip(x,y, colors):
        plt.plot(x0, y0, color=c0,*plotspecs_list,**plotspecs_dict)
    return out


def animate(i):
    ax.view_init(elev=10., azim=i)
    return fig,

