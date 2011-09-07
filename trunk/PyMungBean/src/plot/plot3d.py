'''
Created on Aug 4, 2011

@author: anol
'''
import numpy as np
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.pyplot as plt
from matplotlib.pyplot import setp
#from pylab  import *

def randrange(n, vmin, vmax):
    return (vmax - vmin) * np.random.rand(n) + vmin

def plotHu(data={}):
    fig = []
    color = ['r', 'b', 'g', 'c', 'y']
    mark = ['o', '^', 'x', 's']
    
        
    for i in range(1, 8, 1):
        for j in range(2, 8, 1):
            if i >= j:
                continue
            else:
#                print kind
                f = plt.figure()
                ax = p3.Axes3D(fig=f)
                n = 0
                legend = []
                for key, hu in data.items():
                #    n = 100
                    legend.append(key)
                    zs = range(len(hu[i - 1]))
#                    ax.scatter( \
#                                hu[i - 1], hu[j - 1], z, \
#                               c=color[n], \
##                               c=color[np.random.randint(0, high=4)], \
#                               marker=mark[n] \
#                               )
                    ax.plot( \
                                     hu[i - 1], hu[j - 1], zs, \
                                     label=key, \
                                     c=color[n], \
#                               c=color[np.random.randint(0, high=4)], \
                                    marker=mark[n] \
                               )
                    n += 1
#                ax.legend(tuple(legend))#set_title(key)
                
                ax.set_xlabel('hu%d' % (i))
                ax.set_ylabel('hu%d' % (j))
                ax.set_zlabel('number of seed')
                
                fig.append(f)
                
                
              
    plt.show()


#if __name__ == '__main__':
#    pass
