'''
Created on Aug 4, 2011

@author: anol
'''
import numpy as np
import mpl_toolkits.mplot3d.axes3d as p3
import pylab as p

def randrange(n, vmin, vmax):
    return (vmax-vmin)*np.random.rand(n) + vmin

fig = p.figure()
ax = p3.Axes3D(fig)
n = 100
for c, m, zl, zh in [('r', 'o', -50, -25), ('b', '^', -30, -5)]:
    xs = randrange(n, 23, 32)
    ys = randrange(n, 0, 100)
    zs = randrange(n, zl, zh)
    ax.scatter(xs, ys, zs, c=c, marker=m)

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

p.show()


#if __name__ == '__main__':
#    pass
