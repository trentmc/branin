from mpl_toolkits.mplot3d import Axes3D  

import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np

def branin_mesh(X0, X1):
    #b,c,t = 5.1/(4.*(pi)**2), 5./pi, 1./(8.*pi)
    b,c,t = 0.12918450914398066, 1.5915494309189535, 0.039788735772973836
    u = X1 - b*X0**2 + c*X0 - 6
    r = 10.*(1. - t) * np.cos(X0) + 10
    Z = u**2 + r
    return Z

def arff_string(title, name, X, y):
    """
    Inputs:
      title - string
      name - string (no spaces)
      X - float array [npoints, ndim] - inputs
      y - float array [npoints] - output

    Outputs:
      s - string
    """
    assert X.shape[0] == y.shape[0]
    npoints, ndim = X.shape
    
    s = ""
    
    s += "% 1. Title: " 
    s += "%s\n" % title
    
    s += "% 3. Number of instances: " 
    s += "%s\n" % npoints
    
    s += "% 6. Number of attributes: " 
    s += "%s\n" % ndim
    s += "\n"
    
    s += "@relation %s\n" % name
    s += "\n"
    
    for dim_i in range(ndim):
        s += "@attribute 'x%s' numeric\n" % dim_i
        
    s += "@attribute 'y' numeric\n"
    s += "\n"
        
    s += "@data\n"
    for point_i in range(npoints):
        for dim_i in range(ndim):
            s += "%0.4f," % X[point_i,dim_i]
        s += "%0.4f" % y[point_i]
        s += "\n"
        
    return s

# Make data
npoints = 15
X0_vec = np.linspace(-5., 10., npoints)
X1_vec = np.linspace(0., 15., npoints)
X0, X1 = np.meshgrid(X0_vec, X1_vec)
Z = branin_mesh(X0, X1)

# Plot the surface
fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(X0, X1, Z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
plt.show()

# Convert to 2d X and 1d y
X = np.empty((X0.shape[0]*X0.shape[1],2))
X[:,0] = np.ravel(X0)
X[:,1] = np.ravel(X1)
y = np.ravel(Z)

# Output arff file
s = arff_string("Branin Function", "branin", X, y)
f = open("branin.arff", "w")
f.write(s)
f.close()

#
