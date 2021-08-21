import matplotlib
matplotlib.use('TkAgg')
from matplotlib import cm, pyplot 
from mpl_toolkits.mplot3d import Axes3D
import numpy
from sklearn import gaussian_process

def branin_mesh(X0, X1):
    #b,c,t = 5.1/(4.*(pi)**2), 5./pi, 1./(8.*pi)
    b,c,t = 0.12918450914398066, 1.5915494309189535, 0.039788735772973836
    u = X1 - b*X0**2 + c*X0 - 6
    r = 10.*(1. - t) * numpy.cos(X0) + 10
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

def model_sin_1d():
    # not branin, but useful for testing plotting and gpr
    
    # create data
    npoints = 20
    xvec = numpy.linspace(0.0, 10.0, npoints)
    X = numpy.reshape(xvec, (npoints,1))
    y = numpy.sin(xvec)

    # build model
    kernel = gaussian_process.kernels.RBF()
    model = gaussian_process.GaussianProcessRegressor(
        kernel=kernel, normalize_y=True, random_state=0)
    model.fit(X, y)
    print(f"score={model.score(X,y)}")
    yhat = model.predict(X, return_std=False)
    
    # plot
    fig, ax = pyplot.subplots()
    line1, = ax.plot(xvec, y, label='y')
    line2, = ax.plot(xvec, yhat, label='yhat', linestyle="dashed", marker="o")
    ax.legend()
    pyplot.show()

def model_branin_2d():
    # Make data
    npoints = 15
    X0_vec = numpy.linspace(-5., 10., npoints)
    X1_vec = numpy.linspace(0., 15., npoints)
    X0, X1 = numpy.meshgrid(X0_vec, X1_vec)
    Z = branin_mesh(X0, X1)

    # Shape data for modeling
    X = numpy.empty((X0.shape[0]*X0.shape[1],2))
    X[:,0] = numpy.ravel(X0)
    X[:,1] = numpy.ravel(X1)
    y = numpy.ravel(Z)

    # Surface plot
    fig, ax = pyplot.subplots(subplot_kw={"projection": "3d"})
    ax.plot_surface(X0, X1, Z, cmap=cm.coolwarm,
                    linewidth=0, antialiased=False)
    pyplot.title("Data")
    pyplot.show()

    # Output arff file
    s = arff_string("Branin Function", "branin", X, y)
    f = open("branin.arff", "w")
    f.write(s)
    f.close()

    # Build model

    kernel = gaussian_process.kernels.RBF()
    model = gaussian_process.GaussianProcessRegressor(
        kernel=kernel, normalize_y=True, random_state=0)
    model.fit(X, y)
    print(f"score={model.score(X,y)}")
    yhat = model.predict(X, return_std=False)

    # Plot model
    fig, ax = pyplot.subplots(subplot_kw={"projection": "3d"})
    ax.plot_surface(X0, X1, Z, cmap=cm.coolwarm,
                    linewidth=0, antialiased=False)
    pyplot.title("Model")
    pyplot.show()


if __name__ == "__main__":
    model_branin_2d()
    #model_sin_1d()
