import matplotlib
matplotlib.use('agg')
from matplotlib import cm, pyplot
import numpy
from sklearn import gaussian_process
import io
import arff


def branin_mesh(X0, X1):
    # b,c,t = 5.1/(4.*(pi)**2), 5./pi, 1./(8.*pi)
    b, c, t = 0.12918450914398066, 1.5915494309189535, 0.039788735772973836
    u = X1 - b*X0**2 + c*X0 - 6
    r = 10.*(1. - t) * numpy.cos(X0) + 10
    Z = u**2 + r
    return Z


def run_gpr():
    npoints = 15
    X0_vec = numpy.linspace(-5., 10., npoints)
    X1_vec = numpy.linspace(0., 15., npoints)
    X0, X1 = numpy.meshgrid(X0_vec, X1_vec)
    Z = branin_mesh(X0, X1)

    with open('branin.arff', 'r') as datafile:
        res = arff.load(datafile)

    mat = numpy.stack(res["data"])
    [X, y] = numpy.split(mat, [2], axis=1)

    model = gaussian_process.GaussianProcessRegressor()
    model.fit(X, y)
    yhat = model.predict(X, return_std=False)
    Zhat = numpy.reshape(yhat, (npoints, npoints))

    # plot data + model
    fig, ax = pyplot.subplots(subplot_kw={"projection": "3d"})
    ax.plot_wireframe(X0, X1, Z, linewidth=1)
    ax.scatter(X0, X1, Zhat, c="r", label="model")
    pyplot.title("Data + model")

    buf = io.BytesIO()
    pyplot.savefig(buf, format='png')
    buf.seek(0)
    print(buf.read())
    buf.close()


if __name__ == "__main__":
    run_gpr()
