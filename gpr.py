import json
import os
import sys

import arff
import matplotlib
matplotlib.use('agg')
from matplotlib import cm, pyplot
import numpy
from sklearn import gaussian_process

from branin_mesh import create_mesh


def get_input(local=False):
    if local:
        print("Reading local file branin.arff.")

        return 'branin.arff'

    dids = json.loads(os.getenv('DIDS', None))

    if not dids:
        print("No DIDs found in environment. Aborting.")
        return

    for did in dids:
        filename = '/data/ddos/' + did
        print(f"Reading asset file {filename}.")

        return filename


def plot(Zhat, npoints):
    X0, X1, Z = create_mesh(npoints)
    # plot data + model
    fig, ax = pyplot.subplots(subplot_kw={"projection": "3d"})
    ax.plot_wireframe(X0, X1, Z, linewidth=1)
    ax.scatter(X0, X1, Zhat, c="r", label="model")
    pyplot.title("Data + model")
    pyplot.show()


def run_gpr(local=False):
    npoints = 15

    filename = get_input(local)
    if not filename:
        print("Could not retrieve filename.")
        return

    with open(filename) as datafile:
        res = arff.load(datafile)

    print("Stacking data.")
    mat = numpy.stack(res["data"])
    [X, y] = numpy.split(mat, [2], axis=1)

    print("Applying Gaussian processing.")
    model = gaussian_process.GaussianProcessRegressor()
    model.fit(X, y)
    yhat = model.predict(X, return_std=False)
    Zhat = numpy.reshape(yhat, (npoints, npoints))

    if local:
        print("Plotting results")
        plot(Zhat, npoints)


if __name__ == "__main__":
    local = (len(sys.argv) == 2 and sys.argv[1] == "local")
    run_gpr(local)
