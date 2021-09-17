import json
import os
import pickle
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

    dids = os.getenv('DIDS', None)

    if not dids:
        print("No DIDs found in environment. Aborting.")
        return

    print(f"dids: {dids}")
    print(os.listdir('data/ddos'))
    print(os.listdir('/data/ddos'))
    dids = json.loads(dids)

    for did in dids:
        filename = f'/data/ddos/{did}/0'
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

    filename = 'gpr.pickle' if local else "/data/outputs/result"
    with open(filename, 'wb') as pickle_file:
        print(f"Pickling results in {filename}")
        pickle.dump(Zhat, pickle_file)


if __name__ == "__main__":
    local = (len(sys.argv) == 2 and sys.argv[1] == "local")
    run_gpr(local)
