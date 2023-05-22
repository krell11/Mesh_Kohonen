import numpy as np
# 1) After installation, the package can be imported as follows
from grgen.kohonen import Kohonen
from grgen.auxiliary import Plotter

def buildTriangularCutoutGeometry():
    x = np.linspace(-1, 1, 200)
    y = np.linspace(-0.5, 0.5, 200)
    xx, yy = np.meshgrid(x, y)
    mask = yy > np.abs(xx)
    rectangle = np.column_stack((xx.flatten(), yy.flatten()))
    cutout = rectangle[mask.flatten()]

    outer = np.array([[-1, 0.5], [1, 0.5], [1, -0.5], [-1, -0.5], [-1, 0.5]])

    geometry = [outer, cutout]

    return geometry


def main():
    geometry = buildTriangularCutoutGeometry()
    som = Kohonen(0.03, geometry, vertexType="triangular")
    som.summary()
    som.plotter = Plotter("output", "triangular", 200, "gif", fps=30)
    som.train()
    som.timer.printTimerSummary()
    som.plotter.gif()
    som.plotter.removePng()


if __name__ == '__main__':
    main()
