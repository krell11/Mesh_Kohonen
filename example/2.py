import numpy as np
from grgen.kohonen import Kohonen
from grgen.auxiliary import Plotter


def buildGeometry():
    x = np.linspace(-1, 1, 200)
    y = np.linspace(-0.5, 0.5, 200)
    xx, yy = np.meshgrid(x, y)

    rectangle = np.column_stack((xx.flatten(), yy.flatten()))

    theta = np.linspace(0, 2 * np.pi, 100)
    r = 0.2 + 0.1 * np.cos(4 * theta)
    star_x = r * np.cos(theta)
    star_y = r * np.sin(theta)
    star = np.column_stack((star_x, star_y))

    geometry = [rectangle, star]

    return geometry


def main():
    geometry = buildGeometry()

    som = Kohonen(0.03, geometry, vertexType="triangular")
    som.summary()

    som.plotter = Plotter("output", "custom_geometry", 200, "gif", fps=1)

    som.train()

    som.timer.printTimerSummary()
    som.plotter.gif()
    som.plotter.removePng()


if __name__ == '__main__':
    main()
