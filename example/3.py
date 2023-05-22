import numpy as np
from grgen.kohonen import Kohonen
from grgen.auxiliary import Plotter


def buildPolygonNACA(xx):
    x = np.linspace(-1, 1, 200)
    y = np.linspace(-0.5, 0.5, 200)
    xx, yy = np.meshgrid(x, y)

    # Define the shape of the outer boundary
    outer = np.array([[-1, 0.5], [1, 0.5], [1, -0.5], [-1, -0.5], [-1, 0.5]])

    # Define the radius and center of the circular cutout
    radius = 0.3
    center = [0, 0]

    # Create a mask for the circular cutout
    mask = (xx - center[0]) ** 2 + (yy - center[1]) ** 2 > radius ** 2

    # Apply the mask to the rectangular grid
    cutout = np.column_stack((xx.flatten(), yy.flatten()))[mask.flatten()]

    # Combine the outer boundary and cutout to form the custom geometry
    geometry = [outer.tolist(), cutout.tolist()]  # Convert arrays to lists
    return geometry
def main():
    geometry = buildPolygonNACA(12)

    som = Kohonen(0.03, geometry, vertexType="triangular")
    som.summary()

    som.plotter = Plotter("output", "custom_cutout", 200, "gif", fps=10)

    som.train()

    som.timer.printTimerSummary()
    som.plotter.gif()
    som.plotter.removePng()


if __name__ == '__main__':
    main()
