import tensorflow as tf
import numpy as np
import scipy.spatial
import random
from shapely.geometry import Point, Polygon
from shapely.ops import nearest_points
from grgen.auxiliary import Timer
from grgen.auxiliary import Plotter

class Kohonen:
    def __init__(self, spacing, geometry, dim=2, s=0.1, iterations=None, iterationsFactor=1, minRadius=None,
                 maxRadius=None, batchSize=None, vertexType="triangular"):

        self.spacing = spacing
        self.geometry = geometry
        self.dim = dim
        self.s = s
        self.iterations = iterations
        self.minRadius = minRadius
        self.maxRadius = maxRadius
        self.batchSize = batchSize
        self.vertexType = vertexType

        self.weights = None
        self.startWeights = None
        self.noPoints = None
        self.noInternalPoints = None
        self.noBoundaryPoints = None
        self.noCells = None
        self.boundingBox = None

        self.eps = 10e-12
        self.dataType = np.float32

        self.tmpWeight = None
        self.geometryProbability = None
        self.vertexProbability = None

        self.connection = None
        self.neighbors = None
        self.boundary = None
        self.boundaryIdx = None
        self.innerIdx = None
        self.boundaryId = None
        self.boundaryFace = None

        self.timer = Timer()
        self.plotter = None

        self.calculateBoundingBox()
        if maxRadius == None:
            delta = np.subtract(self.boundingBox[:, 1], self.boundingBox[:, 0])
            self.maxRadius = np.max(delta) + 10 * spacing
        if minRadius == None:
            self.minRadius = 2 * spacing

        self.buildWeights()

        self.removeGridCoordinates()

        self.buildGridTopology()

        if iterations == None:
            self.iterations = self.noPoints
        self.iterations = int(iterationsFactor * self.iterations)

        self.calculateBoundaryProbability()

    def maskCornerPoints(self):
        removeIndices = list()

        for c in self.geometry[0]:
            tmp = tf.cast(c, dtype=self.dataType)
            neighbor = self.findNN(tf.gather(self.weights, self.boundaryIdx), tmp)

            tf.compat.v1.scatter_update(self.weights, tf.Variable(self.boundaryIdx[neighbor], dtype=np.int64), tmp)
            removeIndices.append(neighbor)

        self.boundaryIdx = np.delete(self.boundaryIdx, removeIndices)

    def findNN(self, searchSet, coordinates):

        squaredDistance = tf.reduce_sum((searchSet - tf.expand_dims(coordinates, axis=0)) ** 2, axis=1)

        # return the best matching unit
        return tf.argmin(squaredDistance, axis=0)

    def calculateBoundingBox(self):
        self.timer.startTimer("calculateBoundingBox")

        boundingBox = np.zeros((self.dim, 2, len(self.geometry)))
        index = 0

        for g in self.geometry:
            boundingBox[0, 0, index] = np.min(g[:, 0])
            boundingBox[0, 1, index] = np.max(g[:, 0])
            boundingBox[1, 0, index] = np.min(g[:, 1])
            boundingBox[1, 1, index] = np.max(g[:, 1])
            index += 1

        a = np.min(boundingBox[:, 0, :], axis=1).reshape(-1, 1)
        b = np.max(boundingBox[:, 1, :], axis=1).reshape(-1, 1)

        self.boundingBox = np.concatenate((a, b), axis=1)

        self.timer.stopTimer("calculateBoundingBox")

    def buildWeights(self):

        self.timer.startTimer("buildWeights")

        minX = self.boundingBox[0, 0]
        minY = self.boundingBox[1, 0]
        maxX = self.boundingBox[0, 1]
        maxY = self.boundingBox[1, 1]

        if (self.vertexType == "triangular"):
            spacingY = np.sqrt(self.spacing ** 2 - (self.spacing / 2) ** 2)
        else:
            spacingY = self.spacing

        rangeX = np.arange(minX - 3 * self.spacing, maxX + 3 * self.spacing, self.spacing)
        rangeY = np.arange(minY - 3 * spacingY, maxY + 3 * spacingY, spacingY)

        x, y = np.meshgrid(rangeX, rangeY)

        if (self.vertexType == "triangular"):
            x[::2, :] += self.spacing / 2

        x = x.reshape(-1, 1)
        y = y.reshape(-1, 1)

        self.weights = tf.Variable(np.concatenate((x, y), axis=1), dtype=self.dataType)
        self.noPoints = tf.shape(self.weights)[0]

        self.timer.stopTimer("buildWeights")

    def removeGridCoordinates(self):
        self.timer.startTimer("removeGridCoordinates")
        removeCoord = np.ones((tf.shape(self.weights)[0]), dtype=bool)

        inner = Polygon(self.geometry[1])
        outer = Polygon(self.geometry[0])

        for i in range(0, np.shape(self.weights)[0]):

            point = Point(self.weights[i, 0], self.weights[i, 1])

            if (inner.contains(point)):
                removeCoord[i] = False
            else:
                if (outer.contains(point)):
                    removeCoord[i] = True
                else:
                    removeCoord[i] = False

        self.weights = tf.Variable(tf.boolean_mask(self.weights, removeCoord), dtype=self.dataType)
        self.startWeights = self.weights
        self.noPoints = tf.shape(self.weights)[0]

        self.timer.stopTimer("removeGridCoordinates")

    def buildGridTopology(self):
        self.timer.startTimer("buildGridTopology")

        triangulation = scipy.spatial.Delaunay(self.weights.numpy())
        self.connection = triangulation.simplices
        self.neighbors = triangulation.neighbors

        it = 0
        remove = list()

        for x in self.connection:

            vertex = tf.gather(self.weights, x, axis=0)
            minimum = tf.math.reduce_min(vertex, axis=0)
            maximum = tf.math.reduce_max(vertex, axis=0)

            if ((maximum[0] - minimum[0]) * (maximum[1] - minimum[1]) / 2 > self.spacing ** 2 / 2 + self.eps):
                remove.append(it)

            it += 1

        self.connection = np.delete(self.connection, remove, axis=0)

        self.neighbors[np.isin(self.neighbors, remove)] = -1
        self.neighbors = np.delete(self.neighbors, remove, axis=0)

        self.boundary = np.argwhere(self.neighbors < 0)

        tmpBndry = list()
        for b in self.boundary:

            if (b[1] == 0):
                tmpBndry.append(self.connection[b[0], 1])
                tmpBndry.append(self.connection[b[0], 2])
            if (b[1] == 1):
                tmpBndry.append(self.connection[b[0], 2])
                tmpBndry.append(self.connection[b[0], 0])
            if (b[1] == 2):
                tmpBndry.append(self.connection[b[0], 0])
                tmpBndry.append(self.connection[b[0], 1])

        self.boundaryIdx = np.unique(np.array(tmpBndry))
        self.innerIdx = np.arange(0, self.noPoints, 1, dtype=np.int32)
        self.innerIdx = np.delete(self.innerIdx, self.boundaryIdx)

        self.noCells = np.shape(self.connection)[0]

        self.noInternalPoints = np.shape(self.innerIdx)[0]
        self.noBoundaryPoints = np.shape(self.boundaryIdx)[0]

        self.timer.stopTimer("buildGridTopology")

    def produceRandomInput(self, tensorflow=True):
        self.timer.startTimer("produceRandomInput")

        minX = self.boundingBox[0, 0]
        minY = self.boundingBox[1, 0]
        maxX = self.boundingBox[0, 1]
        maxY = self.boundingBox[1, 1]

        inner = Polygon(self.geometry[1])
        outer = Polygon(self.geometry[0])

        while (True):

            randomCoordinate = np.array([random.uniform(minX, maxX), random.uniform(minY, maxY)])

            point = Point(randomCoordinate[0], randomCoordinate[1])

            if (inner.contains(point)):
                continue
            else:
                if (outer.contains(point)):
                    if (tensorflow):
                        return tf.Variable(randomCoordinate, dtype=self.dataType)
                    else:
                        return randomCoordinate
                else:
                    continue
        self.timer.stopTimer("produceRandomInput")

    def calculateBoundaryProbability(self):
        self.geometryProbability = list()
        self.vertexProbability = list()

        for idx in range(0, len(self.geometry)):
            self.vertexProbability.append(
                np.sqrt(np.sum((self.geometry[idx] - np.roll(self.geometry[idx], 1, axis=0)) ** 2, axis=1)))

            self.geometryProbability.append(np.sum(self.vertexProbability[idx], axis=0))

            self.vertexProbability[idx] = self.vertexProbability[idx] / np.sum(self.vertexProbability[idx])

        self.geometryProbability = self.geometryProbability / np.sum(self.geometryProbability)

    def produceRandomInputBoundary(self, tensorflow=True):

        self.timer.startTimer("produceRandomInputBoundary")

        idx = np.random.choice(len(self.geometry), size=1, p=self.geometryProbability)

        idx = int(idx)

        nbr = np.shape(self.geometry[idx])[0]

        v = np.random.choice(nbr, size=1, p=self.vertexProbability[idx])

        minX = self.geometry[idx][v, 0]
        minY = self.geometry[idx][v, 1]
        maxX = np.roll(self.geometry[idx], 1, axis=0)[v, 0]
        maxY = np.roll(self.geometry[idx], 1, axis=0)[v, 1]

        randomCoordinate = np.array([random.uniform(minX, maxX), random.uniform(minY, maxY)]).reshape(-1, )

        self.timer.stopTimer("produceRandomInputBoundary")

        if (tensorflow):
            return tf.Variable(randomCoordinate, dtype=self.dataType)
        else:
            return randomCoordinate

    def produceRandomBatch(self):

        batchData = np.zeros((self.batchSize, self.dim))

        for i in range(0, self.batchSize):
            batchData[i, :] = self.produceRandomInputBoundary(False)

        return tf.Variable(batchData, dtype=self.dataType)

    def moveBoundaryPoints(self):

        self.timer.startTimer("moveBoundaryPoints")

        inner = Polygon(self.geometry[1])
        outer = Polygon(self.geometry[0])

        movement = np.zeros((np.shape(self.boundaryIdx)[0], 2))

        weightsBoundary = tf.Variable(tf.gather(self.weights, self.boundaryIdx), dtype=self.dataType).numpy()

        for idx in range(0, np.shape(self.boundaryIdx)[0]):

            point = Point(weightsBoundary[idx, 0], weightsBoundary[idx, 1])

            pOuter, p = nearest_points(outer.boundary, point)
            pInner, p = nearest_points(inner.boundary, point)

            if (point.distance(pInner) > point.distance(pOuter)):

                movement[idx, 0] = pOuter.x
                movement[idx, 1] = pOuter.y
            else:
                movement[idx, 0] = pInner.x
                movement[idx, 1] = pInner.y

        print(np.shape(self.boundaryIdx))
        print(np.shape(self.boundaryIdx))

        tf.compat.v1.scatter_update(self.weights, self.boundaryIdx, movement)

        self.timer.stopTimer("moveBoundaryPoints")

    def trainingOperation(self, inputData, searchSet, searchSetStart, trainingSetStart, delta, radius, k=0,
                          boundaryTraining=False):

        bmuIndex = self.findNN(searchSet, inputData)

        if (k > 0 or boundaryTraining):
            inputData = searchSetStart[bmuIndex, :]

        squaredDistanceStart = tf.reduce_sum(
            (trainingSetStart - tf.expand_dims(searchSetStart[bmuIndex, :], axis=0)) ** 2, axis=1)

        lateralConnection = self.s ** ((tf.math.sqrt(squaredDistanceStart) + k * self.spacing) ** 2 / (radius ** 2))

        self.tmpWeights = self.tmpWeights + (
                tf.expand_dims(delta * lateralConnection * (1 + k * tf.math.sqrt(squaredDistanceStart)), axis=1)
                * (tf.expand_dims(inputData, axis=0) - self.tmpWeights))

    def train(self):

        self.timer.startTimer("train")

        print("adaption")

        # self.moveBoundaryPoints()
        # self.startWeights = self.weights

        # self.tmpWeights = self.weights#tf.Variable(tf.gather(self.weights, self.boundaryIdx), dtype=self.dataType)
        self.tmpWeights = tf.Variable(self.weights, dtype=self.dataType)
        searchSetStart = tf.gather(self.startWeights, self.boundaryIdx)
        searchSet = tf.gather(self.startWeights, self.boundaryIdx)
        trainingSetStart = self.startWeights  # tf.gather(self.weights, self.boundaryIdx)

        timeIt = 1

        for it in range(1, int(10 * self.noBoundaryPoints / self.noPoints * self.iterations)):

            X = tf.cast(1 - tf.exp(5 * (it - self.iterations) / self.iterations), dtype=self.dataType)
            delta = 0.225 * tf.cast((it) ** (-0.2) * X, dtype=self.dataType)
            radius = 2 * self.spacing

            self.trainingOperation(self.produceRandomInputBoundary(),
                                   searchSet,
                                   searchSetStart,
                                   trainingSetStart,
                                   delta,
                                   radius,
                                   boundaryTraining=False)

            # tf.compat.v1.scatter_update(self.weights, self.boundaryIdx, self.tmpWeights)
            if (not self.plotter == None):
                self.plotter.plot(timeIt, self.weights[:, 0], self.weights[:, 1], self.connection)
            timeIt += 1

        self.weights = tf.Variable(self.tmpWeights, dtype=self.dataType)

        self.maskCornerPoints()
        self.moveBoundaryPoints()

        self.tmpWeights = tf.Variable(tf.gather(self.weights, self.innerIdx), dtype=self.dataType)
        searchSetStartCase1 = tf.gather(self.startWeights, self.boundaryIdx)
        searchSetStartCase2 = self.startWeights
        trainingSetStart = tf.gather(self.startWeights, self.innerIdx)
        delta = 0.04 * 0.05
        k = 10
        radius = k * self.spacing
        alpha_prob = self.noInternalPoints / (self.noBoundaryPoints * k + self.noInternalPoints)

        print("smoothing")

        for it in range(1, int(self.noInternalPoints / self.noPoints * self.iterations)):

            alpha = np.random.uniform(0, 1, 1)
            if (alpha > alpha_prob):

                searchSetCase1 = tf.cast(tf.gather(self.weights, self.boundaryIdx), dtype=self.dataType)
                self.trainingOperation(self.produceRandomInputBoundary(),
                                       searchSetCase1,
                                       searchSetStartCase1,
                                       trainingSetStart,
                                       delta,
                                       radius,
                                       k)
                tf.compat.v1.scatter_update(self.weights, self.innerIdx, self.tmpWeights)

                if (not self.plotter == None):
                    self.plotter.plot(timeIt, self.weights[:, 0], self.weights[:, 1], self.connection)
                timeIt += 1

            else:

                searchSetCase2 = self.weights

                self.trainingOperation(self.produceRandomInput(),
                                       searchSetCase2,
                                       searchSetStartCase2,
                                       trainingSetStart,
                                       delta,
                                       radius,
                                       boundaryTraining=True)
                tf.compat.v1.scatter_update(self.weights, self.innerIdx, self.tmpWeights)

                if (not self.plotter == None):
                    self.plotter.plot(timeIt, self.weights[:, 0], self.weights[:, 1], self.connection)
                timeIt += 1

        self.timer.stopTimer("train")

    def summary(self):
        print("_________________________________________________________")
        print("                        ")
        print("Summary of the grid")
        print("_________________________________________________________")
        print("spacing:         ", self.spacing)
        print("dimension:       ", self.dim)
        print("minimum x:       ", self.boundingBox[0, 0])
        print("maximum x:       ", self.boundingBox[0, 1])
        print("minimum y:       ", self.boundingBox[1, 0])
        print("maximum y:       ", self.boundingBox[1, 1])
        print("s:               ", self.s)
        print("iterations:      ", self.iterations)
        print("minRadius :      ", self.minRadius)
        print("maxRadius:       ", self.maxRadius)
        print("noPoints         ", self.noPoints)
        print("noCells:         ", np.shape(self.connection)[0])
        print("noBoundaryCells: ", np.shape(self.boundary)[0])
        print("_________________________________________________________")
