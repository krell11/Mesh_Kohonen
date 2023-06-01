import random
import pyvista as pv
import numpy as np


class Shape:
    def __init__(self, x, y):
        self.x = x
        self.y = y

class TwoDimensionalShape(Shape):
    def area(self):
        pass

class Rectangle(TwoDimensionalShape):
    def __init__(self, x, y, width, height):
        super().__init__(x, y)
        self.width = width
        self.height = height

    def area(self):
        return self.width * self.height

class Circle(TwoDimensionalShape):
    def __init__(self, x, y, radius):
        super().__init__(x, y)
        self.radius = radius

    def area(self):
        return 3.14159 * self.radius ** 2

class ThreeDimensionalShape(Shape):
    def volume(self):
        pass

class Sphere(ThreeDimensionalShape):
    def __init__(self, x, y, z, radius):
        super().__init__(x, y)
        self.z = z
        self.radius = radius

    def volume(self):
        return (4 / 3) * 3.14159 * self.radius ** 3

def generate_shapes(num_shapes):
    shapes = []
    for _ in range(num_shapes):
        x = random.randint(0, 100)
        y = random.randint(0, 100)
        shape_type = random.choice(["Rectangle", "Circle", "Sphere"])
        if shape_type == "Rectangle":
            width = random.randint(1, 10)
            height = random.randint(1, 10)
            shape = Rectangle(x, y, width, height)
        elif shape_type == "Circle":
            radius = random.randint(1, 10)
            shape = Circle(x, y, radius)
        else:
            z = random.randint(1, 10)
            radius = random.randint(1, 5)
            shape = Sphere(x, y, z, radius)
        shapes.append(shape)
    return shapes

def create_mesh(shape):
    mesh = pv.PolyData()
    if isinstance(shape, TwoDimensionalShape):
        if isinstance(shape, Rectangle):
            vertices = [
                [shape.x, shape.y, 0],
                [shape.x + shape.width, shape.y, 0],
                [shape.x + shape.width, shape.y + shape.height, 0],
                [shape.x, shape.y + shape.height, 0]
            ]
            faces = [[0, 1, 2], [0, 2, 3]]
            mesh_points = pv.PolyData(vertices)
            mesh_faces = pv.PolyData(faces)
            mesh += pv.PolyData(mesh_points.points, mesh_faces.faces)
        elif isinstance(shape, Circle):
            resolution = 50  # Increase resolution for smoother circle
            theta = [2 * 3.14159 * i / resolution for i in range(resolution)]
            vertices = [[shape.x + shape.radius * np.cos(t), shape.y + shape.radius * np.sin(t), 0] for t in theta]
            faces = [[i, (i + 1) % resolution, resolution] for i in range(resolution)]
            mesh_points = pv.PolyData(vertices)
            mesh_faces = pv.PolyData(faces)
            mesh += pv.PolyData(mesh_points.points, mesh_faces.faces)
    elif isinstance(shape, ThreeDimensionalShape):
        sphere = pv.Sphere(radius=shape.radius, center=(shape.x, shape.y, shape.z))
        mesh += sphere
    return mesh


def save_mesh_to_file(mesh, filename):
    mesh.save(filename)


def vtk_to_obj(input_file, output_file):
    with open(input_file, 'r', encoding='latin-1') as vtk:
        lines = vtk.readlines()

    vertices = []
    faces = []

    for line in lines:
        if line.startswith('v '):
            vertex = line.strip().split()[1:]
            vertices.append(vertex)
        elif line.startswith('f '):
            face = line.strip().split()[1:]
            faces.append(face)

    with open(output_file, 'w', encoding='utf-8') as obj:
        for vertex in vertices:
            obj.write(f"v {' '.join(vertex)}\n")

        for face in faces:
            obj.write(f"f {' '.join(face)}\n")


shapes = generate_shapes(10)

for i, shape in enumerate(shapes):
    mesh = create_mesh(shape)
    filename = f"mesh_{i}.vtk"
    save_mesh_to_file(mesh, filename)
    vtk_to_obj('mesh.vtk', 'output.obj')
