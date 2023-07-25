import pygame
import random
import glm
import math
import numpy as np
from numba import jit, njit
from dataclasses import dataclass

def pygame_vec_to_glm_vec3(vec):
    return glm.vec3(vec.x, vec.y, vec.z)

def pygame_vec_to_glm_vec(vec):
    return glm.vec4(vec[0], vec[1], vec[2], 1)

def numpy_vecs_to_glm_vec(vecs):
    return glm.vec4(vec[0], vec[1], vec[2], 1)

def glm_vec_to_pygame_vec(vec):
    return pygame.Vector3(vec.x, vec.y, vec.z)

@jit(nopython=True, fastmath=True, cache=True)
def uv_to_coord(u, v, width, height):
    return (int(u * width), int(height - (v * height)))

@jit(nopython=True, fastmath=True, cache=True)
def barycentric(a, b, c, p):
    barya = ((b[1] - c[1]) * (p[0] - c[0]) + (c[0] - b[0]) * (p[1] - c[1])) / (0.00001 + (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1]))
    baryb = ((c[1] - a[1]) * (p[0] - c[0]) + (a[0] - c[0]) * (p[1] - c[1])) / (0.00001 + (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1]))
    baryc = 1 - barya - baryb
    return np.array(
        [barya,
        baryb,
        baryc]
    )

@jit(nopython=True, fastmath=True, cache=True)
def uv(bary, uv1, uv2, uv3):
    u = bary[0] * uv1[0] + bary[1] * uv2[0] + bary[2] * uv3[0]
    v = bary[0] * uv1[1] + bary[1] * uv2[1] + bary[2] * uv3[1]
    return np.array([u, v])


@jit(nopython=True, fastmath=True, nogil=True)
def draw_triangle(pixels, points, color):
    indices = points[:, 1].argsort()

    xs, ys = points[indices[0]]
    xm, ym = points[indices[1]]
    xe, ye = points[indices[2]]

    s1 = (xe - xs) / ((ye - ys) + 0.000000000001)
    s2 = (xm - xs) / ((ym - ys) + 0.000000000001)
    s3 = (xe - xm) / ((ye - ym) + 0.000000000001)

    p1 = np.array([xs, ys])
    p2 = np.array([xm, ym])
    p3 = np.array([xe, ye])

    for y in range(ys, ye):
        x1 = xs + int((y - ys) * s1)

        if y < ym:
            x2 = xs + int((y - ys) * s2)
        else:
            x2 = xm + int((y - ym) * s3)
        
        if x1 > x2:
            x1, x2 = x2, x1

        pixels[x1:x2, y] = color
        # for x in range(x1, x2):
        #     _uv = uv(
        #         barycentric(p1, p2, p3, np.array([x, y])), 
        #         uv1, uv2, uv3
        #     )
        #     u, v = uv_to_coord(_uv[0], _uv[1], image.shape[0], image.shape[1])
        #     pixels[x, y] = image[u-1,v-1]

@dataclass
class Face:
    vertices: list
    uvs: list   
    material: dict

@jit(nopython=False, cache=True)
def calculate_polygons(x, position, rotation):
    return x.dot(position).dot(rotation)

class Model:
    @staticmethod
    def parse_material(f):
        material = {}
        while True:
            try:
                line = next(f).strip()
            except StopIteration:
                break
            if line.startswith("Kd"):
                color = line.split(" ")[1:]
                material["color"] = (float(color[0]), float(color[1]), 
                float(color[2]))
            if line.startswith("map_Kd"):
                path = line.split(" ")[1]
                try:
                    material["image"] = pygame.surfarray.pixels3d(pygame.image.load(path))
                except:
                    pass
            if not line:
                break
        return material

    @staticmethod   
    def parse_mtl_file(mtl_file):
        materials = {}
        print("parsing mtl")
        lines = []
        with open(mtl_file, "r") as f:
            line = ""
            while True:
                try:
                    line = next(f).strip()
                except StopIteration:
                    break
                if line.startswith("newmtl"):
                    name = line.split(" ")[1]
                    material = Model.parse_material(f)
                    materials[name] = material
        return materials

    @staticmethod
    def parse_obj_file(obj_file, materials):
        vertices = []
        vertex_textures = []
        faces = []
        current_material = {}
        with open(obj_file, "r") as f:
            line = ""
            while True:
                try:
                    line = next(f).strip()
                    if line.startswith("v "):
                        vertex = line.split(" ")[1:]
                        vertex = pygame.Vector3([float(v) for v in vertex])
                        vertex.z += 2
                        vertices.append(pygame_vec_to_glm_vec(pygame.Vector3(vertex.x, vertex.y, vertex.z)))
                    
                    if line.startswith("vt "):
                        vt = line.split(" ")[1:]
                        vertex_textures.append(pygame.Vector3([float(v) for v in vertex]))
                    if line.startswith("usemtl "):
                        current_material = materials[line.split("usemtl ")[1]]
                    if line.startswith("f "):
                        vs, vts = [], []
                        for point in line.split(" ")[1:]:
                            if "//" not in line:
                                v, vt, vn = point.split("/")
                                v, vt, vn = int(v), int(vt), int(vn)
                                vs.append(v - 1)
                                vts.append(vt - 1)
                            else:
                                v, vn = point.split("//")
                                v, vn = int(v), int(vn)
                                vs.append(v - 1)
                        
                        faces.append(Face(vs, vts, current_material))
                except StopIteration:
                    break
        return vertices, vertex_textures, faces

    def __init__(self, obj_file, mtl_file):
        self.materials = self.parse_mtl_file(mtl_file)
        self.vertices, self.uvs, self.faces = self.parse_obj_file(obj_file, self.materials)
        self.vertices = np.array(self.vertices, dtype=np.double)
        self.uvs = np.array(self.uvs)
        self.faces = np.array(self.faces)

    @staticmethod
    @jit(nopython=True, fastmath=True, nogil=True)
    def screen(v):
        return np.column_stack(((v[:, 0] + 1) / 2 * 1000, (1 - (v[:, 1] + 1) / 2) * 800))

    @staticmethod
    @jit(nopython=True, fastmath=True, nogil=True)
    def three_to_two(v):
        return np.column_stack(((v[:, 0] / v[:, 2]), (v[:, 1] / v[:, 2])))

    @staticmethod
    def color(_color):
        off = 0
        _color = np.array([int(float(_color[0]) * 255) + off, int(float(_color[1]) * 255) + off, int(float(_color[2]) * 255) + off])
        return _color

    def render(self, display, position, rotation):
        vertices = model.vertices.dot(position).dot(rotation)
        # self.faces = sorted(
        #     self.faces, 
        #     key=lambda face: -sum([vertices[j][3] for j in face.vertices]) / len(face.vertices)
        # )
        vertices = self.screen(self.three_to_two(vertices))
        
        closest_z = 99999999999
        for face in self.faces:
            # face_z = sum([vertices[j][3] for j in face.vertices]) / len(face.vertices)
            # if face_z < closest_z:
            pygame.draw.polygon(display, Model.color(face.material["color"]), vertices[face.vertices])
                # closest_z = face.z
            # draw_triangle(display, vertices[face.vertices], Model.color(face.material["color"]))

display = pygame.display.set_mode((1000, 800))
pixels = pygame.surfarray.pixels3d(display)
model = Model("Tree/model.obj", "Tree/materials.mtl")

pos = glm.mat4()
pos = glm.translate(pos, glm.vec3(8, 0, 10))
pos = glm.scale(pos, glm.vec3(10, 10, 10))
pos = np.array(pos)

# print(np.array(model.vertex_indices.reshape(208, 4)).dot(np.array(pos))) 
clock = pygame.time.Clock()

x = 0
while True:
    display.fill((0, 0, 0))

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            raise SystemExit


    pos = glm.mat4()
    pos = glm.translate(pos, glm.vec3(8, 0, 40))
    pos = glm.scale(pos, glm.vec3(10, 10, 10))
    pos = np.array(pos, dtype=np.double)

    rot = glm.mat4()
    rot = glm.rotate(rot, x, glm.vec3(0,0,1))
    model.render(display, pos, rot)

    x += 0.1

    clock.tick()
    pygame.display.set_caption(f"{clock.get_fps()}")
    pygame.display.update()