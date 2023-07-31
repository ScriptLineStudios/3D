import pygame
import pygame._sdl2
import random
import glm
import math
import numpy as np
from numba import jit, njit
from dataclasses import dataclass
import time
from functools import lru_cache

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
    normals: list
    material: dict
    light: float

@jit(nopython=False, cache=True)
def calculate_polygons(x, position, rotation):
    return x.dot(position).dot(rotation)

def calculate_culling(cam, vertices):
    V0 = glm.vec3(vertices[0][0:3])
    P = glm.normalize(-cam)
    N = glm.cross((glm.vec3(vertices[1][0:3]) - glm.vec3(vertices[0][0:3])), (glm.vec3(vertices[2][0:3]) - glm.vec3(vertices[1][0:3])))
    return glm.dot(glm.normalize(V0 - P), N)
    
class Camera:
    def __init__(self):
        self.position = glm.vec3(0.0, 0.0, 2)
        self.up = glm.vec3(0, 1, 0)
        self.orientation = glm.vec3(0, 0, -1)

        self.speed = .1
        self.hidden = False
        self.sensitivity = 10

    def handle_input(self):
        keys = pygame.key.get_pressed()

        if keys[pygame.K_d]:
            self.position += self.speed * glm.normalize(glm.cross(self.orientation, self.up))
        if keys[pygame.K_a]:
            self.position -= self.speed * glm.normalize(glm.cross(self.orientation, self.up))
        if keys[pygame.K_w]:
            self.position -= self.speed * self.orientation
        if keys[pygame.K_s]:
            self.position += self.speed * self.orientation
        if keys[pygame.K_SPACE]:
            self.position += self.speed * self.up
        if keys[pygame.K_LSHIFT]:
            self.position -= self.speed * self.up

        if self.hidden:
            mx, my = pygame.mouse.get_pos()
            rot_x = self.sensitivity * (my - 400) / 400
            rot_y = self.sensitivity * (mx - 500) / 500
            
            new_orientation = glm.rotate(self.orientation, glm.radians(rot_x), glm.normalize(glm.cross(self.orientation, self.up)))
            self.orientation = new_orientation

            self.orientation = glm.rotate(self.orientation, glm.radians(rot_y), self.up)
            
            pygame.mouse.set_pos((500, 400))

    def update(self):
        self.handle_input()
        view = glm.mat4()

        view = glm.lookAt(self.position, self.position + self.orientation, self.up)
        return view

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
                material["color"] = (float(color[0]) + random.uniform(-0.01, 0.01), 
                float(color[1]) + random.uniform(-0.01, 0.01), 
                float(color[2]) + random.uniform(-0.01, 0.01))
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
        vertex_normals = []
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
                        vertex_textures.append(pygame.Vector2([float(v) for v in vt]))

                    if line.startswith("vn "):
                        vn = line.split(" ")[1:]
                        vertex_normals.append(pygame.Vector3([float(v) for v in vn]))
                    
                    if line.startswith("usemtl "):
                        current_material = materials[line.split("usemtl ")[1]]
                    if line.startswith("f "):
                        vs, vts, vns = [], [], []
                        for point in line.split(" ")[1:]:
                            if "//" not in line:
                                v, vt, vn = point.split("/")
                                v, vt, vn = int(v), int(vt), int(vn)
                                vs.append(v - 1)
                                vts.append(vt - 1)
                                vns.append(vn - 1)
                            else:
                                v, vn = point.split("//")
                                v, vn = int(v), int(vn)
                                vs.append(v - 1)
                                vns.append(vn - 1)
                        
                        faces.append(Face(vs, vts, vns, current_material.copy(), random.uniform(-0.02, 0.02)))
                except StopIteration:
                    break
        return vertices, vertex_textures, vertex_normals, faces

    def __init__(self, obj_file, mtl_file):
        self.materials = self.parse_mtl_file(mtl_file)
        self.vertices, self.uvs, self.normals, self.faces = self.parse_obj_file(obj_file, self.materials)
        print(self.vertices[0])
        self.vertices = np.array(self.vertices, dtype=np.double)
        self.uvs = np.array(self.uvs)
        self.normals = np.array(self.normals)
        self.faces = np.array(self.faces)

        self.position_matrix = glm.mat4()
        self.position = glm.vec3()

        self.rotation_matrix = glm.mat4()

        self.average_z = 0

    @staticmethod
    @jit(nopython=True, fastmath=True, nogil=True)
    def screen(v):
        return np.column_stack((((v[:, 0] + 1) / 2) * 1000, (1 - (v[:, 1] + 1) / 2) * 800))

    @staticmethod
    @jit(nopython=True, fastmath=True, nogil=True)
    def three_to_two(v):
        return np.column_stack(((v[:, 0] / (v[:, 2] + 1)), (v[:, 1] / (v[:, 2] + 1))))

    @staticmethod
    def color(_color):
        off = 0
        _color = np.array([
            int(float(_color[0]) * 255), 
            int(float(_color[1]) * 255), 
            int(float(_color[2]) * 255),
        ])
        return _color

    def render(self, display, matrix, light, camera):
        vertices = self.vertices.copy()
        old_vertices = vertices

        self.position_matrix = glm.mat4()
        self.rotation_matrix = glm.mat4()

        self.position_matrix = glm.translate(self.position_matrix, self.position)

        for i, vertex in enumerate(vertices):
            v = glm.vec4(vertex)
            vertices[i] = (matrix * self.position_matrix * self.rotation_matrix) * v
            self.average_z += vertices[i][2]

        self.average_z /= len(vertices)

        self.faces = sorted(
            self.faces, 
            key=lambda face: -sum([vertices[j][2] for j in face.vertices]) / len(face.vertices)
        )

        screen_vertices = self.screen(self.three_to_two(vertices))

        for face in self.faces:
            z = sum([old_vertices[j][2] for j in face.vertices]) / len(face.vertices)
            normals = self.normals[face.normals]
            if z < 0.7:
                continue

            cull = calculate_culling(camera.orientation, vertices[face.vertices])
  
            if cull > 0:
                continue

            d = glm.max(glm.dot(glm.vec3(normals[0]), -light), 0.0)
            color = glm.vec3(face.material["color"])
            if d > 0.4:
                # pygame.draw.polygon(display, Model.color(color * d), vertices[face.vertices])
                color = Model.color(color * d)
                r = int(color[0])
                g = int(color[1])
                b = int(color[2])
                a = 255

                display.draw_color = r, g, b, a

                if len(screen_vertices[face.vertices]) == 3:
                    display.fill_triangle(*screen_vertices[face.vertices])
            else:
                # pygame.draw.polygon(display, Model.color((color * (d + 0.2 + face.light))), vertices[face.vertices])
                color = Model.color((color * (d + 0.2 + face.light)))
                r = int(color[0])
                g = int(color[1])
                b = int(color[2])
                a = 255

                display.draw_color = r, g, b, a
            
                if len(screen_vertices[face.vertices]) == 3:
                    display.fill_triangle(*screen_vertices[face.vertices])

display = pygame._sdl2.Window('Pygame 3D Renderer', (1000, 800))
renderer = pygame._sdl2.Renderer(display)

model1 = Model("Cube/C.obj", "Cube/C.mtl")
model1.direction = glm.vec3(.0, .0, .0)

model2 = Model("Model/model.obj", "Model/model.mtl")
model2.direction = glm.vec3(-.01, .0, .0)

models = [model1, model2]

clock = pygame.time.Clock()

camera = Camera()

num_frames = 0
avg_fps = 0
x = 0

while True:
    renderer.draw_color = (0, 0, 0, 255)
    renderer.clear()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            raise SystemExit

        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                pygame.mouse.set_pos((500, 400))
                pygame.mouse.set_visible(camera.hidden)
                camera.hidden = not camera.hidden
     
    matrix = camera.update()

    models = sorted(models, key=lambda model: model.average_z)
    for i, model in enumerate(reversed(models)):
        model.average_z = 0
        model.render(renderer, matrix, glm.normalize(glm.vec3(-1, -1, -1)), camera)
        model.position += model.direction
    
    clock.tick()
    avg_fps += int(clock.get_fps())
    num_frames += 1
    display.title = f"Pygame 3D Renderer. Running at an average of: {avg_fps // num_frames} fps" 
    renderer.present()