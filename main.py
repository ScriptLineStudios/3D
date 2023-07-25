import pygame
import random
import glm
import math
import numpy as np
from numba import jit
from dataclasses import dataclass

pygame.init()

display = pygame.display.set_mode((1000, 800))
clock = pygame.time.Clock()

materials = {}


with open("creeper/source/creeper.geo.mtl") as f:
    mat_lines = [line.strip() for line in f.readlines()]


def pygame_vec_to_glm_vec3(vec):
    return glm.vec3(vec.x, vec.y, vec.z)

def pygame_vec_to_glm_vec(vec):
    return glm.vec4(vec.x, vec.y, vec.z, 1)

def glm_vec_to_pygame_vec(vec):
    return pygame.Vector3(vec.x, vec.y, vec.z)

data = []
name = ""
for line in mat_lines:
    if line.split(" ")[0] == "newmtl":
        data = []
        name = line.split(" ")[1]
        materials[name] = {}
        materials[name]["color"] = pygame.Color(100, 100, 100)
    else:
        if line.split(" ")[0] == "Kd":
            materials[name]["color"] = pygame.Color(*[int(float(x) * 255) for x in line.split(" ")[1:]])
        elif line.split(" ")[0] == "map_Kd":
            image = pygame.transform.scale(pygame.image.load(line.split(" ")[1]), (100,100))
            materials[name]["pixels"] = pygame.surfarray.pixels3d(image)

with open("creeper/source/creeper.geo.obj", "r") as model:
    lines = [line.strip() for line in model.readlines()]

vertices = []
uvs = []
faces = []

@dataclass
class Face:
    vertices: list
    color: pygame.Color
    uvs: list
    image: any
    width: int = -1
    height: int = -1

def mutate(color):
    amount = 10
    return color
    return pygame.Color(color.r + random.randrange(-amount, amount), color.g + random.randrange(-amount, amount), color.b + random.randrange(-amount, amount))

color = pygame.Color(255, 0, 0)
image = None
for line in lines:
    usemtl = line.split("usemtl ")
    if len(usemtl) > 1:
        color = materials[usemtl[1]]["color"]
        image = materials[usemtl[1]]["pixels"] 
    v = line.split("v ")
    if len(v) > 1:
        vertex = pygame.Vector3([float(x) for x in v[1].split(" ")])
        vertex[2] += 2
        vertices.append(vertex)
    vt = line.split("vt ")
    if len(vt) > 1:
        uv = np.array([float(x) for x in vt[1].split(" ")])
        uvs.append(uv)

    f = line.split("f ")
    if len(f) > 1:
        if "//" in f[1]:
            face = Face([int(x.split("//")[0])-1 for x in f[1].split(" ")], mutate(color), [], image)
        elif "/" in f[1]:
            face = Face([int(x.split("/")[0])-1 for x in f[1].split(" ")], mutate(color), [int(x.split("/")[1])-1 for x in f[1].split(" ")], image)
        else:
            face = Face([int(x)-1 for x in f[1].split(" ")], mutate(color), [], image)
        faces.append(face)

def screen(v):
    return pygame.Vector2((v.x+1)/2 * 1000, (1 - (v.y + 1)/2) * 800)

def three_to_two(v):
    return pygame.Vector2(v.x / (v.z), (v.y-1) / (v.z))

@jit(nopython=True, fastmath=True, cache=True)
def uv_to_coord(u, v, width, height):
    return (int(u * width) + 1, int(height - (v * height) + 1))

@jit(nopython=True, fastmath=True, cache=True)
def barycentric(a, b, c, p):
    barya = ((b[1] - c[1]) * (p[0] - c[0]) + (c[0] - b[0]) * (p[1] - c[1])) / (0.00000001 + (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1]))
    baryb = ((c[1] - a[1]) * (p[0] - c[0]) + (a[0] - c[0]) * (p[1] - c[1])) / (0.00000001 + (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1]))
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


z_buffer = np.tile(np.inf, (1000, 800))
@jit(nopython=True, fastmath=True, nogil=True)
def draw_triangle(pixels, points, color, uvs, image, orig_points, z_buffer):
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

    uv1 = np.array([uvs[indices[0]][0], uvs[indices[0]][1]])
    uv2 = np.array([uvs[indices[1]][0], uvs[indices[1]][1]])
    uv3 = np.array([uvs[indices[2]][0], uvs[indices[2]][1]])

    z1 = orig_points[indices[0]][2]
    z2 = orig_points[indices[1]][2]
    z3 = orig_points[indices[2]][2]
    #Zp = W1 * Z1 + W2 * Z2 + W3 * Z3

    for y in range(ys, ye):
        x1 = xs + int((y - ys) * s1)

        if y < ym:
            x2 = xs + int((y - ys) * s2)
        else:
            x2 = xm + int((y - ym) * s3)
        
        if x1 > x2:
            x1, x2 = x2, x1

        if y > 0 and y < 800:
            for x in range(x1, x2):
                if x > 0 and x < 1000:
                    bary = barycentric(p1, p2, p3, np.array([x, y]));

                    Zp = bary[0] * z1 + bary[1] * z2 + bary[2] * z3
                    if Zp < z_buffer[x, y] and Zp > 1:
                        _uv = uv(
                            bary,
                            uv1, uv2, uv3
                        )
                        u, v = uv_to_coord(_uv[0], _uv[1], image.shape[0], image.shape[1])
                        z_buffer[x, y] = Zp
                        pixels[x, y] = image[u-1,v-1]
                        
cam = pygame.Vector3(0, 0, 0)
pixels = pygame.surfarray.pixels3d(display)
cached = np.tile(np.inf, (1000, 800))
degree = 0
print(faces[0].vertices)
print(vertices[0])
dt = 0
while True:
    degree += 100 * dt
    z_buffer = cached.copy()
    display.fill((43, 69, 100))

    for event in pygame.event.get():
        if event.type == pygame.QUIT or pygame.key.get_pressed()[pygame.K_q]:
            pygame.quit()
            raise SystemExit
    keys = pygame.key.get_pressed()
    if keys[pygame.K_w]:
        cam.z -= 0.1
    if keys[pygame.K_s]:
        cam.z += 0.1
    if keys[pygame.K_a]:
        cam.x += 0.1
    if keys[pygame.K_d]:
        cam.x -= 0.1
    if keys[pygame.K_SPACE]:
        cam.y -= 0.1
    if keys[pygame.K_LSHIFT]:
        cam.y += 0.1


    view = glm.mat4()
    view = glm.rotate(view, glm.radians(degree), glm.vec3(0, 1, 0))
    
    pos = glm.mat4()
    pos = glm.translate(pos, glm.vec3(cam.x, cam.y, cam.z))
    # faces = list(reversed(sorted(faces, key=lambda face: sum([glm_vec_to_pygame_vec(view * pos * pygame_vec_to_glm_vec(vertices[j])).z for j in face.vertices]) / len(face.vertices))))
    for i, face in enumerate(faces):
        calc = [round(screen(three_to_two(glm_vec_to_pygame_vec(pos * view * pygame_vec_to_glm_vec(vertices[vertex]))))) for vertex in face.vertices]
        v = np.array([(calc[0].x, calc[0].y), (calc[1].x, calc[1].y), (calc[2].x, calc[2].y)])
        
        draw_triangle(pixels, v, None, np.array([uvs[uv] for uv in face.uvs]), face.image, np.array([pos * view * pygame_vec_to_glm_vec(vertices[vertex]) for vertex in face.vertices]), z_buffer)

    dt = clock.tick() / 1000
    pygame.display.set_caption(f"Pygame 3D Renderer. Running at {int(clock.get_fps())} fps")
    pygame.display.update()