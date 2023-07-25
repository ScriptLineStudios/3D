import pygame
import math
import numpy as np
from numba import jit
from dataclasses import dataclass

pygame.init()

display = pygame.display.set_mode((600, 600))
clock = pygame.time.Clock()

texture = pygame.image.load("channels4_profile.jpg")
tex_pixels = pygame.surfarray.pixels3d(texture)

"""
We need to map this texture from (A):
    (0, 0) ########## (10, 0)
           ##########
           ##########
           ##########
           ##########
    (0, 6) ########## (10, 6)

to this (B):

    (0, 1) ########## (1, 1)
           ##########
           ##########
           ##########
           ##########
    (0, 0) ########## (1, 0)

given coord (x, y) in system A we need to peform the following:
    (x / width, (height - y) / height)

in reverse:
    (x * width, height - (y * height))
"""

@jit(nopython=True, fastmath=True, cache=True)
def uv_to_coord(u, v, width, height):
    return (int(u * width), int(height - (v * height)))

@jit(nopython=True, fastmath=True, cache=True)
def barycentric(a, b, c, p):
    barya = ((b[1] - c[1]) * (p[0] - c[0]) + (c[0] - b[0]) * (p[1] - c[1])) / ((b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1]))
    baryb = ((c[1] - a[1]) * (p[0] - c[0]) + (a[0] - c[0]) * (p[1] - c[1])) / ((b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1]))
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

width = texture.get_width()
height = texture.get_height()

@jit(nopython=True, fastmath=True, nogil=True)
def draw_triangle(pixels, points, color, uvs, image):
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

    for y in range(ys, ye):
        x1 = xs + int((y - ys) * s1)

        if y <= ym:
            x2 = xs + int((y - ys) * s2)
        else:
            x2 = xm + int((y - ym) * s3)
        
        if x1 > x2:
            x1, x2 = x2, x1

        for x in range(x1, x2):
            _uv = uv(
                barycentric(p1, p2, p3, np.array([x, y])), 
                uv1, uv2, uv3
            )
            u, v = uv_to_coord(_uv[0], _uv[1], width, height)

            pixels[x, y] = image[u-1,v-1]

pixels = pygame.surfarray.pixels3d(display)
x = 0
uvs = np.array([(0, 1), (0, 0), (1, 0)])
while True:
    display.fill((0, 0, 0))
    # print(pixels)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            raise SystemExit

    draw_triangle(pixels, np.array([(0, 100), (100, 400), (200, 200)]), (255, 0, 0), uvs, tex_pixels)
    x += 1
    pygame.display.set_caption(f"{clock.get_fps()}")
    clock.tick()
    pygame.display.update()
