import copy
import math
import os
import time

import cv2
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
import numpy as np
from PIL import Image


def load_off_np(mesh_path):
    with open(mesh_path, 'r') as fp:
        lines = fp.readlines()
        n_points = int(lines[1].split(" ")[0])
        all_strings = ''.join(lines[2:2+n_points])
        array_ = np.fromstring(all_strings, dtype=np.float32, sep='\n')

        all_strings = ''.join(lines[2+n_points:])
        array_int = np.fromstring(all_strings, dtype=np.int32, sep='\n')

        array_ = array_.reshape((-1, 3))

    return array_, array_int.reshape((-1, 4))[:, 1::]


def proj2d(x3d, pose, viewport=2000.0):
    azim = pose['azimuth']
    elev = pose['elevation']
    theta = pose['theta']
    dist = pose['distance']

    C = np.array([
        dist * np.cos(elev) * np.sin(azim),
        - dist * np.cos(elev) * np.cos(azim),
        dist * np.sin(elev)
    ])[:, np.newaxis]

    azim = - azim
    elev = - (np.pi / 2 - elev)

    Rz = np.array([
        [math.cos(azim), -math.sin(azim), 0.0],
        [math.sin(azim), math.cos(azim), 0.0],
        [0.0, 0.0, 1.0]])
    Rx = np.array([
        [1.0, 0.0, 0.0],
        [0.0, math.cos(elev), -math.sin(elev)],
        [0.0, math.sin(elev), math.cos(elev)]])
    R = np.dot(Rx, Rz)

    P = np.dot(np.array([
        [viewport, 0.0, 0.0],
        [0.0, viewport, 0.0],
        [0.0, 0.0, -1.0]]), np.hstack((R, np.dot(-R, C))))

    x = np.dot(P, np.hstack((x3d, np.ones((len(x3d), 1)))).T)
    x = x[0:2, :] / x[2:3, :]

    R2d = np.array([
        [math.cos(theta), -math.sin(theta)],
        [math.sin(theta), math.cos(theta)]])
    x = np.dot(R2d, x).T

    x[:, 1] = - x[:, 1]

    x[:, 0] += pose['px']
    x[:, 1] += pose['py']

    return x


def draw_kp(img, verts, color):
    for i in range(len(verts)):
        img = cv2.circle(img, (int(verts[i, 0]), int(verts[i, 1])), 1, color, -1)
    return img


def draw_patches(img, verts, faces, color=(0, 100, 0), alpha=0.4):
    all_patches = [Polygon(verts[f], closed=True) for f in faces]
    p = PatchCollection(all_patches, alpha=alpha)

    ax = plt.gca()
    ax.add_collection(p)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])

    fname = f'tmp_{np.random.randint(100000000)}.png'
    plt.savefig(fname)
    plt.cla()
    img = np.array(Image.open(fname))
    os.system(f'rm {fname}')

    return img


def matplotlib_render(img_path, mesh_path, pose, alpha=0.4):
    # 1588.42 / 1000
    verts, faces = load_off_np(mesh_path)
    verts2d = proj2d(verts, pose)

    img = np.array(Image.open(img_path).convert('RGB'))

    plt.imshow(img)
    # img = draw_kp(img, verts2d, (0, 255, 0))
    img = draw_patches(img, verts2d, faces, alpha=alpha)

    return Image.fromarray(img)


def draw_lines(img, verts, faces, color=(0, 0, 100), alpha=0.4):
    overlay = copy.deepcopy(img)

    lines = {}
    for (a, b, c) in faces:
        if a > b: d = a; a = b; b = d
        if b > c: d = b; b = c; c = d
        if a > b: d = a; a = b; b = d
        if a not in lines: lines[a] = set()
        if b not in lines: lines[b] = set()
        if b not in lines[a]: overlay = cv2.line(overlay, (int(verts[a, 0]), int(verts[a, 1])), (int(verts[b, 0]), int(verts[b, 1])), color, 1); lines[a].add(b)
        if c not in lines[a]: overlay = cv2.line(overlay, (int(verts[a, 0]), int(verts[a, 1])), (int(verts[c, 0]), int(verts[c, 1])), color, 1); lines[a].add(c)
        if c not in lines[b]: overlay = cv2.line(overlay, (int(verts[b, 0]), int(verts[b, 1])), (int(verts[c, 0]), int(verts[c, 1])), color, 1); lines[b].add(c)

    return cv2.addWeighted(overlay, alpha, img, 1-alpha, 0)


def matplotlib_line_render(img, mesh_path, pose, color=(0, 0, 100), alpha=0.4, add_margin=False, viewport=2000.0):
    # 144.77 / 1000
    verts, faces = load_off_np(mesh_path)
    verts2d = proj2d(verts, pose, viewport=viewport)

    img = draw_lines(img, verts2d, faces, color=color, alpha=alpha)

    if add_margin:
        img[[0, -1], :] = 0
        img[:, [0, -1]] = 0

    return img


def render(img, cad_path, azim, elev, theta, dist, px, py, viewport, good=True):
    if good:
        color = (39, 158, 255)
    else:
        color = (236, 112, 99)
    pose = {
        'azimuth': float(azim),
        'elevation': float(elev),
        'theta': float(theta),
        'distance': float(dist),
        'px': float(px),
        'py': float(py)}
    img = matplotlib_line_render(img, cad_path, pose, color=color, alpha=0.75, viewport=viewport)
    return img
