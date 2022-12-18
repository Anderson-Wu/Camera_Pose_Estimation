import open3d as o3d
import cv2 as cv
import numpy as np
from scipy.spatial.transform import Rotation as R
import sys, os
import pandas as pd
import re
import cv2
import argparse
def add_point(cube_vertices):
    pts = []


    #front
    hordis = (cube_vertices[1] - cube_vertices[0])/9
    verdis = (cube_vertices[4] - cube_vertices[0])/9
    for i in range(1,9):
        for j in range(1,9):
            pt =  cube_vertices[0]+hordis*i+verdis*j
            pt = pt.tolist()
            pts.append([pt,(255,0,0)])

    #back
    hordis = (cube_vertices[3] - cube_vertices[2])/9
    verdis = (cube_vertices[6] - cube_vertices[2])/9
    for i in range(1,9):
        for j in range(1,9):
            pt =  cube_vertices[2]+hordis*i+verdis*j
            pt = pt.tolist()
            pts.append([pt,(0,255,0)])


    #up
    hordis = (cube_vertices[3] - cube_vertices[2])/9
    verdis = (cube_vertices[0] - cube_vertices[2])/9
    for i in range(10):
        for j in range(10):
            pt =  cube_vertices[2]+hordis*i+verdis*j
            pt = pt.tolist()
            pts.append([pt,(0,0,255)])


    #down
    hordis = (cube_vertices[7] - cube_vertices[6])/9
    verdis = (cube_vertices[4] - cube_vertices[6])/9
    for i in range(10):
        for j in range(10):
            pt =  cube_vertices[6]+hordis*i+verdis*j
            pt = pt.tolist()
            pts.append([pt,(0,255,255)])


    #left
    hordis = (cube_vertices[0] - cube_vertices[2])/9
    verdis = (cube_vertices[6] - cube_vertices[2])/9
    for i in range(10):
        for j in range(1,9):
            pt =  cube_vertices[2]+hordis*i+verdis*j
            pt = pt.tolist()
            pts.append([pt,(255,255,0)])



    #right
    hordis = (cube_vertices[1] - cube_vertices[3])/9
    verdis = (cube_vertices[7] - cube_vertices[3])/9
    for i in range(10):
        for j in range(1,9):
            pt =  cube_vertices[3]+hordis*i+verdis*j
            pt = pt.tolist()
            pts.append([pt,(255,0,255)])


    return pts


def drawcube(img,rot,tran,cube_vertices,cameraMatrix):
    h,w,ch = img.shape
    rot = R.from_quat(rot).as_matrix().reshape(3,3)
    pts = add_point(cube_vertices)
    pts.sort(key=lambda x:np.linalg.norm(x[0]-tran),reverse=True)
    for index,[vertice,color] in enumerate(pts):
        vertice = np.array(vertice)

        tran = tran.reshape(3)
        u = cameraMatrix.dot(rot.dot((vertice-tran)))
        u = u/u[2]
        u = tuple(u[:2].astype(np.int32))
        if u[0] < 0 or u[1] < 0 or u[1] > h-1 or u[0] > w-1:
            pass
        else:
            img = cv2.circle(img,u , 20, color, -1)
    return img

if __name__ =="__main__":

    parser = argparse.ArgumentParser(description='draw trajectory')
    parser.add_argument('--r', type=str, help='path of Rotation vector\'s npy file', required=True)
    parser.add_argument('--t', type=str, help='path of Translation vector\'s npy file', required=True)
    parser.add_argument('--c',type=str,help='path of cube vertices npy file',required=True)

    args = parser.parse_args()
    path_r = args.r
    path_t = args.t
    path_c = args.c


    cube_vertices = np.load(path_c)
    images_df = pd.read_pickle("data/images.pkl")
    rotation = np.load(path_r)
    translation = np.load(path_t)
    cameraMatrix = np.array([[1868.27, 0, 540], [0, 1869.18, 960], [0, 0, 1]])
    path = './data/frames/'
    files = [f for f in os.listdir(path) if f.find('valid') != -1]
    imagename_point = []

    for index,f in enumerate(files):
        imagename_point.append((int(re.findall('[0-9]+',f)[0]),f))
    imagename_point.sort(key=lambda x:x[0])

    img = cv2.imread(path+imagename_point[0][1])

    h,w,ch = img.shape
    out = cv2.VideoWriter('./video/output.mp4', cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 5, (w,h))
    operatedframe = []
    for index,(fileindex,fname) in enumerate(imagename_point):
        img = cv2.imread(path+fname)
        res = drawcube(img,rotation[index],translation[index],cube_vertices,cameraMatrix)
        out.write(img)

    imagename_point.sort(key=lambda x:x[0])
    out.release()



