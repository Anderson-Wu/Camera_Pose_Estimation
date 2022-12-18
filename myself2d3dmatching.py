import random

from scipy.spatial.transform import Rotation as R
import pandas as pd
import numpy as np
import cv2
import time
import math
import re
import argparse


def average(x):
    return list(np.mean(x,axis=0))

def average_desc(train_df, points3D_df):
    train_df = train_df[["POINT_ID","XYZ","RGB","DESCRIPTORS"]]

    '''
    start = 1
    li = []
    dropli = []
    print(train_df.shape[0])
    for i in range(train_df.shape[0]):
        #print(train_df.loc[i]["POINT_ID"],start)
        if train_df.loc[i]["POINT_ID"] == start:
            li.append(i)
        else:
            randomlist = random.sample(li, int(len(li)*0.5))
            if len(randomlist) != len(li):
                dropli = dropli + randomlist
            li = []
            li.append(i)
            start = start + 1


    print(len(dropli))
    train_df = train_df.drop(train_df.index[dropli])
    print(train_df.shape[0])'''

    desc = train_df.groupby("POINT_ID")["DESCRIPTORS"].apply(np.vstack)
    desc = desc.apply(average)
    desc = desc.reset_index()
    desc = desc.join(points3D_df.set_index("POINT_ID"), on="POINT_ID")
    return desc


def trilaterion(x1,x2,x3,a,b,c):
    try:
        e_x=(x2-x1)/np.linalg.norm(x2-x1)
        i=np.dot(e_x,(x3-x1))
        e_y=(x3-x1-(i*e_x))/(np.linalg.norm(x3-x1-(i*e_x)))
        e_z=np.cross(e_x,e_y)
        d=np.linalg.norm(x2-x1)
        j=np.dot(e_y,(x3-x1))
        x=((a**2)-(b**2)+(d**2))/(2*d)
        y=(((a**2)-(c**2)+(i**2)+(j**2))/(2*j))-((i/j)*(x))
        if a**2-x**2-y**2 < 0:
            z1 = 0
            z2 = 0
        else:
            z1=np.sqrt(a**2-x**2-y**2)
            z2=np.sqrt(a**2-x**2-y**2)*(-1)
        ans1=x1+(x*e_x)+(y*e_y)+(z1*e_z)
        ans2=x1+(x*e_x)+(y*e_y)+(z2*e_z)
    except:
        return None,None
    return ans1,ans2



def p3p(points3D,points2D,cameraMatrix):

    x1,x2,x3,x4 =  points3D[0], points3D[1], points3D[2],points3D[3]

    X1, X2, X3,X4 = np.append(points3D[0], 1), np.append(points3D[1], 1), np.append(points3D[2], 1),np.append(points3D[3], 1)
    u1, u2, u3,u4 = np.append(points2D[0], 1), np.append(points2D[1], 1), np.append(points2D[2], 1),np.append(points2D[3], 1)


    Rab = np.linalg.norm(x1 - x2)
    Rac = np.linalg.norm(x1 - x3)
    Rbc = np.linalg.norm(x2 - x3)

    cameraMatrix_inv = np.linalg.inv(cameraMatrix)

    v1 = cameraMatrix_inv.dot(u1)
    v2 = cameraMatrix_inv.dot(u2)
    v3 = cameraMatrix_inv.dot(u3)

    if Rac < 0.00001 or Rab <0.00001:
        return None,None

    K1 = (Rbc / Rac) ** 2
    K2 = (Rbc / Rab) ** 2


    Cab = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    Cac = np.dot(v1, v3) / (np.linalg.norm(v1) * np.linalg.norm(v3))
    Cbc = np.dot(v2, v3) / (np.linalg.norm(v2) * np.linalg.norm(v3))

    ##polynomial coef
    G4 = (K1 * K2 - K1 - K2) ** 2 - 4 * K1 * K2 * (Cbc ** 2)
    G3 = 4 * (K1 * K2 - K1 - K2) * K2 * (1 - K1) * Cab + 4 * K1 * Cbc * ((K1 * K2 - K1 + K2) * Cac + 2 * K2 * Cab * Cbc)
    G2 = (2 * K2 * (1 - K1) * Cab) ** 2 + 2 * (K1 * K2 - K1 - K2) * (K1 * K2 + K1 - K2) + 4 * K1 * (
                (K1 - K2) * (Cbc ** 2) + K1 * (1 - K2) * (Cac ** 2) - 2 * (1 + K1) * K2 * Cab * Cac * Cbc)
    G1 = 4 * (K1 * K2 + K1 - K2) * K2 * (1 - K1) * Cab + 4 * K1 * (
                (K1 * K2 - K1 + K2) * Cac * Cbc + 2 * K1 * K2 * Cab * (Cac ** 2))
    G0 = (K1 * K2 + K1 - K2) ** 2 - 4 * (K1 ** 2) * K2 * (Cac ** 2)

    coef = [G4, G3, G2, G1, G0]
    res = np.roots(coef)
    solEquation = []
    for i in res:
        if np.isreal(i) == True:
            solEquation.append(np.real(i))

    if len(solEquation) == 0:
        return None,None

    a = []
    b = []
    y = []
    c = []
    for val in solEquation:
        try:
            a_val = math.sqrt((Rab ** 2) / (1 + val ** 2 - 2 * val * Cab))
            a.append(a_val)

            p = 2 * (K1 * Cac - val * Cbc)
            q = val ** 2 - K1
            pprone = 2 * (-val * Cbc)
            qprone = (val ** 2) * (1 - K2) + 2 * val * K2 * Cab - K2
            mprone = 1
            m = 1 - K1
            y_val = -(mprone * q - m * qprone) / (p * mprone - pprone * m)
            y.append(y_val)
            b.append(val * a_val)
            c.append(y_val*a_val)
        except:
            return None,None


    trians1 = []
    trians2 = []
    for i in range(len(solEquation)):
        ans1,ans2 = trilaterion(x1[:3], x2[:3], x3[:3], a[i], b[i], c[i])
        if ans1 is None:
            continue
        trians1.append(ans1)
        trians2.append(ans2)

    if len(trians1) == 0 or len(trians2) == 0:
        return None,None

    lambda1 = a / np.linalg.norm(v1)
    lambda2 = b / np.linalg.norm(v2)
    lambda3 = c / np.linalg.norm(v3)


    sol_rt = []
    for i in range(len(solEquation)):
        lamdavMatrix = np.column_stack((lambda1[i]*v1,lambda2[i]*v2,lambda3[i]*v3))
        xsubtMatrix = np.column_stack((x1-trians1[i],x2-trians1[i],x3-trians1[i]))
        if np.linalg.det(xsubtMatrix) == 0:
            continue
        else:
            r = lamdavMatrix.dot(np.linalg.inv(xsubtMatrix))

        if abs(np.linalg.det(r)-1) < 0.001:
            sol_rt.append([r,trians1[i]])

        lamdavMatrix = np.column_stack((lambda1[i]*v1,lambda2[i]*v2,lambda3[i]*v3))
        xsubtMatrix = np.column_stack((x1-trians2[i],x2-trians2[i],x3-trians2[i]))
        r = lamdavMatrix.dot(np.linalg.inv(xsubtMatrix))

        if abs(np.linalg.det(r)-1) < 0.001:
            sol_rt.append([r,trians2[i]])

    if len(sol_rt) == 0:
        return None,None


    sol_rt_candidaite = []
    for index,[r,tran] in enumerate(sol_rt):
        u,s,vh = np.linalg.svd(r,full_matrices=True)
        rotation = u.dot(vh)
        translation = tran
        sol_rt_candidaite.append((rotation,translation))
    for index,[r,tran] in enumerate(sol_rt):
        u,s,vh = np.linalg.svd(r,full_matrices=True)
        rotation = u.dot(vh)
        translation = tran
        sol_rt_candidaite.append((rotation,translation))



    bestextrinsic =None
    besterr = 0
    for index,(rotation,translation) in enumerate(sol_rt_candidaite):
        proj = rotation.dot(x4-translation)
        proj = proj/proj[2]

        v = (np.linalg.inv(cameraMatrix)).dot(u4)
        v = v/v[2]
        err = np.linalg.norm(proj-v)
        if bestextrinsic == None:
            bestextrinsic = index
            besterr = err
        elif err < besterr:
            bestextrinsic = index
            besterr = err
    if bestextrinsic == None:
        return None,None
    else:
        return sol_rt_candidaite[bestextrinsic][0],sol_rt_candidaite[bestextrinsic][1]



def ransacP3p(points2D,points3D,cameraMatrix):
    best_rot = None
    best_trans = None
    best_inlier = None

    totalpoints = len(points2D)
    sample = 200
    if totalpoints < sample:
        sample = totalpoints

    dividend = 1
    divisor = 1
    for i in range(3):
        divisor = divisor * (i + 1)
        dividend = dividend * (sample - i)
    comb = dividend // divisor

    times = 100
    if comb < times:
        times = comb



    his = []
    for time in range(times):
        inlier = 0
        while True:
            randomlist = random.sample(range(sample), 4)
            randomlist.sort()
            if randomlist in his:
                continue
            else:
                his.append(randomlist)
                break
        samplepoints2D = []
        samplepoints3D = []
        for index in randomlist:
            samplepoints2D.append(points2D[index])
            samplepoints3D.append(points3D[index])

        samplepoints2D = np.array(samplepoints2D)
        samplepoints3D = np.array(samplepoints3D)

        rmat, tvec = p3p(samplepoints3D, samplepoints2D, cameraMatrix)

        if rmat is None:
            continue
        else:
            for index in range(len(points3D)):
                proj = cameraMatrix.dot(rmat.dot((points3D[index] - tvec)))
                proj = proj / proj[2]
                proj = proj[:2]
                point = points2D[index][:2]
                dis = np.linalg.norm(point - proj)

                if dis < 0.1:
                    inlier = inlier + 1
            if best_inlier == None:
                best_inlier = inlier
                best_rot = rmat
                best_trans = tvec
            elif best_inlier < inlier:
                best_inlier = inlier
                best_rot = rmat
                best_trans = tvec
    return best_rot, best_trans


def pnpsolver_self(query,model,cameraMatrix=0,distortion=0):
    kp_query, desc_query = query
    kp_model, desc_model = model


    '''match descriptor of val and average_train_descriptor'''
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(desc_query,desc_model,k=2)

    gmatches = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            gmatches.append(m)


    points2D = np.empty((0,2))
    points3D = np.empty((0,3))

    '''match 2d point in image to 3d point world coor'''
    for mat in gmatches:
        query_idx = mat.queryIdx
        model_idx = mat.trainIdx
        points2D = np.vstack((points2D,kp_query[query_idx]))
        points3D = np.vstack((points3D,kp_model[model_idx]))


    '''intrinstic parameter'''
    cameraMatrix = np.array([[1868.27,0,540],[0,1869.18,960],[0,0,1]])

    distCoeffs = np.array([0.0847023,-0.192929,-0.000201144,-0.000725352])

    points2D = cv2.undistortPoints(points2D,cameraMatrix,distCoeffs,P=cameraMatrix)


    return ransacP3p(points2D,points3D,cameraMatrix)



def pnpsolver(query,model,cameraMatrix=0,distortion=0):
    kp_query, desc_query = query
    kp_model, desc_model = model


    '''match descriptor of val and average_train_descriptor'''
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(desc_query,desc_model,k=2)

    gmatches = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            gmatches.append(m)

    points2D = np.empty((0,2))
    points3D = np.empty((0,3))

    '''match 2d point in image to 3d point world coor'''
    for mat in gmatches:
        query_idx = mat.queryIdx
        model_idx = mat.trainIdx
        points2D = np.vstack((points2D,kp_query[query_idx]))
        points3D = np.vstack((points3D,kp_model[model_idx]))



    '''intrinstic parameter'''
    cameraMatrix = np.array([[1868.27,0,540],[0,1869.18,960],[0,0,1]])    

    distCoeffs = np.array([0.0847023,-0.192929,-0.000201144,-0.000725352])

    return cv2.solvePnPRansac(points3D, points2D, cameraMatrix, distCoeffs)



def loadFile():
    global train_df,points3D_df,desc_df,kp_model,desc_model,images_df,point_desc_df
    train_df = pd.read_pickle("data/train.pkl")
    points3D_df = pd.read_pickle("data/points3D.pkl")

    '''Get average desciptor of 3d point in world coord'''
    '''['POINT_ID', 'DESCRIPTORS', 'XYZ', 'RGB']'''
    desc_df = average_desc(train_df, points3D_df)


    '''Get all world coord of 3d point (111519,3)'''
    kp_model = np.array(desc_df["XYZ"].to_list())

    '''Get mean value of descriptor of 3d point (111519,128)'''
    desc_model = np.array(desc_df["DESCRIPTORS"].to_list()).astype(np.float32)

    '''all pose in train and val images'''
    images_df = pd.read_pickle("data/images.pkl")

    '''point of 3d point to 2d point in train and val images'''
    point_desc_df = pd.read_pickle("data/point_desc.pkl")




def tran_err(tvec,tvec_gt):
    return np.linalg.norm(tvec-tvec_gt)


def rot_err(rotq,rotq_gt):
    rotq = R.from_quat(rotq)
    rotq_inv= rotq.inv()

    rotq_gt = R.from_quat(rotq_gt)
    diff = rotq_gt*rotq_inv
    diff_axis_angle = diff.as_rotvec()
    err = np.linalg.norm(diff_axis_angle)
    return err


if __name__ =="__main__":
    loadFile()
    rot = {}
    tran = {}
    tran_err_list = []
    rot_err_list = []
    for idx in range(164,294): #164 294
        print(idx)
        '''image id start from 1 end 293'''
        # Load quaery image
        fname = ((images_df.loc[images_df["IMAGE_ID"] == idx])["NAME"].values)[0]
        rimg = cv2.imread("data/frames/" + fname, cv2.IMREAD_GRAYSCALE)

        '''get all 3d to 2d points in image'''
        points = point_desc_df.loc[point_desc_df["IMAGE_ID"] == idx]

        # Load query keypoints and descriptors
        '''all 2d points in image'''
        kp_query = np.array(points["XY"].to_list())

        '''descriptor of 2d points in image'''
        desc_query = np.array(points["DESCRIPTORS"].to_list()).astype(np.float32)

        rmat, tvec = pnpsolver_self((kp_query, desc_query), (kp_model, desc_model))
        rvec = (R.from_matrix(rmat)).as_rotvec()


        '''change to quaternion'''
        rotq = R.from_rotvec(rvec.reshape(1, 3)).as_quat()

        fileindex = int(re.findall('[0-9]+', fname)[0])
        rot[fileindex] = rotq
        tran[fileindex] = tvec

        # Get camera pose groudtruth
        ground_truth = images_df.loc[images_df["IMAGE_ID"] == idx]
        rotq_gt = ground_truth[["QX", "QY", "QZ", "QW"]].values
        tvec_gt = ground_truth[["TX", "TY", "TZ"]].values

        print('rot_err is '+str(rot_err(rotq[0],rotq_gt[0])))
        print('tran_err is ' + str(tran_err(-rmat.dot(tvec), tvec_gt[0])))
        rot_err_list.append(rot_err(rotq[0],rotq_gt[0]))
        tran_err_list.append(tran_err(-rmat.dot(tvec),tvec_gt[0]))

    rot = sorted(rot.items(),key=lambda x:x[0])
    tran = sorted(tran.items(), key=lambda x: x[0])

    rot = [val[1] for val in rot]
    tran = [val[1] for val in tran]

    np.save('./pose/Rotation.npy', np.array(rot))
    np.save('./pose/Translation.npy', np.array(tran))
    rot_median_err = np.median(np.array(rot_err_list))
    tran_median_err = np.median(np.array(tran_err_list))

    print('median rotation error: '+str(rot_median_err))
    print('median translation error: '+str(tran_median_err))

