import numpy as np
from copy import deepcopy
from .pytUtils import make_rotation_matrix, ProjectShape


def landmark_marching(pitch, yaw, roll, vertex, isoline, keypoints):  
    ProjectVertex = np.dot(make_rotation_matrix(pitch, yaw, roll), vertex)
    ProjectVertex = ProjectVertex - np.min(ProjectVertex, axis=1)[:, np.newaxis] + 1
    ProjectVertex /= np.max(np.abs(ProjectVertex))
    
    keypoints_pose = deepcopy(keypoints)
    # 1. get the keypoints needing modifying    
    if yaw <= 0:
        modify_key = np.array(range(8))
    else:
        modify_key = np.array(range(9,17))
    # 2. get the contour line of each modify key
    contour_line = [isoline[i] for i in modify_key]    
    # 3. get the outest point on the contour line
    for i in range(len(modify_key)):
        if yaw < 0:
            min_ind = np.argmin(ProjectVertex[0, contour_line[i]])
            keypoints_pose[modify_key[i]] = contour_line[i][min_ind]
        else:
            max_ind = np.argmax(ProjectVertex[0, contour_line[i]])          
            keypoints_pose[modify_key[i]] = contour_line[i][max_ind]

    return keypoints_pose 


def retrieve_contour_landmark(fit_result, face_models):
    # NOTE: This only retrieves the contour landmarks.
    # extract parameters from the 3ddfa results
    yaw   = fit_result['face_pose']['yaw']
    pitch = fit_result['face_pose']['pitch']
    roll  = fit_result['face_pose']['roll']
    vertex  = fit_result['vertex']
    fR, T = fit_result['camera_transform']['fR'], fit_result['camera_transform']['T']
    roi_box = fit_result['roi_box']    
    # start retrieving 2D-style landmarks
    new_keypoints_contour = landmark_marching(pitch, yaw, roll, vertex,
                                              face_models['parallel_contour'],
                                              face_models['keypoints_contour'])
    vertex_projected = ProjectShape(vertex, fR, T, roi_box)
    # 3D-style landmarks
    landmarks_3d_style = vertex_projected[:2, face_models['keypoints_contour']].T
    # 2D-style landmarks
    landmarks_2d_style = vertex_projected[:2, new_keypoints_contour].T 

    return landmarks_3d_style, landmarks_2d_style   


def retrieve_contour_landmark_aug(new_result, old_result, face_models):
    # NOTE: This only retrieves the contour landmarks.
    # extract parameters from the 3ddfa results
    roi_box = old_result['roi_box']
    f = old_result['face_pose']['f']
    fR, T = old_result['camera_transform']['fR'], old_result['camera_transform']['T']
    vertex = old_result['vertex']
    
    new_pitch, new_yaw, new_roll = [new_result['face_pose'][key] for key in ['pitch', 'yaw', 'roll']]

    # prepare transform matrix for rotated mesh
    new_fR = f*make_rotation_matrix(new_pitch, new_yaw, new_roll)
    new_t3d = np.mean(fR.dot(vertex)+T, axis=1) - np.mean(new_fR.dot(vertex), axis=1)    
    
    # start retrieving 2D-style landmarks
    new_keypoints_contour = landmark_marching(new_pitch, new_yaw, new_roll, vertex, 
                                              face_models['parallel_contour'], face_models['keypoints_contour'])
    
    vertex_projected = ProjectShape(vertex, new_fR, new_t3d[:, np.newaxis], roi_box)  
    # 3D-style landmarks
    landmarks_3d_style = vertex_projected[:2, face_models['keypoints_contour']].T  
    # 2D-style landmarks
    landmarks_2d_style = vertex_projected[:2, new_keypoints_contour].T    
    
    return landmarks_3d_style, landmarks_2d_style
