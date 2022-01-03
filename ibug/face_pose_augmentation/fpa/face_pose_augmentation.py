import cv2
import igraph
import itertools
import numpy as np
from shapely.ops import nearest_points
from shapely.geometry import Point, Polygon
from typing import Dict, Optional, Sequence, Tuple
from .utils import (make_rotation_matrix, project_shape, image_meshing, image_rotation,
                    create_rotated_correspondence_map, remap_image, model_completion_bfm,
                    z_buffer_tri, calc_barycentric_coordinates)


__all__ = ['generate_profile_faces', 'generate_profile_face']


def generate_profile_faces(delta_poses: np.ndarray, fit_result: Dict, image: np.ndarray,
                           face_models: Dict, return_corres_map: bool = False,
                           further_adjust_z: bool = False, landmarks: Optional[np.ndarray] = None,
                           mouth_point_indices: Sequence[int] = range(48, 68)) -> Tuple[np.ndarray, np.ndarray]:
    if delta_poses.size == 0:
        return np.array([]), np.array([])

    # 1. Get fitting result from 3DDFA (vertex is without affine transform and image space transform)
    roi_box = fit_result['roi_box']
    vertex = fit_result['vertex']
    f_rot, tr = fit_result['camera_transform']['fR'], fit_result['camera_transform']['T']
    
    yaw, pitch, roll = [fit_result['face_pose'][key] for key in ['yaw', 'pitch', 'roll']]
    t3d, f = fit_result['face_pose']['t3d'], fit_result['face_pose']['f']

    vertex_full, tri_full = model_completion_bfm(
        vertex, face_models['Model_FWH'], face_models['Model_Completion'], face_models['conn_point_info'])

    im_height, im_width = image.shape[:2]

    # apply camera rotation to the 3D mesh    
    projected_vertex_full = project_shape(vertex_full, f_rot, tr, roi_box)
    projected_vertexm_full = project_shape(face_models['vertexm_full'], f_rot, tr, roi_box)

    # 2. Image Meshing
    contlist_src, bg_tri, face_contour_ind, wp_num, hp_num = image_meshing(
        vertex, vertex_full, tri_full, projected_vertex_full, projected_vertexm_full, f_rot, tr, roi_box, pitch, yaw,
        face_models['keypoints'], face_models['keypointsfull_contour'], face_models['parallelfull_contour'],
        im_width, im_height, face_models['layer_width'], eliminate_inner_tri=further_adjust_z)

    bg_vertex_src = np.hstack(contlist_src)
    all_vertex_src = np.hstack([bg_vertex_src, projected_vertex_full])
    all_tri = np.hstack([bg_tri, tri_full + bg_vertex_src.shape[1]])
    bg_tri_alt = bg_tri.copy()
    for idx, vt in enumerate(face_contour_ind):
        bg_tri_alt[bg_tri == idx] = vt + bg_vertex_src.shape[1]

    # 2.5 Find best matching triangles for the landmarks
    matching_triangles = dict()
    if landmarks is not None:
        all_tri[:, :bg_tri.shape[1]] = bg_tri_alt
        face_graph_edges = set()
        for tri in all_tri.T:
            for a, b in itertools.combinations(tri, 2):
                if a > b:
                    a, b = b, a
                face_graph_edges.add((a, b))
        face_graph_weights = np.linalg.norm(all_vertex_src.T[[x[0] for x in face_graph_edges]] -
                                            all_vertex_src.T[[x[1] for x in face_graph_edges]], axis=1)
        face_graph = igraph.Graph(edges=face_graph_edges, directed=False)
        for idx, pt in enumerate(landmarks):
            pt_barycentric = calc_barycentric_coordinates(pt, all_vertex_src.T[:, :2], all_tri.T)
            potential_matches = np.where(np.all(np.hstack((0.0 <= pt_barycentric, pt_barycentric <= 1.0)), axis=1))[0]
            if idx in mouth_point_indices and not np.any(np.logical_and(bg_tri.shape[1] <= potential_matches,
                                                                        potential_matches < bg_tri.shape[1] +
                                                                        face_models['tri_plus'].shape[1])):
                # Fix the mouth point if it falls into a hole
                lm = Point(pt)
                closest_pts = []
                for tri_idx in range(bg_tri.shape[1] + face_models['tri'].shape[1],
                                     bg_tri.shape[1] + face_models['tri'].shape[1] +
                                     face_models['tri_mouth'].shape[1]):
                    mouth_tri = Polygon([all_vertex_src[:2, x] for x in all_tri[:, tri_idx]])
                    _, closest_pt = nearest_points(lm, mouth_tri)
                    closest_pts.append([closest_pt.x, closest_pt.y])
                closest_pts = np.vstack(closest_pts)
                closest_idx = np.argmin(np.linalg.norm(closest_pts - pt, axis=1))
                matching_tri_idx = bg_tri.shape[1] + face_models['tri'].shape[1] + closest_idx
                matching_triangles[idx] = (matching_tri_idx, calc_barycentric_coordinates(
                    closest_pts[closest_idx], all_vertex_src.T[:, :2],
                    np.expand_dims(all_tri[:, matching_tri_idx], 0))[0])
            elif len(potential_matches) > 0:
                potential_tri = all_tri[:, potential_matches]
                paths_to_potential_tri = face_graph.get_shortest_paths(
                    v=face_models['keypoints'][idx] + bg_vertex_src.shape[1], to=potential_tri.flatten(),
                    weights=face_graph_weights, mode=igraph.ALL, output='epath')
                dists_to_potential_tri = np.array([face_graph_weights[p].sum() for
                                                   p in paths_to_potential_tri]).reshape((3, -1))
                a = all_vertex_src[:, potential_tri[0]]
                b = all_vertex_src[:, potential_tri[1]]
                c = all_vertex_src[:, potential_tri[2]]
                potential_pt_3d = (a * pt_barycentric[potential_matches, 0] +
                                   b * pt_barycentric[potential_matches, 1] +
                                   c * pt_barycentric[potential_matches, 2])
                pt_dists_to_a = np.linalg.norm(potential_pt_3d - a, axis=0)
                pt_dists_to_b = np.linalg.norm(potential_pt_3d - b, axis=0)
                pt_dists_to_c = np.linalg.norm(potential_pt_3d - c, axis=0)
                match = np.argmin((dists_to_potential_tri +
                                   np.vstack((pt_dists_to_a, pt_dists_to_b, pt_dists_to_c))).min(axis=0))
                matching_triangles[idx] = (potential_matches[match], pt_barycentric[potential_matches[match]])
        all_tri[:, :bg_tri_alt.shape[1]] = bg_tri

    maps_or_images = []
    warped_landmarks = []
    for pitch_delta, yaw_delta, roll_delta in delta_poses:
        # 3. Rotating and Anchor Adjustment
        pitch_ref = pitch + pitch_delta
        yaw_ref   = yaw + yaw_delta
        roll_ref  = roll + roll_delta
        R_ref = make_rotation_matrix(pitch_ref, yaw_ref, roll_ref)

        t3d_ref = np.mean(f_rot.dot(vertex_full)+tr, axis=1) - np.mean(f*R_ref.dot(vertex_full), axis=1)
        RefVertex = project_shape(vertex_full, f*R_ref, t3d_ref[:, np.newaxis], roi_box)

        Pose_Para_src = np.array([pitch, yaw, roll]+list(t3d)+[f])
        Pose_Para_ref = np.array([pitch_ref, yaw_ref, roll_ref] + list(t3d_ref) + [f])

        contlist_ref = image_rotation(contlist_src, bg_tri, vertex_full, face_models['keypointsfull_contour'],
                                      face_models['parallelfull_contour'], Pose_Para_src, Pose_Para_ref,
                                      RefVertex, f_rot, tr, roi_box)

        bg_vertex_ref = np.hstack(contlist_ref)
        all_vertex_ref = np.hstack([bg_vertex_ref, RefVertex])

        bg_tri_num = bg_tri.shape[1]
        
        # Further adjust z
        if further_adjust_z:
            if yaw_ref < 0:
                face_contour_modify = np.array(list(range(8)) + list(range(24, 30)))
            else:
                face_contour_modify = np.array(range(9, 23))

            face_contour_nonmodify = sorted(list(set(range(len(
                face_models['keypointsfull_contour']))).difference(set(face_contour_modify)))) 

            zmax_bin = []
            for i in range(len(contlist_ref)-1): 
                cont = contlist_ref[i]
                tmp_bin = np.zeros(cont.shape[1])
                tmp_bin[face_contour_nonmodify] = 1
                zmax_bin.append(tmp_bin)

            zmax_bin.append(np.zeros(contlist_ref[-1].shape[1]))
            zmax_bin = np.hstack(zmax_bin)
            zmax_ind = np.where(zmax_bin==1)[0]

            tmp_bin = np.in1d(bg_tri.ravel(), zmax_ind).reshape((3,-1))
            tmp_bin = np.any(tmp_bin, axis=0)
            zmax_ind = np.unique(bg_tri[:,tmp_bin])
            all_vertex_ref[2, zmax_ind] = all_vertex_ref[2].max()

        # 4. Get Correspondence
        tri_ind = z_buffer_tri(all_vertex_ref, all_tri, im_width, im_height)[0]
        corres_map = create_rotated_correspondence_map(tri_ind, all_vertex_src, all_vertex_ref, all_tri)
        if yaw_delta < 0:
            pts = np.vstack(([corres_map.shape[1], -1], 
                             contlist_ref[-1][:2, wp_num + 1: wp_num + hp_num + 3].T - 1,
                             [corres_map.shape[1], corres_map.shape[0]]))
            masked = corres_map[..., 0].copy()
            cv2.fillPoly(masked, [pts.astype(np.int32)], -1.0)
            corres_map[..., 0] = masked
            masked = corres_map[..., 1].copy()
            cv2.fillPoly(masked, [pts.astype(np.int32)], -1.0)
            corres_map[..., 1] = masked
        elif yaw_delta > 0:
            pts = np.vstack(([-1, corres_map.shape[0]], contlist_ref[-1][:2, -hp_num - 1:].T,
                             contlist_ref[-1][:2, 0].T, [-1, -1]))
            masked = corres_map[..., 0].copy()
            cv2.fillPoly(masked, [pts.astype(np.int32)], -1.0)
            corres_map[..., 0] = masked
            masked = corres_map[..., 1].copy()
            cv2.fillPoly(masked, [pts.astype(np.int32)], -1.0)
            corres_map[..., 1] = masked
        if return_corres_map:
            maps_or_images.append(corres_map)
        else:
            maps_or_images.append(remap_image(image, corres_map))

        # get the landmarks
        if landmarks is not None:
            warped_landmarks.append((all_vertex_ref[:, face_models['keypoints']+bg_vertex_src.shape[1]],
                                     all_vertex_ref[:, face_models['keypoints']+bg_vertex_src.shape[1]].copy()))
            for idx, (tri_idx, pt_barycentric) in matching_triangles.items():
                a = all_vertex_ref[:, all_tri[0, tri_idx]]
                b = all_vertex_ref[:, all_tri[1, tri_idx]]
                c = all_vertex_ref[:, all_tri[2, tri_idx]]
                warped_landmarks[-1][-1][:, idx] = (a * pt_barycentric[0] + b * pt_barycentric[1] +
                                                    c * pt_barycentric[2])
        else:
            warped_landmarks.append(all_vertex_ref[:, face_models['keypoints']+bg_vertex_src.shape[1]])
    
    return np.stack(maps_or_images), np.stack(warped_landmarks)


def generate_profile_face(pitch_delta: float, yaw_delta: float, roll_delta: float,
                          fit_result: Dict, image: np.ndarray, face_models: Dict,
                          return_corres_map: bool = False, landmarks: Optional[np.ndarray] = None,
                          mouth_point_indices: Sequence[int] = range(48, 68)) -> Tuple[np.ndarray, np.ndarray]:
    map_or_image, warped_landmarks = generate_profile_faces(np.array([(pitch_delta, yaw_delta, roll_delta)]),
                                                            fit_result, image, face_models, return_corres_map,
                                                            landmarks, mouth_point_indices)
    return map_or_image[0], warped_landmarks[0]
