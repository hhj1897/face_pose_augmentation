import cv2
import igraph
import itertools
import numpy as np
from shapely.ops import nearest_points
from shapely.geometry import Point, Polygon
from typing import Dict, Optional, Sequence, Tuple
from .utils import (make_rotation_matrix, project_shape, image_meshing, image_rotation,
                    create_correspondence_map, remap_image, model_completion_bfm,
                    z_buffer_tri, calc_barycentric_coordinates, refine_contour_points)


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
    f_rot, tr = [fit_result['camera_transform'][key] for key in ('fR', 'T')]
    yaw, pitch, roll = [fit_result['face_pose'][key] for key in ('yaw', 'pitch', 'roll')]
    t3d, f = [fit_result['face_pose'][key] for key in ('t3d', 'f')]
    vertex_full, tri_full = model_completion_bfm(
        vertex, face_models['Model_FWH'], face_models['Model_Completion'], face_models['conn_point_info'])
    im_height, im_width = image.shape[:2]

    # apply camera rotation to the 3D mesh    
    projected_vertex_src = project_shape(vertex_full, f_rot, tr, roi_box)
    projected_vertexm_src = project_shape(face_models['vertexm_full'], f_rot, tr, roi_box)

    # 2. Image Meshing
    contlist_src, bg_tri, face_contour_ind, wp_num, hp_num, inner_bg_tri_ind = image_meshing(
        vertex_full, tri_full, projected_vertex_src, projected_vertexm_src, f_rot, tr, roi_box, pitch, yaw,
        face_models['keypoints'], face_models['keypointsfull_contour'], face_models['parallelfull_contour'],
        im_width, im_height, face_models['layer_width'], eliminate_inner_tri=further_adjust_z,
        anchor_z_medians=face_models['anchor_stats']['anchor_z_medians'],
        anchor_radial_z_dist_thresholds=face_models['anchor_radial_z_dist_thresholds'])

    bg_vertex_src = np.hstack(contlist_src)
    all_vertex_src = np.hstack([bg_vertex_src, projected_vertex_src])
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
        all_tri_x = np.vstack([all_vertex_src[0, all_tri[x, :]] for x in (0, 1, 2)])
        all_tri_y = np.vstack([all_vertex_src[1, all_tri[x, :]] for x in (0, 1, 2)])
        all_tri_left = np.min(all_tri_x, axis=0)
        all_tri_right = np.max(all_tri_x, axis=0)
        all_tri_top = np.min(all_tri_y, axis=0)
        all_tri_bottom = np.max(all_tri_y, axis=0)
        for idx, pt in enumerate(landmarks[:len(face_models['keypoints'])]):
            potential_tri_ind = np.where(np.all([all_tri_left <= pt[0], pt[0] <= all_tri_right,
                                                 all_tri_top <= pt[1], pt[1] <= all_tri_bottom], axis=0))[0]
            if len(potential_tri_ind) > 0:
                pt_barycentric = calc_barycentric_coordinates(pt, all_vertex_src.T[:, :2],
                                                              all_tri.T[potential_tri_ind])
                potential_matches = np.where(np.all(np.hstack(
                    (0.0 <= pt_barycentric, pt_barycentric <= 1.0)), axis=1))[0]
                if idx in mouth_point_indices and len(
                        [x for x in potential_tri_ind[potential_matches] if
                         x not in inner_bg_tri_ind and x < bg_tri.shape[1] + face_models['tri_plus'].shape[1]]) == 0:
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
                    if len(potential_matches) > 1:
                        potential_tri = all_tri[:, potential_tri_ind[potential_matches]]
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
                    else:
                        match = 0
                    matching_triangles[idx] = (potential_tri_ind[potential_matches[match]],
                                               pt_barycentric[potential_matches[match]])
        all_tri[:, :bg_tri_alt.shape[1]] = bg_tri

    maps_or_images = []
    warped_landmarks = []
    for pitch_delta, yaw_delta, roll_delta in delta_poses:
        # 3. Rotating and Anchor Adjustment
        pitch_ref = pitch + pitch_delta
        yaw_ref = yaw + yaw_delta
        roll_ref = roll + roll_delta
        rot_mat_ref = make_rotation_matrix(pitch_ref, yaw_ref, roll_ref)
        t3d_ref = np.mean(f_rot.dot(vertex_full) + tr, axis=1) - np.mean(f * rot_mat_ref.dot(vertex_full), axis=1)

        projected_vertex_ref = project_shape(vertex_full, f * rot_mat_ref, t3d_ref[:, np.newaxis], roi_box)
        pose_params_src = [pitch, yaw, roll] + t3d.tolist() + [f]
        pose_params_ref = [pitch_ref, yaw_ref, roll_ref] + t3d_ref.tolist() + [f]

        contlist_ref = image_rotation(contlist_src, bg_tri, vertex_full, face_models['keypointsfull_contour'],
                                      face_models['parallelfull_contour'], pose_params_src, pose_params_ref,
                                      projected_vertex_ref, f_rot, tr, roi_box)

        bg_vertex_ref = np.hstack(contlist_ref)
        all_vertex_ref = np.hstack([bg_vertex_ref, projected_vertex_ref])

        # Further adjust z
        if further_adjust_z:
            if yaw_ref < 0:
                face_contour_modify = list(range(8)) + list(range(24, 30))
            else:
                face_contour_modify = list(range(9, 23))
            face_contour_no_modify = list(
                set(range(len(face_models['keypointsfull_contour']))).difference(set(face_contour_modify)))
            zmax_bins = [np.zeros(contour.shape[1], dtype=bool) for contour in contlist_ref]
            for bins, contour in zip(zmax_bins[:-1], contlist_ref[:-1]):
                bins[face_contour_no_modify] = True
            zmax_bins = np.hstack(zmax_bins)
            tri_ind = [idx for idx, tri in enumerate(bg_tri.T) if np.any(zmax_bins[tri])]
            zmax_ind = np.unique(bg_tri[:, tri_ind])
            all_vertex_ref[2, zmax_ind] = all_vertex_ref[2].max()

        # 4. Get Correspondence
        tri_ind = z_buffer_tri(all_vertex_ref, all_tri, im_width, im_height)[0]
        corres_map = create_correspondence_map(tri_ind, all_vertex_src, all_vertex_ref, all_tri)
        if pitch_delta != 0 or yaw_delta != 0:
            border_region = None
            corres_map_diag = np.linalg.norm(corres_map.shape[:2])
            enlarge_factor = ((corres_map_diag + 16.0) / min(corres_map.shape[:2]) - 1.0) / 2.0
            enlarged_corres_map_corners = np.array([
                [-corres_map.shape[1] * enlarge_factor, -corres_map.shape[0] * enlarge_factor],
                [corres_map.shape[1] * (1.0 + enlarge_factor), -corres_map.shape[0] * enlarge_factor],
                [corres_map.shape[1] * (1.0 + enlarge_factor), corres_map.shape[0] * (1.0 + enlarge_factor)],
                [-corres_map.shape[1] * enlarge_factor, corres_map.shape[0] * (1.0 + enlarge_factor)]])
            corres_map_centre = np.array(corres_map.shape[1::-1]) / 2.0
            enlarged_corres_map_corners = (enlarged_corres_map_corners - corres_map_centre).dot(
                np.array([[np.cos(roll_delta), -np.sin(roll_delta)],
                          [np.sin(roll_delta), np.cos(roll_delta)]])) + corres_map_centre
            border_tl, border_tr, border_br, border_bl = enlarged_corres_map_corners
            img_contour_tl, img_contour_tr, img_contour_br, img_contour_bl = (
                0, wp_num + 1, wp_num + hp_num + 2, wp_num * 2 + hp_num + 3)
            if pitch_delta < 0:
                if yaw_delta < 0:
                    border_region = np.floor(np.vstack(
                        (contlist_ref[-1][:2, img_contour_tr: img_contour_bl + 1].T - 1,
                         border_bl, border_br, border_tr))).astype(np.int32)
                elif yaw_delta > 0:
                    border_region = np.floor(np.vstack(
                        (contlist_ref[-1][:2, img_contour_br:].T - 1,
                         contlist_ref[-1][:2, img_contour_tl].T - 1,
                         border_tl, border_bl, border_br))).astype(np.int32)
                else:
                    border_region = np.floor(np.vstack(
                        (contlist_ref[-1][:2, img_contour_br: img_contour_bl + 1].T - 1,
                         border_bl, border_br))).astype(np.int32)
            elif pitch_delta > 0:
                if yaw_delta < 0:
                    border_region = np.floor(np.vstack(
                        (contlist_ref[-1][:2, img_contour_tl: img_contour_br + 1].T - 1,
                         border_br, border_tr, border_tl))).astype(np.int32)
                elif yaw_delta > 0:
                    border_region = np.floor(np.vstack(
                        (contlist_ref[-1][:2, img_contour_bl:].T - 1,
                         contlist_ref[-1][:2, img_contour_tl: img_contour_tr + 1].T - 1,
                         border_tr, border_tl, border_bl))).astype(np.int32)
                else:
                    border_region = np.floor(np.vstack(
                        (contlist_ref[-1][:2, img_contour_tl: img_contour_tr + 1].T - 1,
                         border_tr, border_tl))).astype(np.int32)
            else:
                if yaw_delta < 0:
                    border_region = np.floor(np.vstack(
                        (contlist_ref[-1][:2, img_contour_tr: img_contour_br + 1].T - 1,
                         border_br, border_tr))).astype(np.int32)
                elif yaw_delta > 0:
                    border_region = np.floor(np.vstack(
                        (contlist_ref[-1][:2, img_contour_bl:].T - 1,
                         contlist_ref[-1][:2, img_contour_tl].T - 1,
                         border_tl, border_bl))).astype(np.int32)
            if border_region is not None:
                corres_map_x = np.ascontiguousarray(corres_map[..., 0])
                corres_map_y = np.ascontiguousarray(corres_map[..., 1])
                for offset in [(0, 0), (0, 1), (1, 0), (1, 1)]:
                    corres_map_x = cv2.fillPoly(corres_map_x, [border_region], -1.0, offset=offset)
                    corres_map_y = cv2.fillPoly(corres_map_y, [border_region], -1.0, offset=offset)
                corres_map[..., 0] = corres_map_x
                corres_map[..., 1] = corres_map_y
        if return_corres_map:
            maps_or_images.append(corres_map)
        else:
            maps_or_images.append(remap_image(image, corres_map))

        # get 3d style landmarks
        landmarks_3d = projected_vertex_ref[:, face_models['keypoints']]

        # get 2d style landmarks (by adjusting the contour points)
        landmarks_2d = landmarks_3d.copy()
        if yaw < 0:
            jaw_points_modify = list(range(8))
        else:
            jaw_points_modify = list(range(9, 17))
        jaw_points_ind = refine_contour_points(pitch_ref, yaw_ref, vertex, face_models['parallel_contour'],
                                               face_models['keypoints_contour'], jaw_points_modify)
        landmarks_2d[:, :len(jaw_points_ind)] = projected_vertex_ref[:, jaw_points_ind]

        # get extract landmarks
        if landmarks is not None:
            landmarks_exact = landmarks_3d.copy()
            for idx, (tri_idx, pt_barycentric) in matching_triangles.items():
                a = all_vertex_ref[:, all_tri[0, tri_idx]]
                b = all_vertex_ref[:, all_tri[1, tri_idx]]
                c = all_vertex_ref[:, all_tri[2, tri_idx]]
                landmarks_exact[:, idx] = (a * pt_barycentric[0] + b * pt_barycentric[1] + c * pt_barycentric[2])
            warped_landmarks.append(np.stack([landmarks_3d, landmarks_2d, landmarks_exact]))
        else:
            warped_landmarks.append(np.stack([landmarks_3d, landmarks_2d]))

    return np.stack(maps_or_images), np.stack(warped_landmarks)


def generate_profile_face(pitch_delta: float, yaw_delta: float, roll_delta: float,
                          fit_result: Dict, image: np.ndarray, face_models: Dict,
                          return_corres_map: bool = False, further_adjust_z: bool = False,
                          landmarks: Optional[np.ndarray] = None,
                          mouth_point_indices: Sequence[int] = range(48, 68)) -> Tuple[np.ndarray, np.ndarray]:
    map_or_image, warped_landmarks = generate_profile_faces(np.array([(pitch_delta, yaw_delta, roll_delta)]),
                                                            fit_result, image, face_models, return_corres_map,
                                                            further_adjust_z, landmarks, mouth_point_indices)
    return map_or_image[0], warped_landmarks[0]
