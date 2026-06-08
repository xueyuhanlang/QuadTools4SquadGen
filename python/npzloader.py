import numpy as np
import point_cloud_utils as pcu  # used for debugging and texturing
import os  # used for debugging only
import argparse
# the following import is used for texturing
import trimesh
import xatlas
from PIL import Image
import matplotlib.pyplot as plt

def sample_point_on_edges(border_edge_ids, edge_length, edge_id, subdiv_vertex, num_samples):
    total_length = edge_length[border_edge_ids].sum()
    unique_vertices = np.unique(edge_id[border_edge_ids].reshape(-1))
    border_vertex = subdiv_vertex[unique_vertices]

    if border_vertex.shape[0] >= num_samples:
        return border_vertex[np.random.choice(border_vertex.shape[0], num_samples, replace=False)]

    target_num = num_samples - border_vertex.shape[0]
    sample_rate = target_num / total_length
    sampled_points = []

    for eid in border_edge_ids:
        edge = edge_id[eid]
        num = int(np.ceil(edge_length[eid] * sample_rate))
        if num > 0:
            points = subdiv_vertex[edge[0]] + (subdiv_vertex[edge[1]] - subdiv_vertex[edge[0]]) * np.linspace(0, 1, num + 2)[1:-1, None]
            sampled_points.append(points)

    sampled_points = np.concatenate(sampled_points, axis=0)
    np.random.shuffle(sampled_points)

    if sampled_points.shape[0] > target_num:
        sampled_points = sampled_points[:target_num]

    return np.concatenate([border_vertex, sampled_points], axis=0)


def sample_point_around_edges(
    border_edge_ids,
    edge_length,
    edge_id,
    edge_neighbor_faceids,
    edge_opposite_vertex_ids,
    subdiv_vertex,
    num_samples,
    no_num_limit: bool = False,
):
    total_border_edge_length = edge_length[border_edge_ids].sum()
    border_sample_rate = num_samples / total_border_edge_length
    border_sample_points, border_sample_face_ids = [], []
    border_vertex_point, border_vertex_point_face_ids = [], []
    shift_scale, edge_point_shift_scale = 1.0e-5, 1.0e-5

    for eid in border_edge_ids:
        edge = edge_id[eid]
        opposite_vertex_ids = edge_opposite_vertex_ids[eid]
        neighbor_face_ids = edge_neighbor_faceids[eid]
        edge_mid_point = (subdiv_vertex[edge[0]] + subdiv_vertex[edge[1]]) * 0.5
        shift_opposite_vertex_points = subdiv_vertex[opposite_vertex_ids] + shift_scale * (edge_mid_point - subdiv_vertex[opposite_vertex_ids])
        num = max(2, int(np.ceil(edge_length[eid] * border_sample_rate)))
        even_num = num + (num % 2)
        shift_opposite_vertex_point_array = np.tile(shift_opposite_vertex_points, (even_num, 1))[:num]
        neighbor_face_id_array = np.tile(neighbor_face_ids, (even_num, 1)).reshape(-1)[:num]

        edge_sample_points = subdiv_vertex[edge[0]] + (subdiv_vertex[edge[1]] - subdiv_vertex[edge[0]]) * np.linspace(1 / (2 * num), 1 - 1 / (2 * num), num=num)[:, None]
        edge_sample_points += edge_point_shift_scale * (shift_opposite_vertex_point_array - edge_sample_points)
        border_sample_points.append(edge_sample_points[1:-1])
        border_sample_face_ids.append(neighbor_face_id_array[1:-1])
        border_vertex_point.append(edge_sample_points[[0, -1]])
        border_vertex_point_face_ids.append(neighbor_face_id_array[[0, -1]])

    border_sample_points = np.concatenate(border_sample_points, axis=0)
    border_sample_face_ids = np.concatenate(border_sample_face_ids, axis=0)
    border_vertex_point = np.concatenate(border_vertex_point, axis=0)
    border_vertex_point_face_ids = np.concatenate(border_vertex_point_face_ids, axis=0)

    if no_num_limit:
        return (
            np.concatenate([border_vertex_point, border_sample_points], axis=0),
            np.concatenate([border_vertex_point_face_ids, border_sample_face_ids], axis=0),
        )

    if border_vertex_point.shape[0] >= num_samples:
        idx = np.random.permutation(border_vertex_point.shape[0])[:num_samples]
        return border_vertex_point[idx], border_vertex_point_face_ids[idx]

    target_num = num_samples - border_vertex_point.shape[0]
    idx = np.random.permutation(border_sample_points.shape[0])[:target_num]
    return (
        np.concatenate([border_vertex_point, border_sample_points[idx]], axis=0),
        np.concatenate([border_vertex_point_face_ids, border_sample_face_ids[idx]], axis=0),
    )


def get_patch_grading_colors(sample_points, sample_point_tri_id, cur_edge_dir, next_edge_dir, v0, v1, denom, indices, idx1, idx2, cur_edge_dir_color, next_edge_dir_color, tri_normals):
    def compute_patch_grading_t(diff, next_edge_dir, denom, indices, idx1, idx2, cur_edge_dir_color, tri_normals):
        selected_denom = denom[np.arange(diff.shape[0]), indices]
        t = np.clip(np.cross(diff, next_edge_dir)[np.arange(diff.shape[0]), indices] / selected_denom, 0, 1)
        color = cur_edge_dir_color[:, 0] + t * (cur_edge_dir_color[:, 1] - cur_edge_dir_color[:, 0])
        grad = np.zeros_like(diff)
        grad[np.arange(len(indices)), idx1] = next_edge_dir[np.arange(len(indices)), idx2]
        grad[np.arange(len(indices)), idx2] = -next_edge_dir[np.arange(len(indices)), idx1]
        grad = ((cur_edge_dir_color[:, 1] - cur_edge_dir_color[:, 0]) / selected_denom)[:, None] * grad
        grad = grad - np.sum(grad * tri_normals, axis=-1)[:, None] * tri_normals
        return color, grad

    diff_v0 = sample_points - v0[sample_point_tri_id]
    diff_v1 = sample_points - v1[sample_point_tri_id]
    cur_edge_dir_ = cur_edge_dir[sample_point_tri_id]
    next_edge_dir_ = next_edge_dir[sample_point_tri_id]
    denom_ = denom[sample_point_tri_id]
    indices_ = indices[sample_point_tri_id]
    idx1_ = idx1[sample_point_tri_id]
    idx2_ = idx2[sample_point_tri_id]
    cur_edge_dir_color_ = cur_edge_dir_color[sample_point_tri_id]
    next_edge_dir_color_ = next_edge_dir_color[sample_point_tri_id]
    tri_normals_ = tri_normals[sample_point_tri_id]
    color_v0, grad_v0 = compute_patch_grading_t(
        diff_v0,
        next_edge_dir_,
        denom_,
        indices_,
        idx1_,
        idx2_,
        cur_edge_dir_color_,
        tri_normals_,
    )

    color_v1, grad_v1 = compute_patch_grading_t(
        diff_v1,
        cur_edge_dir_,
        -denom_,
        indices_,
        idx1_,
        idx2_,
        next_edge_dir_color_,
        tri_normals_,
    )
    
    # normalize the gradients
    norm_v0 = np.linalg.norm(grad_v0, axis=1, keepdims=True) + 1e-10
    norm_v1 = np.linalg.norm(grad_v1, axis=1, keepdims=True) + 1e-10
    grad_v0 /= norm_v0
    grad_v1 /= norm_v1

    mask = color_v1 < color_v0
    color_dcdf = np.clip(np.where(mask, color_v1, color_v0), 0, 1)
    grad_dcdf = np.where(mask[:, None], grad_v1, grad_v0)

    mask = 1- color_v1 < 1 - color_v0
    color_cdf = np.clip(np.where(mask, 1 - color_v1, 1 - color_v0), 0, 1)
    grad_cdf = np.where(mask[:, None], -grad_v1, -grad_v0)

    return color_dcdf, grad_dcdf, color_cdf, grad_cdf, color_v0, grad_v0, color_v1, grad_v1 


def save_lines_as_obj(border_edge_ids, edge_id, subdiv_vertex, obj_file_name):
    count = 0
    border_edges = open(obj_file_name, "w")
    for i in range(border_edge_ids.shape[0]):
        eid = border_edge_ids[i]
        border_edges.write(
            "v %f %f %f\n"
            % (
                subdiv_vertex[edge_id[eid, 0], 0],
                subdiv_vertex[edge_id[eid, 0], 1],
                subdiv_vertex[edge_id[eid, 0], 2],
            )
        )
        border_edges.write(
            "v %f %f %f\n"
            % (
                subdiv_vertex[edge_id[eid, 1], 0],
                subdiv_vertex[edge_id[eid, 1], 1],
                subdiv_vertex[edge_id[eid, 1], 2],
            )
        )
        border_edges.write(f"l {count + 1} {count + 2}\n")
        count += 2

def load_patch_data_reformat(
    npzfile: str,
    num_checker_border_samples: int = 2048,
    num_sharp_feature_sample: int = 1024,
    sharp_feature_degree: float = 120,
    loopsubdivision: bool = False,
    debug: bool = False,
    output_texture: bool = False,
    textured_obj_file_name: str = None,
    resolution: int = 1024,
    colorscheme: str = 'turbo', # you can use plt color names, or None (gray color)
    color_pattern: str = "dcdf", # "dcdf", "cdf", "both"; for texturing
    div: int = 0, # use for subdividing the color pattern
):
    """
    checkerboard npz loader
    """
    # Load the npz file
    npzdata = np.load(npzfile)
    if npzdata is None:
        raise ValueError(f"npz file {npzfile} not found")

    # the checkboard_tag is used for distinguishing different types of data that I developed for SQuadGen. now 2 is the default tag.
    # 0: non-checkerboard, 1: checkerboard, 2: patchboard
    checkerboard_tag = npzdata["is_checkerboard"].squeeze()
    if checkerboard_tag != 2:
        raise ValueError(f"npz file {npzfile} is not a patchboard")

    # transformation
    invT = npzdata["invT"]  # inverse transformation matrix (4 x 3)

    # Original mesh info
    # vertex_num  = npzdata["quad_mesh_vertex_num"].squeeze()  # number of vertices
    # quad_facet = npzdata["quad_facet"]  # mesh facet indices (nf x 4)

    # SUBDIV MESH INFO
    subdiv_vertex = npzdata["subdiv_vertex"]  # mesh vertices (snv x 3)
    subdiv_facet = npzdata["subdiv_facet"]  # mesh facet indices (snf x 4)
    quad_split = npzdata["quad_split"].squeeze().astype(bool)  # how the quad is splitted into triangles facet indices (snf)

    # PATCH and QUAD INFO
    face2quad = npzdata["face2quad"].squeeze()  # quad face to quad patch id (snf)
    quad2patch = npzdata["quad2patch"].squeeze()  # quad patch id to checker id (nq)

    # OFFSET INFO
    offset_abcd_id = npzdata["offset_abcd_id"]  # offset vertex id of quad patch (nq x 4)
    offset_123_id = npzdata["offset_123_id"]  # offset vertex id of quad patch (pointing other three checkers) (nq x 3)

    # Edge eolor info
    edge_color_store = npzdata["edge_color_store"]  # edge color info (ne x 2, float)
    edge_vertex_color = npzdata["edge_vertex_color"].squeeze()  # edge color info (ne, int)

    ########
    # If you want to add noise on vertex positions, add your function here
    # vertex_purturbation(subdiv_vertex)
    ########

    # TRIANGLE MESH INFO
    tri_faces = subdiv_facet[np.arange(subdiv_facet.shape[0])[:, None], [0, 1, 2, 2, 3, 0]]
    tri_faces_tmp = subdiv_facet[np.arange(subdiv_facet.shape[0])[:, None], [3, 0, 1, 1, 2, 3]]
    tri_faces[quad_split] = tri_faces_tmp[quad_split]
    tri_faces = tri_faces.reshape(-1, 3)  # triangle mesh faces (tnf x 3)

    # OFFSET INFO: index pointing to the subdivided mesh
    offset_abcd = subdiv_vertex[offset_abcd_id]
    offset_123 = subdiv_vertex[offset_123_id]

    # TRIMESH EDGE INFO
    if num_checker_border_samples > 0 or num_sharp_feature_sample > 0:
        edge_neighbor_face_ids = npzdata["edge_faceids"]  # edge face ids (ne x 2)
        edge_info = npzdata["edge_info"]  # edge info (ne x 4)
        edge_checkerborder_tag = edge_info[:, -1]  # edge checker border tag (ne x 1)

        edge_id = np.stack([subdiv_facet[edge_neighbor_face_ids[:, 0] // 2, edge_info[:, 0]], subdiv_facet[edge_neighbor_face_ids[:, 0] // 2, (edge_info[:, 0] + 1) % 4]], axis=1)
        edge_opposite_vertex_ids = np.stack([tri_faces[edge_neighbor_face_ids[:, 0], edge_info[:, 1]], tri_faces[edge_neighbor_face_ids[:, 1], edge_info[:, 2]]], axis=1)  # edge opposite vert ids (ne x 2)

    #apply loopsubdivision
    if loopsubdivision:
        sorted_tri_faces = np.sort(tri_faces, axis=1)
        tri_edges = np.concat([sorted_tri_faces[:, [0, 1]], sorted_tri_faces[:, [1, 2]], sorted_tri_faces[:, [0, 2]]])
        tri_oppsite_vertex_ids = np.concat([sorted_tri_faces[:, 2], sorted_tri_faces[:, 0], sorted_tri_faces[:, 1]])
        unique_edges, unique_indices, unique_inverse = np.unique(tri_edges, axis=0, return_index=True, return_inverse=True)
        tri_edge_opposite_vertex_ids = -np.ones((unique_edges.shape[0], 2), dtype=np.int32)
        tri_edge_opposite_vertex_ids[:, 0] = tri_oppsite_vertex_ids[unique_indices]
        myindex = unique_inverse != np.arange(tri_edges.shape[0])
        tri_edge_opposite_vertex_ids[unique_inverse[myindex], 1] = tri_oppsite_vertex_ids[myindex]
        boundary_edge_tag = tri_edge_opposite_vertex_ids[:, 1] == -1
        tri_edge_opposite_vertex_ids[boundary_edge_tag, 1] = tri_edge_opposite_vertex_ids[boundary_edge_tag, 0]       
        boundary_vertex_tag = np.zeros(subdiv_vertex.shape[0], dtype=bool)
        boundary_vertex_tag[unique_edges[boundary_edge_tag].flatten()] = True
        vertex_degrees = np.bincount(unique_edges.flatten(), minlength=subdiv_vertex.shape[0])
        vertex_neighbor_sum = np.zeros_like(subdiv_vertex)
        edge_v0 = unique_edges[:, 0].reshape(-1)
        edge_v1 = unique_edges[:, 1].reshape(-1)
        tag = (boundary_vertex_tag[edge_v0] & boundary_vertex_tag[edge_v1]) ^ (~boundary_vertex_tag[edge_v0] & ~boundary_vertex_tag[edge_v1])
        np.add.at(vertex_neighbor_sum, edge_v0[tag], subdiv_vertex[edge_v1[tag]])
        np.add.at(vertex_neighbor_sum, edge_v1[tag], subdiv_vertex[edge_v0[tag]])
        tag = (boundary_vertex_tag[edge_v0] & ~boundary_vertex_tag[edge_v1])
        np.add.at(vertex_neighbor_sum, edge_v1[tag], subdiv_vertex[edge_v0[tag]])
        tag = (~boundary_vertex_tag[edge_v0] & boundary_vertex_tag[edge_v1])
        np.add.at(vertex_neighbor_sum, edge_v0[tag], subdiv_vertex[edge_v1[tag]])
        
        edge_split_vertex = np.where(
            boundary_edge_tag[:, None],
            (subdiv_vertex[unique_edges[:, 0]] + subdiv_vertex[unique_edges[:, 1]]) / 2,
            (3.0 / 8.0) * (subdiv_vertex[unique_edges[:, 0]] + subdiv_vertex[unique_edges[:, 1]]) +
            (1.0 / 8.0) * (subdiv_vertex[tri_edge_opposite_vertex_ids[:, 0]] + subdiv_vertex[tri_edge_opposite_vertex_ids[:, 1]])
        )
        
        vertex_neighbor_sum /= vertex_degrees[:, None] 
        subdiv_vertex[boundary_vertex_tag] = 0.25 * vertex_neighbor_sum[boundary_vertex_tag] + 0.75 * subdiv_vertex[boundary_vertex_tag]


        interior_degree_3_tag = (~boundary_vertex_tag) & (vertex_degrees == 3)
        interior_degree_other_tag = (~boundary_vertex_tag) & (vertex_degrees != 3)
        subdiv_vertex[interior_degree_3_tag] = (9.0 / 16.0) * vertex_neighbor_sum[interior_degree_3_tag] + (7.0 / 16.0) * subdiv_vertex[interior_degree_3_tag]

        beta = (3 / 8 + 0.25 * np.cos(2 * np.pi / vertex_degrees[interior_degree_other_tag])) ** 2
        alpha = 0.625 - beta
        calpha = 0.375 + beta
        subdiv_vertex[interior_degree_other_tag] = calpha[:,None]  * subdiv_vertex[interior_degree_other_tag] + alpha[:,None] * vertex_neighbor_sum[interior_degree_other_tag]
        # loopoints = np.concat([edge_split_vertex, subdiv_vertex], axis=0).reshape(-1, 3)
        # pcu.save_mesh_v("loop_subdivision.ply", v=loopoints)  #debug

    # compute tri_normals, quad_sizing, checker_sizing
    tri_normal = np.cross(subdiv_vertex[tri_faces[:, 1]] - subdiv_vertex[tri_faces[:, 0]], subdiv_vertex[tri_faces[:, 2]] - subdiv_vertex[tri_faces[:, 0]])
    tri_area = np.linalg.norm(tri_normal, axis=1, keepdims=True) + 1e-12
    tri_normal = tri_normal / tri_area
    tri_area = tri_area.squeeze()

    tri_to_quad_map = (np.arange(tri_faces.shape[0], dtype=np.int32) // 2).astype(np.int32)
    tri_to_QUAD = face2quad[tri_to_quad_map]
    QUAD_num = np.max(tri_to_QUAD) + 1
    QUAD_area = np.zeros(QUAD_num)
    np.add.at(QUAD_area, tri_to_QUAD, tri_area / 2)
    quad_sizing = np.sqrt(QUAD_area)  # size of QUADs
    check_num = np.max(quad2patch) + 1
    checker_area = np.zeros(check_num)
    np.add.at(checker_area, quad2patch, QUAD_area)
    checker_sizing = np.sqrt(checker_area)  # size of checkers


    #compute quad face directions
    subdiv_quad_faces = subdiv_vertex[subdiv_facet]  # quad faces (snf, 4, 3)
    subdiv_quad_dirs = np.stack([subdiv_quad_faces[:, 1] - subdiv_quad_faces[:, 0], subdiv_quad_faces[:, 2] - subdiv_quad_faces[:, 1], subdiv_quad_faces[:, 3] - subdiv_quad_faces[:, 2], subdiv_quad_faces[:, 0] - subdiv_quad_faces[:, 3]], axis=1)  # quad face directions (snf, 4, 3)
    subdiv_quad_dirs = subdiv_quad_dirs / (np.linalg.norm(subdiv_quad_dirs, axis=-1, keepdims=True) + 1e-10)  # quad face directions (snf, 4, 3)
    quad_dir_len_0123 = np.sum(np.cross(subdiv_quad_dirs[:, 0], subdiv_quad_dirs[:, 2], axis=1)**2, axis=1, keepdims=True)  # quad face directions length (snf)
    quad_dir_len_1230 = np.sum(np.cross(subdiv_quad_dirs[:, 1], subdiv_quad_dirs[:, 3], axis=1)**2,axis=1, keepdims=True)# quad face directions length (snf)
    quad_dir = np.where(quad_dir_len_0123 < quad_dir_len_1230, subdiv_quad_dirs[:, 0]-subdiv_quad_dirs[:, 2], subdiv_quad_dirs[:, 1] - subdiv_quad_dirs[:, 3])  # quad face directions (snf, 3)
    quad_dir = np.repeat(quad_dir, 2, axis=0).reshape(-1, 3)  # quad face directions (2*snf, 3)
    dir_check = np.cross(tri_normal, quad_dir, axis=1)
    quad_dir = np.where(dir_check[:, 2, None] > 0, quad_dir, -quad_dir)  
    quad_dir = quad_dir - np.sum(quad_dir* tri_normal, axis=-1)[:, None] * tri_normal 
    quad_dir = quad_dir / (np.linalg.norm(quad_dir, axis=1, keepdims=True) + 1e-10) 

    # prepare interpolation info

    tri_faces_edge_id = np.tile([0, 1, 2, 3], (subdiv_facet.shape[0], 1))
    tri_faces_edge_id[quad_split] = np.tile([3, 0, 1, 2], (quad_split.sum(), 1))
    tri_faces_edge_id = tri_faces_edge_id.reshape(-1, 2)
    tri_faces_edge_id = np.stack([4 * (np.arange(tri_faces_edge_id.shape[0]) // 2) + tri_faces_edge_id[:, 0], 4 * (np.arange(tri_faces_edge_id.shape[0]) // 2) + tri_faces_edge_id[:, 1]], axis=1)
    tri_faces_edge_color_id = edge_vertex_color[tri_faces_edge_id.reshape(-1)].reshape(tri_faces_edge_id.shape[0], 2)

    # Compute absolute values of edge color IDs for indexing
    tri_faces_edge_color_id_abs = np.abs(tri_faces_edge_color_id)
    # Current edge direction color
    cur_edge_dir_color = edge_color_store[tri_faces_edge_color_id_abs[:, 0]]
    negative_mask_cur = tri_faces_edge_color_id[:, 0] < 0
    cur_edge_dir_color[negative_mask_cur] = cur_edge_dir_color[negative_mask_cur][:, [1, 0]]
    # Next edge direction color
    next_edge_dir_color = edge_color_store[tri_faces_edge_color_id_abs[:, 1]]
    negative_mask_next = tri_faces_edge_color_id[:, 1] < 0
    next_edge_dir_color[negative_mask_next] = next_edge_dir_color[negative_mask_next][:, [1, 0]]

    interpolation_v0 = subdiv_vertex[tri_faces[:, 0]]
    interpolation_v1 = subdiv_vertex[tri_faces[:, 1]]
    cur_edge_dir = interpolation_v1 - interpolation_v0
    next_edge_dir = subdiv_vertex[tri_faces[:, 2]] - interpolation_v1

    denom = np.cross(cur_edge_dir, next_edge_dir)
    indices = np.argmax(np.abs(denom), axis=1)
    cross_indices = [(1, 2), (2, 0), (0, 1)]
    idx1, idx2 = np.array([cross_indices[i] for i in indices]).T



    # SAMPLE POINTS
    sample_bary_coords = npzdata["sample_bary_coords"]  ## barycentric coordinates (N x 3)
    sample_point_tri_id = npzdata["sample_point_tri_id"].squeeze()  ## triangle id (N)
    # compute sample_points using barycentric coordinates
    sample_points = np.einsum("ij,ijk->ik", sample_bary_coords, subdiv_vertex[tri_faces][sample_point_tri_id])
    sample_point_quad_id = tri_to_QUAD[sample_point_tri_id].astype(np.int32)
    sample_point_quad_size = quad_sizing[sample_point_quad_id]
    sample_point_normal = tri_normal[sample_point_tri_id]

    sample_point_dcdf_color_grading, sample_point_dcdf_color_gradient, sample_point_cdf_color_grading, sample_point_cdf_color_gradient, sample_point_xvalue, sample_point_xgradient, sample_point_yvalue, sample_point_ygradient = get_patch_grading_colors(sample_points, sample_point_tri_id, cur_edge_dir, next_edge_dir, interpolation_v0, interpolation_v1, denom, indices, idx1, idx2, cur_edge_dir_color, next_edge_dir_color, tri_normal)
    sample_point_offset_abcd = offset_abcd[sample_point_quad_id] - sample_points[:, None, :]
    sample_point_offset_123 = offset_123[sample_point_quad_id] - sample_points[:, None, :]
    sample_point_direction = quad_dir[sample_point_tri_id]
    # FPS POINTS
    fps_num_list = [int(k.split("_")[-1]) for k in npzdata.keys() if k.startswith("fps_bary_coords_")]
    fps_dict = {}
    for fps_num in fps_num_list:
        fps_bary_coords = npzdata[f"fps_bary_coords_{fps_num}"]  ## barycentric coordinates of fps points (M x nfps x 3)
        fps_point_tri_id = npzdata[f"fps_point_tri_id_{fps_num}"].squeeze()  ## triangle id of fps points (M x nfps)
        if fps_bary_coords.ndim == 2:
            fps_bary_coords = fps_bary_coords[None, :, :]
        if fps_bary_coords.ndim == 1:
            fps_bary_coords = fps_bary_coords[None, :]

        fps_points = np.einsum("ijkl,ijk->ijl", subdiv_vertex[tri_faces][fps_point_tri_id], fps_bary_coords)
        fps_point_quad_id = tri_to_QUAD[fps_point_tri_id].astype(np.int32)
        fps_point_normal = tri_normal[fps_point_tri_id]
        fps_point_grading_dcdf_color, fps_point_grading_dcdf_color_gradient, fps_point_grading_cdf_color, fps_point_grading_cdf_color_gradient, fps_point_grading_xvalue, fps_point_grading_xgradient, fps_point_grading_yvalue, fps_point_grading_ygradient = get_patch_grading_colors(fps_points.reshape(-1, 3), fps_point_tri_id.reshape(-1), cur_edge_dir, next_edge_dir, interpolation_v0, interpolation_v1, denom, indices, idx1, idx2, cur_edge_dir_color, next_edge_dir_color, tri_normal)

        fps_point_grading_dcdf_color = fps_point_grading_dcdf_color.reshape(fps_point_tri_id.shape[0], fps_point_tri_id.shape[1])
        fps_point_grading_dcdf_color_gradient = fps_point_grading_dcdf_color_gradient.reshape(fps_point_tri_id.shape[0], fps_point_tri_id.shape[1], 3)
        fps_point_grading_cdf_color = fps_point_grading_cdf_color.reshape(fps_point_tri_id.shape[0], fps_point_tri_id.shape[1])
        fps_point_grading_cdf_color_gradient = fps_point_grading_cdf_color_gradient.reshape(fps_point_tri_id.shape[0], fps_point_tri_id.shape[1], 3)
        fps_point_grading_xvalue = fps_point_grading_xvalue.reshape(fps_point_tri_id.shape[0], fps_point_tri_id.shape[1])
        fps_point_grading_xgradient = fps_point_grading_xgradient.reshape(fps_point_tri_id.shape[0], fps_point_tri_id.shape[1], 3)
        fps_point_grading_yvalue = fps_point_grading_yvalue.reshape(fps_point_tri_id.shape[0], fps_point_tri_id.shape[1])
        fps_point_grading_ygradient = fps_point_grading_ygradient.reshape(fps_point_tri_id.shape[0], fps_point_tri_id.shape[1], 3)

        fps_point_offset_abcd = offset_abcd[fps_point_quad_id] - fps_points[:, :, None, :]
        fps_point_offset_123 = offset_123[fps_point_quad_id] - fps_points[:, :, None, :]

        fps_dict.update(
            {
                f"xyz_fps_{fps_num}": fps_points,
                f"normal_fps_{fps_num}": fps_point_normal,
                f"color_grading_fps_{fps_num}": fps_point_grading_dcdf_color[..., None],
                f"color_gradient_fps_{fps_num}": fps_point_grading_dcdf_color_gradient,
                f"color_cdf_fps_{fps_num}": fps_point_grading_cdf_color[..., None],
                f"color_gradient_cdf_fps_{fps_num}": fps_point_grading_cdf_color_gradient,
                f"xvalue_fps_{fps_num}": fps_point_grading_xvalue[..., None],
                f"xgradient_fps_{fps_num}": fps_point_grading_xgradient,
                f"yvalue_fps_{fps_num}": fps_point_grading_yvalue[..., None],
                f"ygradient_fps_{fps_num}": fps_point_grading_ygradient,
                f"offset_abcd_fps_{fps_num}": fps_point_offset_abcd,
                f"offset_123_fps_{fps_num}": fps_point_offset_123,
            }
        )
    # Sample points on the border of the checkerboard and sharp feature edges
    if num_checker_border_samples > 0 or num_sharp_feature_sample > 0:
        edge_dir = subdiv_vertex[edge_id[:, 0]] - subdiv_vertex[edge_id[:, 1]]
        edge_length = np.linalg.norm(edge_dir, axis=1)
        edge_dir /= edge_length[:, None] + 1e-10

    # sample points on the border of the checkerboard
    border_sample_points, border_sample_face_ids = None, None
    if num_checker_border_samples > 0:
        border_edge_ids = np.where(edge_checkerborder_tag == 1)[0]
        border_sample_points, border_sample_face_ids = sample_point_around_edges(border_edge_ids, edge_length, edge_id, edge_neighbor_face_ids, edge_opposite_vertex_ids, subdiv_vertex, num_checker_border_samples)
        border_point_quad_id = tri_to_QUAD[border_sample_face_ids].astype(np.int32)
        border_point_offset_abcd = offset_abcd[border_point_quad_id] - border_sample_points[:, None, :]
        border_point_offset_123 = offset_123[border_point_quad_id] - border_sample_points[:, None, :]
        border_sample_dcdf_color_grading, border_sample_dcdf_color_gradient, border_sample_cdf_color_grading, border_sample_cdf_color_gradient, border_sample_xvalue, border_sample_xgradient, border_sample_yvalue, border_sample_ygradient = get_patch_grading_colors(border_sample_points, border_sample_face_ids, cur_edge_dir, next_edge_dir, interpolation_v0, interpolation_v1, denom, indices, idx1, idx2, cur_edge_dir_color, next_edge_dir_color, tri_normal)

    # sample points on the sharp feature edges
    sharp_sample_points, sharp_sample_face_ids = None, None
    if num_sharp_feature_sample > 0:
        face_centroid = np.mean(subdiv_vertex[tri_faces], axis=1)
        b1 = face_centroid[edge_neighbor_face_ids[:, 0]] - subdiv_vertex[edge_id[:, 1]]
        b2 = face_centroid[edge_neighbor_face_ids[:, 1]] - subdiv_vertex[edge_id[:, 1]]
        b1 -= np.einsum("ij,ij->i", b1, edge_dir)[:, None] * edge_dir
        b2 -= np.einsum("ij,ij->i", b2, edge_dir)[:, None] * edge_dir
        b1 /= np.linalg.norm(b1, axis=1, keepdims=True) + 1e-10
        b2 /= np.linalg.norm(b2, axis=1, keepdims=True) + 1e-10
        dihedral_angle = np.degrees(np.arccos(np.clip(np.einsum("ij,ij->i", b1, b2), -1, 1)))
        sharp_edge_ids = np.where(dihedral_angle < sharp_feature_degree)[0]
        if sharp_edge_ids.shape[0] > 0:
            # sharp_sample_points = sample_point_on_edges(sharp_edge_ids,edge_length,edge_id,subdiv_vertex,num_sharp_feature_sample)
            sharp_sample_points, sharp_sample_face_ids = sample_point_around_edges(sharp_edge_ids, edge_length, edge_id, edge_neighbor_face_ids, edge_opposite_vertex_ids, subdiv_vertex, num_sharp_feature_sample)
            sharp_point_quad_id = tri_to_QUAD[sharp_sample_face_ids].astype(np.int32)
            sharp_point_offset_abcd = offset_abcd[sharp_point_quad_id] - sharp_sample_points[:, None, :]
            sharp_point_offset_123 = offset_123[sharp_point_quad_id] - sharp_sample_points[:, None, :]
            sharp_sample_dcdf_color_grading, sharp_sample_dcdf_color_gradient, sharp_sample_cdf_color_grading, sharp_sample_cdf_color_gradient, sharp_sample_xvalue, sharp_sample_xgradient,sharp_sample_yalue, sharp_sample_ygradient = get_patch_grading_colors(sharp_sample_points, sharp_sample_face_ids, cur_edge_dir, next_edge_dir, interpolation_v0, interpolation_v1, denom, indices, idx1, idx2, cur_edge_dir_color, next_edge_dir_color, tri_normal)

    # verification
    if debug:
        if not os.path.exists("data"):
            os.makedirs("data")

        pcu.save_mesh_vf("data/trimesh.ply", v=subdiv_vertex, f=tri_faces)

        ptscolor = np.ones((sample_points.shape[0], 4))
        ptscolor[:, 0:3] = sample_point_dcdf_color_grading[:, None]
        pcu.save_mesh_vnc("data/sample_point_grading.ply", v=sample_points, n=sample_point_normal, c=ptscolor)
        pcu.save_mesh_vnc("data/sample_point_dcdf_colorgradient.ply", v=sample_points, n=sample_point_dcdf_color_gradient, c=ptscolor)
        pcu.save_mesh_vnc("data/sample_point_direction.ply", v=sample_points, n=sample_point_direction, c=ptscolor)

        for fps_num in fps_num_list:
            fpscolor = np.ones((fps_num, 4))
            for i in range(fps_dict[f"xyz_fps_{fps_num}"].shape[0]):
                fpscolor[:, 0:3] = fps_dict[f"color_grading_fps_{fps_num}"][i]
                pcu.save_mesh_vnc(f"data/fps_points_grading_{fps_num}_{i}.ply", v=fps_dict[f"xyz_fps_{fps_num}"][i], n=fps_dict[f"normal_fps_{fps_num}"][i], c=fpscolor)

        if num_checker_border_samples > 0:
            if border_sample_face_ids is not None:
                border_sample_dcdf_colors = np.ones((border_sample_points.shape[0], 4))
                border_sample_dcdf_colors[:, 0:3] = border_sample_dcdf_color_grading[:, None]
                pcu.save_mesh_vnc("data/border_sample_points_grading.ply", v=border_sample_points, n=tri_normal[border_sample_face_ids], c=border_sample_dcdf_colors)
            else:
                if border_sample_points is not None:
                    pcu.save_mesh_v("./border_sample_points.ply", v=border_sample_points)

        if num_sharp_feature_sample > 0:
            if sharp_sample_face_ids is not None:
                sharp_sample_dcdf_colors = np.ones((sharp_sample_points.shape[0], 4))
                sharp_sample_dcdf_colors[:, 0:3] = sharp_sample_dcdf_color_grading[:, None]
                pcu.save_mesh_vnc("data/sharp_sample_points_grading.ply", v=sharp_sample_points, n=tri_normal[sharp_sample_face_ids], c=sharp_sample_dcdf_colors)
            else:
                if sharp_sample_points is not None:
                    pcu.save_mesh_v("data/sharp_sample_points.ply", v=sharp_sample_points)

        if num_checker_border_samples > 0:
            save_lines_as_obj(border_edge_ids, edge_id, subdiv_vertex, "data/border_edges.obj")
        if num_sharp_feature_sample > 0:
            save_lines_as_obj(sharp_edge_ids, edge_id, subdiv_vertex, "data/sharp_feature_edges.obj")

        count = 0
        abcdlines = open("data/abcd.obj", "w")
        for i in range(offset_abcd.shape[0]):
            for j in range(4):
                abcdlines.write("v %f %f %f\n" % (offset_abcd[i, j, 0], offset_abcd[i, j, 1], offset_abcd[i, j, 2]))
            abcdlines.write(f"l {count + 1} {count + 2}\n")
            abcdlines.write(f"l {count + 2} {count + 3}\n")
            abcdlines.write(f"l {count + 3} {count + 4}\n")
            abcdlines.write(f"l {count + 4} {count + 1}\n")
            count += 4

        count = 0
        offset_123lines = open("data/123.obj", "w")
        for i in range(offset_123.shape[0]):
            offset_123lines.write("v %f %f %f\n" % (offset_abcd[i, 0, 0], offset_abcd[i, 0, 1], offset_abcd[i, 0, 2]))
            for j in range(3):
                offset_123lines.write("v %f %f %f\n" % (offset_123[i, j, 0], offset_123[i, j, 1], offset_123[i, j, 2]))

            offset_123lines.write(f"l {count + 1} {count + 2}\n")
            offset_123lines.write(f"l {count + 2} {count + 3}\n")
            offset_123lines.write(f"l {count + 3} {count + 4}\n")
            offset_123lines.write(f"l {count + 4} {count + 1}\n")
            count += 4

    if output_texture:
        print("Data loading and reformatting completed.")

        #get texture foldername
        texture_folder = os.path.dirname(textured_obj_file_name)
        print(f"Texture folder: {texture_folder}")
        if texture_folder != "" and not os.path.exists(texture_folder):
            os.makedirs(texture_folder, exist_ok=True)   
            print(f"Created texture folder: {texture_folder}")     

        mesh = trimesh.Trimesh(vertices=subdiv_vertex, faces=tri_faces, vertex_normals=tri_normal)
        mesh_v, mesh_f = mesh.vertices, mesh.faces
        if textured_obj_file_name is None:
            textured_obj_file_name = npzfile.split("/")[-1].replace(".npz", ".obj")
        mtl_file_name = textured_obj_file_name.split('/')[-1].replace(".obj", ".mtl")
        texture_file_name = textured_obj_file_name.split('/')[-1].replace(".obj", ".png")
        with open(textured_obj_file_name.replace(".obj", ".mtl"), 'w') as f:
            f.write("newmtl quad_material\n")
            f.write("map_Kd %s\n" % texture_file_name)

        #parametrize the mesh
        vmapping, paraindices, uvs = xatlas.parametrize(mesh_v, mesh_f)


        #write the mesh to obj file
        with open(textured_obj_file_name, 'w') as f:
            f.write("# QUAD Texture\n")
            f.write("mtllib %s\n" % mtl_file_name)
            f.write("usemtl quad_material\n")
            for v in mesh_v[vmapping]:
                f.write("v %f %f %f\n" % (v[0], v[1], v[2]))
            for uv in uvs:
                f.write("vt %f %f\n" % (uv[0], uv[1]))
            for i in paraindices:
                f.write("f %d/%d %d/%d %d/%d\n" % (i[0]+1, i[0]+1, i[1]+1, i[1]+1, i[2]+1, i[2]+1))

        #construct grid points
        x = np.linspace(0, 1, resolution)
        xx,  yy = np.meshgrid(x, x, indexing='xy')
        grid_points = np.stack([xx, yy, np.ones_like(xx, dtype=np.float32)], axis=-1).reshape(-1,3)
        uvs_points = np.append(uvs, np.zeros((uvs.shape[0], 1)), axis=1)
        d, fi, bc = pcu.closest_points_on_mesh(grid_points, uvs_points, paraindices)
        mask = np.all(~np.isnan(bc), axis=1) & (d < 1+1.0e-5)
        projection_points = pcu.interpolate_barycentric_coords(paraindices, fi[mask], bc[mask], mesh_v[vmapping])
        # color_dcdf, grad_dcdf, color_cdf, grad_cdf, color_v0, grad_v0, color_v1, grad_v1 
        color_dcdf, _, color_cdf, _, _, _, _, _ = get_patch_grading_colors(projection_points, fi[mask], cur_edge_dir, next_edge_dir, interpolation_v0, interpolation_v1, denom, indices, idx1, idx2, cur_edge_dir_color, next_edge_dir_color, tri_normal)
        if color_pattern == "dcdf":
            projection_point_colors = color_dcdf
            if div > 0:
                u, v = 1 - color_cdf, color_dcdf
                u_new = np.abs(u * div - (np.floor(u * div + 0.5)))
                v_new = np.abs(v * div - (np.floor(v * div + 0.5)))
                projection_point_colors = 1 - 2 * np.where(u_new > v_new, u_new, v_new)
        elif color_pattern == "cdf":
            projection_point_colors = color_cdf
            if div > 0:
                u, v = 1 - color_cdf, color_dcdf
                u_new = np.abs(u - (np.floor(u * div) + 0.5) / div)
                v_new = np.abs(v - (np.floor(v * div) + 0.5) / div)
                projection_point_colors = 1 - 2 * div * np.where(u_new > v_new, u_new, v_new)
        elif color_pattern == "uvtex":    # this experimental setting does not work for now        
            cres = max(div, 2) #checkerboard resolution > 1
            # threshold = 0.05 #threshold to determine the color
            scaled_u = (1 - color_cdf) * cres
            scaled_v = color_dcdf * cres
            C = np.mod((np.floor(scaled_u) + np.floor(scaled_v)), 2)
            projection_point_colors = np.where(C, np.zeros_like(color_cdf), np.ones_like(color_cdf))
        else:
            print(color_dcdf.shape, color_cdf.shape)
            color_dcdf_flat = color_dcdf.flatten()
            color_cdf_flat = color_cdf.flatten()
            mask_assemble = (color_dcdf_flat <= 0.05) | (color_cdf_flat <= 0.05)
            projection_point_colors = np.where(mask_assemble, np.zeros_like(color_dcdf_flat), np.ones_like(color_dcdf_flat))
        if colorscheme is not None and color_pattern != "both" and color_pattern != "uvtex":
            cmap = plt.get_cmap(colorscheme)
            projection_point_colors = cmap(projection_point_colors)[:, :3] * 255
        else:
            projection_point_colors = (projection_point_colors * 255).astype(np.uint8)
            projection_point_colors = np.repeat(projection_point_colors, 3).reshape(-1, 3)

        texture = np.zeros((resolution, resolution, 3), dtype=np.uint8).reshape(-1, 3)
        texture[mask] = projection_point_colors
        texture = texture.reshape(resolution, resolution, 3)
        texture = np.flipud(texture)
        img = Image.fromarray(texture)
        texture_file_name = textured_obj_file_name.replace(".obj", ".png")
        img.save(texture_file_name) 

        #create a trimesh with the parameterization
        para_mesh = trimesh.Trimesh(vertices=mesh_v[vmapping], faces=paraindices, vertex_normals=mesh.vertex_normals[vmapping], visual=trimesh.visual.TextureVisuals(uv=uvs, image=img))
        # para_mesh.export(textured_obj_file_name.replace(".obj", ".ply"))
        para_mesh.export(textured_obj_file_name.replace(".obj", ".glb"))

    return {
        "xyz": sample_points,
        "normal": sample_point_normal,
        "dcdf_color_grading": sample_point_dcdf_color_grading[..., None],
        "dcdf_color_gradient": sample_point_dcdf_color_gradient,
        "cdf_color_grading": sample_point_cdf_color_grading[..., None],
        "cdf_color_gradient": sample_point_cdf_color_gradient,
        "xvalue": sample_point_xvalue[..., None],
        "xgradient": sample_point_xgradient,
        "yvalue": sample_point_yvalue[..., None],
        "ygradient": sample_point_ygradient,
        "offset_abcd": sample_point_offset_abcd,
        "offset_123": sample_point_offset_123,
        "sample_point_quad_size": sample_point_quad_size[..., None],
        "sample_point_direction": sample_point_direction,
        "quadsize_quad": quad_sizing[..., None],
        "checker_sizing": checker_sizing,
        "quadsize_mean": np.array([np.mean(quad_sizing)]),
        **fps_dict,
        # border_sample_points: border_sample_points can be None
        # sharp_sample_points: sharp_sample_points can be None
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Load NPZ data and reformat sampled patch features.",
    )
    parser.add_argument("npzfile", type=str, help="Input .npz file path")
    parser.add_argument("--num-checker-border-samples", type=int, default=2048)
    parser.add_argument("--num-sharp-feature-sample", type=int, default=1024)
    parser.add_argument("--sharp-feature-degree", type=float, default=120.0)
    parser.add_argument("--loopsubdivision", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--output-texture", action="store_true")
    parser.add_argument("--textured-obj-file-name", type=str, default=None)
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--colorscheme", type=str, default="turbo")
    parser.add_argument(
        "--color-pattern",
        type=str,
        default="cdf",
        choices=["dcdf", "cdf"],
    )
    parser.add_argument("--div", type=int, default=0)
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    print("Loading and reformatting patch data from:", args.npzfile)
    out = load_patch_data_reformat(
        npzfile=args.npzfile,
        num_checker_border_samples=args.num_checker_border_samples,
        num_sharp_feature_sample=args.num_sharp_feature_sample,
        sharp_feature_degree=args.sharp_feature_degree,
        loopsubdivision=args.loopsubdivision,
        debug=args.debug,
        output_texture=args.output_texture,
        textured_obj_file_name=args.textured_obj_file_name,
        resolution=args.resolution,
        colorscheme=args.colorscheme,
        color_pattern=args.color_pattern,
        div=args.div,
    )
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

