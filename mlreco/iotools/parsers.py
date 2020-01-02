from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
from larcv import larcv
from mlreco.utils.ppn import get_ppn_info
from mlreco.utils.groups import filter_duplicate_voxels, filter_nonimg_voxels


def parse_sparse2d_scn(data):
    """
    A function to retrieve sparse tensor input from larcv::EventSparseTensor3D object
    Returns the data in format to pass to SCN
    Args:
        array of larcv::EventSparseTensor3D
    Return:
        voxels - numpy array(int32) with shape (N,3) - coordinates
        data   - numpy array(float32) with shape (N,C) - pixel values/channels
    """
    meta = None
    output = []
    np_voxels = None
    for event_tensor2d in data:

        tensor2d = event_tensor2d.sparse_tensor_2d(0)
        num_point = tensor2d.as_vector().size()

        if meta is None:

            meta = tensor2d.meta()
            np_voxels = np.empty(shape=(num_point, 2), dtype=np.int32)
            larcv.fill_2d_voxels(tensor2d, np_voxels)

        else:
            assert meta == tensor2d.meta()
        np_data = np.empty(shape=(num_point, 1), dtype=np.float32)
        larcv.fill_2d_pcloud(tensor2d, np_data)
        output.append(np_data)
    return np_voxels, np.concatenate(output, axis=-1)


def parse_sparse3d_scn(data):
    """
    A function to retrieve sparse tensor input from larcv::EventSparseTensor3D object
    Returns the data in format to pass to SCN
    Args:
        array of larcv::EventSparseTensor3D
    Return:
        voxels - numpy array(int32) with shape (N,3) - coordinates
        data   - numpy array(float32) with shape (N,C) - pixel values/channels
    """
    meta = None
    output = []
    np_voxels = None
    for event_tensor3d in data:
        num_point = event_tensor3d.as_vector().size()
        if meta is None:
            meta = event_tensor3d.meta()
            np_voxels = np.empty(shape=(num_point, 3), dtype=np.int32)
            larcv.fill_3d_voxels(event_tensor3d, np_voxels)
        else:
            assert meta == event_tensor3d.meta()
        np_data = np.empty(shape=(num_point, 1), dtype=np.float32)
        larcv.fill_3d_pcloud(event_tensor3d, np_data)
        output.append(np_data)
    return np_voxels, np.concatenate(output, axis=-1)


def parse_particle_points(data):
    """
    A function to retrieve particles ground truth points tensor, includes
    spatial coordinates and point type.
    Args:
        length 2 array of larcv::EventSparseTensor3D and larcv::EventParticle
    Return:
        a numpy array with the shape (N,3) where 3 represents (x,y,z)
        coordinate
        a numpy array with the shape (N, 1) where 1 represents class of
        the ground truth point.
    """
    particles_v = data[1].as_vector()
    part_info = get_ppn_info(particles_v, data[0].meta())
    if part_info.shape[0] > 0:
        return part_info[:, :3], part_info[:, 3][:, None]
    else:
        return np.empty(shape=(0, 3), dtype=np.int32), np.empty(shape=(0, 1), dtype=np.float32)


def parse_particle_infos(data):
    """
    A function to retrieve particles ground truth points tensor, includes
    spatial coordinates, point type and other infos (creation energy, pdg, etc)
    Args:
        length 2 array of larcv::EventSparseTensor3D and larcv::EventParticle
    Return:
        a numpy array with the shape (N,3) where 3 represents (x,y,z)
        coordinate
        a numpy array with the shape (N, C) where C represents class of
        the ground truth point + other infos.
    """
    particles_v = data[1].as_vector()
    part_info = get_ppn_info(particles_v, data[0].meta())
    if part_info.shape[0] > 0:
        return part_info[:, :3], part_info[:, 3:]
    else:
        return np.empty(shape=(0, 3), dtype=np.int32), np.empty(shape=(0, 6), dtype=np.float32)


def parse_cluster3d(data):
    """
    A function to retrieve clusters tensor
    Args:
        length 1 array of larcv::EventClusterVoxel3D
    Return:
        a numpy array with the shape (N,3) where 3 represents (x,y,z)
        coordinate
        a numpy array with the shape (N,1) where 1 is cluster id a
    """
    cluster_event = data[0]
    num_clusters = cluster_event.as_vector().size()
    clusters_voxels, clusters_data = [], []
    for i in range(num_clusters):
        cluster = cluster_event.as_vector()[i]
        num_points = cluster.as_vector().size()
        if num_points > 0:
            x = np.empty(shape=(num_points,), dtype=np.int32)
            y = np.empty(shape=(num_points,), dtype=np.int32)
            z = np.empty(shape=(num_points,), dtype=np.int32)
            value = np.empty(shape=(num_points,), dtype=np.int32)
            larcv.as_flat_arrays(cluster_event.as_vector()[i],
                                 cluster_event.meta(),
                                 x, y, z, value)
            value = np.full(shape=(cluster.as_vector().size(), 1),
                            fill_value=i, dtype=np.int32)
            clusters_voxels.append(np.stack([x, y, z], axis=1))
            clusters_data.append(value)
    np_voxels = np.concatenate(clusters_voxels, axis=0)
    np_data = np.concatenate(clusters_data, axis=0)
    return np_voxels, np_data


def parse_cluster3d_groups(data):
    """
    A function to retrieve clusters tensor
    Args:
        length 1 array of larcv::EventClusterVoxel3D
    Return:
        a numpy array with the shape (N,3) where 3 represents (x,y,z)
        coordinate
        a numpy array with the shape (N,1) where 1 is cluster id a
    """
    np_voxels, np_data = parse_cluster3d([data[0]])
    groups, edges = parse_particle_group([data[1]])
    for cluster_index, group_id in enumerate(groups):
        where = np.where(np_data == float(cluster_index))
        np_data[where] = group_id

    return np_voxels, np_data


def parse_sparse3d_clean(data):
    """
    A function to retrieve clusters tensor.  Do the following cleaning:
    1) lexicographically sort coordinates
    2) choose only one group per voxel (by lexicographic order)
    3) get labels from the image labels for each voxel in addition to groups

    Args:
        length 3 array of larcv::EventSparseTensor3D
        Typically [sparse3d_mcst_reco, sparse3d_mcst_reco_group, sparse3d_fivetypes_reco]
    Return:
        a numpy array with the shape (N,3) where 3 represents (x,y,z)
        coordinate
        a numpy array with the shape (N,3) where 3 is energy + cluster id + label
    """
    img_voxels, img_data = parse_sparse3d_scn([data[0]])
    perm = np.lexsort(img_voxels.T)
    img_voxels = img_voxels[perm]
    img_data = img_data[perm]
    img_voxels, unique_indices = np.unique(
        img_voxels, axis=0, return_index=True)
    img_data = img_data[unique_indices]

    grp_voxels, grp_data = parse_sparse3d_scn([data[1]])
    perm = np.lexsort(grp_voxels.T)
    grp_voxels = grp_voxels[perm]
    grp_data = grp_data[perm]
    grp_voxels, unique_indices = np.unique(
        grp_voxels, axis=0, return_index=True)
    grp_data = grp_data[unique_indices]

    label_voxels, label_data = parse_sparse3d_scn([data[2]])
    perm = np.lexsort(label_voxels.T)
    label_voxels = label_voxels[perm]
    label_data = label_data[perm]
    label_voxels, unique_indices = np.unique(
        label_voxels, axis=0, return_index=True)
    label_data = label_data[unique_indices]

    sel2 = filter_nonimg_voxels(grp_voxels, label_voxels[(
        label_data < 5).reshape((-1,)), :], usebatch=False)
    inds2 = np.where(sel2)[0]
    grp_voxels = grp_voxels[inds2]
    grp_data = grp_data[inds2]

    sel2 = filter_nonimg_voxels(img_voxels, label_voxels[(
        label_data < 5).reshape((-1,)), :], usebatch=False)
    inds2 = np.where(sel2)[0]
    img_voxels = img_voxels[inds2]
    img_data = img_data[inds2]
    return grp_voxels, np.concatenate([img_data, grp_data, label_data[label_data < 5][:, None]], axis=1)


def parse_cluster3d_clean(data):
    """
    A function to retrieve clusters tensor.  Do the following cleaning:
    1) lexicographically sort group data (images are lexicographically sorted)
    2) remove voxels from group data that are not in image
    3) choose only one group per voxel (by lexicographic order)

    Args:
        length 2 array of larcv::EventClusterVoxel3D and larcv::EventSparseTensor3D
    Return:
        a numpy array with the shape (N,3) where 3 represents (x,y,z)
        coordinate
        a numpy array with the shape (N,1) where 1 is cluster id
    """
    grp_voxels, grp_data = parse_cluster3d([data[0]])
    img_voxels, img_data = parse_sparse3d_scn([data[1]])

    # step 1: lexicographically sort group data
    perm = np.lexsort(grp_voxels.T)
    grp_voxels = grp_voxels[perm, :]
    grp_data = grp_data[perm]

    perm = np.lexsort(img_voxels.T)
    img_voxels = img_voxels[perm, :]
    img_data = img_data[perm]

    # step 2: remove duplicates
    sel1 = filter_duplicate_voxels(grp_voxels, usebatch=False)
    inds1 = np.where(sel1)[0]
    grp_voxels = grp_voxels[inds1, :]
    grp_data = grp_data[inds1]

    # step 3: remove voxels not in image
    sel2 = filter_nonimg_voxels(grp_voxels, img_voxels, usebatch=False)
    inds2 = np.where(sel2)[0]
    grp_voxels = grp_voxels[inds2, :]
    grp_data = grp_data[inds2]

    return grp_voxels, grp_data


def parse_particle_group(data):
    """
    A function to parse larcv::EventParticle to construct two information:
    1) grouping of particles (i.e. clusters)
    2) edges between particles (i.e. clusters)
    Args:
        length 1 array of larcv::EventParticle
    Return:
        a numpy array of group ID per particle (i.e. cluster), length = particle/cluster count.
        a numpy array of directed edges where each edge is (parent,child) cluster index ID.
    """
    particles = data[0]

    # for convention, construct particle id => cluster id mapping
    particle_to_cluster = np.zeros(
        shape=[particles.as_vector().size()], dtype=np.int32)
    # fill grouping of clusters (particles)
    group_ids = []
    groups = np.zeros(shape=[particles.as_vector().size()], dtype=np.int32)
    for cluster_id in range(particles.as_vector().size()):
        p = particles.as_vector()[cluster_id]
        particle_id = p.id()
        particle_to_cluster[particle_id] = cluster_id

        group_id = p.group_id()
        if not group_id in group_ids:
            group_ids.append(group_id)
        groups[cluster_id] = group_ids.index(group_id)
    # fill edges (directed, [parent,child] pair)
    edges = []
    for cluster_id in range(particles.as_vector().size()):
        p = particles.as_vector()[cluster_id]
        for child in p.children_id():
            edges.append([cluster_id, particle_to_cluster[child]])
    edges = np.array(edges).astype(np.int32)

    return groups, edges


def parse_particle_asis(data):
    """
    A function to copy construct & return an array of larcv::Particle
    Args:
        length 1 array of larcv::EventParticle
    Return:
        a python list of larcv::Particle object
    """
    particles = data[0]
    clusters = data[1]
    assert particles.as_vector().size() in [
        clusters.as_vector().size(), clusters.as_vector().size() - 1]

    meta = clusters.meta()

    particles = [larcv.Particle(p) for p in data[0].as_vector()]
    funcs = ["first_step", "last_step", "position", "end_position"]
    for p in particles:
        for f in funcs:
            pos = getattr(p, f)()
            x = (pos.x() - meta.min_x()) / meta.size_voxel_x()
            y = (pos.y() - meta.min_y()) / meta.size_voxel_y()
            z = (pos.z() - meta.min_z()) / meta.size_voxel_z()
            getattr(p, f)(x, y, z, pos.t())
    return particles


def parse_cluster3d_scales(data):
    """
    Retrieves clusters tensors at different spatial sizes.
    Parameters
    ----------
    data: list
        length 2 array of larcv::EventClusterVoxel3D and larcv::EventSparseTensor3D
    Returns
    -------
    list of tuples
    """
    grp_voxels, grp_data = parse_cluster3d_clean(data)
    spatial_size = data[0].meta().num_voxel_x()
    max_depth = int(np.floor(np.log2(spatial_size)) - 1)
    scales = []
    for d in range(max_depth):
        scale_voxels = np.floor(grp_voxels / 2**d)  # .astype(int)
        scale_voxels, unique_indices = np.unique(
            scale_voxels, axis=0, return_index=True)
        scale_data = grp_data[unique_indices]
        scales.append((scale_voxels, scale_data))
    return scales


def parse_sparse3d_scn_scales(data):
    """
    Retrieves sparse tensors at different spatial sizes.
    Parameters
    ----------
    data: list
        length 1 array of larcv::EventSparseTensor3D
    Returns
    -------
    list of tuples
    """
    grp_voxels, grp_data = parse_sparse3d_scn(data)
    perm = np.lexsort(grp_voxels.T)
    grp_voxels = grp_voxels[perm]
    grp_data = grp_data[perm]

    spatial_size = data[0].meta().num_voxel_x()
    max_depth = int(np.floor(np.log2(spatial_size)) - 1)
    scales = []
    for d in range(max_depth):
        scale_voxels = np.floor(grp_voxels / 2**d)  # .astype(int)
        scale_voxels, unique_indices = np.unique(
            scale_voxels, axis=0, return_index=True)
        scale_data = grp_data[unique_indices]
        # perm = np.lexsort(scale_voxels.T)
        # scale_voxels = scale_voxels[perm]
        # scale_data = scale_data[perm]
        scales.append((scale_voxels, scale_data))
    return scales
