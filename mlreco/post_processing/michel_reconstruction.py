import numpy as np
import os
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import cdist
from mlreco.utils.misc import CSVData

def michel_reconstruction(cfg, data_blob, res, logdir, iteration):
    """
    Very simple algorithm to reconstruct Michel clusters from UResNet semantic
    segmentation output.

    Parameters
    ----------
    data_blob: dict
        Input dictionary returned by iotools
    res: dict
        Results from the network, dictionary using `analysis_keys`
    cfg: dict
        Configuration
    idx: int
        Iteration number

    Notes
    -----
    Assumes 3D

    Input
    -----
    Requires the following analysis keys:
    - `segmentation` output of UResNet
    - `ghost` predictions of GhostNet
    Requires the following input keys:
    - `input_data`
    - `segment_label`
    - `particles_label` to get detailed information such as energy.
    - `clusters_label` from `cluster3d_mcst` for true clusters informations

    Output
    ------
    Writes 2 CSV files:
    - `michel_reconstruction-*`
    - `michel_reconstruction2-*`
    """
    method_cfg = cfg['post_processing']['michel_reconstruction']

    # Create output CSV
    store_per_iteration = True
    if method_cfg is not None and method_cfg.get('store_method',None) is not None:
        assert(method_cfg['store_method'] in ['per-iteration','per-event'])
        store_per_iteration = method_cfg['store_method'] == 'per-iteration'

    fout_reco,fout_true=None,None
    if store_per_iteration:
        fout_reco=CSVData(os.path.join(logdir, 'michel-reconstruction-reco-iter-%07d.csv' % iteration))
        fout_true=CSVData(os.path.join(logdir, 'michel-reconstruction-true-iter-%07d.csv' % iteration))

    # Loop over events
    for batch_id,data in enumerate(data_blob['input_data']):

        event_idx = data_blob['index'          ][batch_id]

        if not store_per_iteration:
            fout_reco=CSVData(os.path.join(logdir, 'michel-reconstruction-reco-event-%07d.csv' % event_idx))
            fout_true=CSVData(os.path.join(logdir, 'michel-reconstruction-true-event-%07d.csv' % event_idx))

        # from input/labels
        label       = data_blob['segment_label'  ][batch_id][:,-1]
        clusters    = data_blob['clusters_label' ][batch_id]
        particles   = data_blob['particles_label'][batch_id]
        Michel_particles = particles[particles[:, 4] == 4]

        # from network output
        segmentation = res['segmentation'][batch_id]
        predictions  = np.argmax(segmentation,axis=1)
        ghost_mask   = (np.argmax(res['ghost'][batch_id],axis=1) == 0)

        data_pred    = data[ghost_mask]  # coords
        label_pred   = label[ghost_mask]  # labels
        predictions  = (np.argmax(segmentation,axis=1))[ghost_mask]
        segmentation = segmentation[ghost_mask]


        # 0. Retrieve coordinates of true and predicted Michels
        # MIP_coords = data[(label == 1).reshape((-1,)), ...][:, :3]
        # Michel_coords = data[(label == 4).reshape((-1,)), ...][:, :3]
        MIP_coords = clusters[clusters[:, -1] == 1][:, :3]
        Michel_coords = clusters[clusters[:, -1] == 4][:, :3]
        if Michel_coords.shape[0] == 0:  # FIXME
            continue
        MIP_coords_pred = data_pred[(predictions == 1).reshape((-1,)), ...][:, :3]
        Michel_coords_pred = data_pred[(predictions == 4).reshape((-1,)), ...][:, :3]

        one_pixel = 20#2.8284271247461903
        # 1. Find true particle information matching the true Michel cluster
        # Michel_true_clusters = DBSCAN(eps=one_pixel, min_samples=5).fit(Michel_coords).labels_
        # Michel_true_clusters = [Michel_coords[Michel_coords[:, -2] == gid] for gid in np.unique(Michel_coords[:, -2])]
        Michel_true_clusters = clusters[clusters[:, -1] == 4][:, -2].astype(np.int64)
        Michel_start = Michel_particles[:, :3]
        for cluster in np.unique(Michel_true_clusters):
            # print("True", np.count_nonzero(Michel_true_clusters == cluster))
            # TODO sum_pix
            fout_true.record(('batch_id', 'iteration', 'event_idx', 'num_pix', 'sum_pix'),
                             (batch_id, iteration, event_idx,
                              np.count_nonzero(Michel_true_clusters == cluster),
                              clusters[clusters[:, -1] == 4][Michel_true_clusters == cluster][:, -3].sum()))
            fout_true.write()
        # e.g. deposited energy, creation energy
        # TODO retrieve particles information
        # if Michel_coords.shape[0] > 0:
        #     Michel_clusters_id = np.unique(Michel_true_clusters[Michel_true_clusters>-1])
        #     for Michel_id in Michel_clusters_id:
        #         current_index = Michel_true_clusters == Michel_id
        #         distances = cdist(Michel_coords[current_index], MIP_coords)
        #         is_attached = np.min(distances) < 2.8284271247461903
        #         # Match to MC Michel
        #         distances2 = cdist(Michel_coords[current_index], Michel_start)
        #         closest_mc = np.argmin(distances2, axis=1)
        #         closest_mc_id = closest_mc[np.bincount(closest_mc).argmax()]

        # TODO how do we count events where there are no predictions but true?
        if MIP_coords_pred.shape[0] == 0 or Michel_coords_pred.shape[0] == 0:
            continue
        # print("Also predicted!")
        # 2. Compute true and predicted clusters
        MIP_clusters = DBSCAN(eps=one_pixel, min_samples=10).fit(MIP_coords_pred).labels_
        Michel_pred_clusters = DBSCAN(eps=one_pixel, min_samples=5).fit(Michel_coords_pred).labels_
        Michel_pred_clusters_id = np.unique(Michel_pred_clusters[Michel_pred_clusters>-1])
        # print(len(Michel_pred_clusters_id))
        # Loop over predicted Michel clusters
        for Michel_id in Michel_pred_clusters_id:
            current_index = Michel_pred_clusters == Michel_id
            # 3. Check whether predicted Michel is attached to a predicted MIP
            # and at the edge of the predicted MIP
            distances = cdist(Michel_coords_pred[current_index], MIP_coords_pred[MIP_clusters>-1])
            # is_attached = np.min(distances) < 2.8284271247461903
            is_attached = np.min(distances) < 5
            is_edge = False  # default
            # print("Min distance:", np.min(distances))
            if is_attached:
                Michel_min, MIP_min = np.unravel_index(np.argmin(distances), distances.shape)
                MIP_id = MIP_clusters[MIP_clusters>-1][MIP_min]
                MIP_min_coords = MIP_coords_pred[MIP_clusters>-1][MIP_min]
                MIP_cluster_coords = MIP_coords_pred[MIP_clusters==MIP_id]
                ablated_cluster = MIP_cluster_coords[np.linalg.norm(MIP_cluster_coords-MIP_min_coords, axis=1)>15.0]
                if ablated_cluster.shape[0] > 0:
                    new_cluster = DBSCAN(eps=one_pixel, min_samples=5).fit(ablated_cluster).labels_
                    is_edge = len(np.unique(new_cluster[new_cluster>-1])) == 1
                else:
                    is_edge = True
            # print(is_attached, is_edge)

            michel_pred_num_pix_true, michel_pred_sum_pix_true = -1, -1
            michel_true_num_pix, michel_true_sum_pix = -1, -1
            michel_true_energy = -1
            if is_attached and is_edge and Michel_coords.shape[0] > 0:
                # Distance from current Michel pred cluster to all true points
                distances = cdist(Michel_coords_pred[current_index], Michel_coords)
                closest_clusters = Michel_true_clusters[np.argmin(distances, axis=1)]
                closest_clusters_final = closest_clusters[(closest_clusters > -1) & (np.min(distances, axis=1)<one_pixel)]
                if len(closest_clusters_final) > 0:
                    # print(closest_clusters_final, np.bincount(closest_clusters_final), np.bincount(closest_clusters_final).argmax())
                    # cluster id of closest true Michel cluster
                    # we take the one that has most overlap
                    # closest_true_id = closest_clusters_final[np.bincount(closest_clusters_final).argmax()]
                    closest_true_id = np.bincount(closest_clusters_final).argmax()
                    overlap_pixels_index = (closest_clusters == closest_true_id) & (np.min(distances, axis=1)<one_pixel)
                    if closest_true_id > -1:
                        closest_true_index = label_pred[predictions==4][current_index]==4
                        # Intersection
                        michel_pred_num_pix_true = 0
                        michel_pred_sum_pix_true = 0.
                        for v in data_pred[(predictions==4).reshape((-1,)), ...][current_index]:
                            count = int(np.any(np.all(v[:3] == Michel_coords[Michel_true_clusters == closest_true_id], axis=1)))
                            michel_pred_num_pix_true += count
                            if count > 0:
                                michel_pred_sum_pix_true += v[-1]

                        michel_true_num_pix = np.count_nonzero(Michel_true_clusters == closest_true_id)
                        michel_true_sum_pix = clusters[clusters[:, -1] == 4][Michel_true_clusters == closest_true_id][:, -3].sum()
                        # Register true energy
                        # Match to MC Michel
                        distances2 = cdist(Michel_coords[Michel_true_clusters == closest_true_id], Michel_start)
                        closest_mc = np.argmin(distances2, axis=1)
                        closest_mc_id = closest_mc[np.bincount(closest_mc).argmax()]
                        michel_true_energy = Michel_particles[closest_mc_id, 7]
            # Record every predicted Michel cluster in CSV
            fout_reco.record(('batch_id', 'iteration', 'event_idx', 'pred_num_pix', 'pred_sum_pix',
                              'pred_num_pix_true', 'pred_sum_pix_true',
                              'true_num_pix', 'true_sum_pix',
                              'is_attached', 'is_edge', 'michel_true_energy'),
                             (batch_id, iteration, event_idx, np.count_nonzero(current_index),
                              data_pred[(predictions==4).reshape((-1,)), ...][current_index][:, -1].sum(),
                              michel_pred_num_pix_true, michel_pred_sum_pix_true, michel_true_num_pix, michel_true_sum_pix,
                              is_attached, is_edge, michel_true_energy))
            fout_reco.write()

        if not store_per_iteration:
            fout_reco.close()
            fout_true.close()

    if store_per_iteration:
        fout_reco.close()
        fout_true.close()
