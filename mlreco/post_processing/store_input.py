import os
from mlreco.utils.misc import CSVData

def store_input(cfg, data_blob, res, logdir, iteration):
    # 0 = Event voxels and values
    if not 'input_data' in data_blob: return

    method_cfg = cfg['post_processing']['store_input']

    threshold = 0. if method_cfg is None else method_cfg.get('threshold',0.)

    index      = data_blob.get('index',None)
    input_dat  = data_blob.get('input_data',None)
    label_ppn  = data_blob.get('particles_label',None)
    label_seg  = data_blob.get('segment_label',None)
    label_cls  = data_blob.get('clusters_label',None)
    label_mcst = data_blob.get('cluster3d_mcst_true',None)

    store_per_iteration = True
    if method_cfg is not None and method_cfg.get('store_method',None) is not None:
        assert(method_cfg['store_method'] in ['per-iteration','per-event'])
        store_per_iteration = method_cfg['store_method'] == 'per-iteration'
    fout=None
    if store_per_iteration:
        fout=CSVData(os.path.join(logdir, 'input-iter-%07d.csv' % iteration))

    if input_dat is None: return

    for data_index,tree_index in enumerate(index):

        if not store_per_iteration:
            fout=CSVData(os.path.join(logdir, 'input-event-%07d.csv' % tree_index))

        mask = input_dat[data_index][:,-1] > threshold

        # type 0 = input data
        for row in input_dat[data_index][mask]:
            fout.record(('idx','x','y','z','type','value'),(tree_index,row[0],row[1],row[2],0,row[4]))
            fout.write()

        # type 1 = Labels for PPN
        if label_ppn is not None:
            for row in label_ppn[data_index]:
                fout.record(('idx','x','y','z','type','value'),(tree_index,row[0],row[1],row[2],1,row[4]))
                fout.write()
        # 2 = UResNet labels
        if label_seg is not None:
            for row in label_seg[data_index][mask]:
                fout.record(('idx','x','y','z','type','value'),(tree_index,row[0],row[1],row[2],2,row[4]))
                fout.write()
        # type 15 = group id, 16 = semantic labels, 17 = energy
        if label_cls is not None:
            for row in label_cls[data_index]:
                fout.record(('idx','x','y','z','type','value'),(tree_index,row[0],row[1],row[2],15,row[5]))
                fout.write()
            for row in label_cls[data_index]:
                fout.record(('idx','x','y','z','type','value'),(tree_index,row[0],row[1],row[2],16,row[6]))
                fout.write()
            for row in label_cls[data_index]:
                fout.record(('idx','x','y','z','type','value'),(tree_index,row[0],row[1],row[2],17,row[4]))
                fout.write()
        # type 18 = cluster3d_mcst_true
        if label_mcst is not None:
            for row in label_mcst[data_index]:
                fout.record(('idx','x','y','z','type','value'),(tree_index,row[0],row[1],row[2],19,row[4]))
                fout.write()

        if not store_per_iteration: fout.close()

    if store_per_iteration: fout.close()
