def model_dict():

    # Models
    from . import uresnet_chain
    from . import acnn_chain
    from . import uresnext_chain
    from . import deeplab_chain
    from . import uresnet_lovasz
    from . import fpn_chain

    # Losses
    from mlreco.nn.loss.segmentation import SegmentationLoss

    models = {
        # URESNET CHAIN
        "uresnet_chain": (uresnet_chain.UResNet_Chain, uresnet_chain.SegmentationLoss),
        "uresnet_lovasz": (uresnet_lovasz.UResNet_Chain, SegmentationLoss), 
        "fpn_chain": (fpn_chain.FPN_Chain, SegmentationLoss), 
        "acnn_chain": (acnn_chain.ACNN_Chain, acnn_chain.SegmentationLoss),
        "uresnext_chain": (uresnext_chain.UResNeXt_Chain, uresnet_chain.SegmentationLoss),
        "deeplab_chain": (deeplab_chain.DeepLab_Chain, uresnet_chain.SegmentationLoss)
    }
    return models

def construct(name):
    models = model_dict()
    if name not in models:
        raise Exception("Unknown model name provided")
    return models[name]
