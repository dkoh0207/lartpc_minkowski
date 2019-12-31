def model_dict():

    from . import uresnet_chain
    from . import acnn_chain

    models = {
        # URESNET CHAIN
        "uresnet_chain": (uresnet_chain.UResNet_Chain, uresnet_chain.SegmentationLoss),
        "acnn_chain": (acnn_chain.ACNN_Chain, acnn_chain.SegmentationLoss)
    }
    return models

def construct(name):
    models = model_dict()
    if name not in models:
        raise Exception("Unknown model name provided")
    return models[name]
