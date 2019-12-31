def model_dict():

    from . import uresnet_chain

    models = {
        # URESNET CHAIN
        "uresnet_chain": (uresnet_chain.UResNet_Chain, uresnet_chain.SegmentationLoss)
    }
    return models

def construct(name):
    models = model_dict()
    if name not in models:
        raise Exception("Unknown model name provided")
    return models[name]
