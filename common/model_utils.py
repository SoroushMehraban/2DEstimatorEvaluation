try:
    from common.model_poseformer import PoseTransformerV2
except ModuleNotFoundError:
    from model_poseformer import PoseTransformerV2


def count_number_of_parameters(model):
    """
    Counts number of parameters that model has
    
    Args:
        model (torch.nn.Module): PyTorch model

    returns:
        model_params (int): Number of parameters
    """
    model_params = 0
    for parameter in model.parameters():
        model_params += parameter.numel()
    return model_params