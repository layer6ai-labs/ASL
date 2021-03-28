from models.asl_model import ASLModel

class ModelFactory:

    @staticmethod
    def get_model(model_name, config):
        if model_name == "ThumosModel":
            return ASLModel(config.len_feature, config.num_classes, config.num_segments)
        else:
            raise NotImplementedError("No such model")