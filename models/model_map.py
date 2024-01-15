from models.transformer.transformer_predictor import TransformerPredictor
from models.transformer.vit import ViT

model_by_model_name = {'ReversePredictor': TransformerPredictor, 'ViT': ViT}
