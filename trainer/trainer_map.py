from trainer.transformer.reverse_trainer import ReverseTrainer
from trainer.transformer.vit_trainer import ViTTrainer

model_trainer_by_model_name = {
    'ReversePredictor': ReverseTrainer,
    'ViT': ViTTrainer
}
