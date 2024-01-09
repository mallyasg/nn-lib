from datasets.reverse_dataset import rev_train_loader, rev_val_loader, rev_test_loader

dataset_by_model_name = {
    'ReversePredictor': {
        'train': rev_train_loader,
        'val': rev_val_loader,
        'test': rev_test_loader
    }
}
