"""
Example: Training from an UNZIPPED folder (zip=False) with rtdc_toolbox

Training set is in TRAINING_SET_DIR

Expected training set folder structure
  D:/TrainingSet/
    RBC/
      img001.png
      ...
    platelets/
      img010.png
      ...
    WBC/
      img020.png
      ...

The script trains a model based on cnn backbones defined in CNN_BACKBONES

Training results are saved in TRAINING_RESULTS_DIR
"""

from rtdc_toolbox import train_model
import os

CNN_BACKBONES= ["convnext_tiny"] # the following are not so good "resnet18","resnet50","efficientnet_b0"

TRAINING_SET_DIR=r"D:\Naiad\Training\TrainingSet.ZIP"
TRAINING_RESULTS_DIR=r"D:\Naiad\Training\TrainingResults"

def main():
    print("Starting training from folder...", flush=True)
    
    for cnn_backbone in CNN_BACKBONES:
        model_path = train_model(
            # Folder that contains class subfolders
            path_to_training_set=TRAINING_SET_DIR,
            training_results_dir=os.path.join(TRAINING_RESULTS_DIR, cnn_backbone),
            model_name=cnn_backbone,         
    
            # Backbone (torchvision only)
            cnn_backbone=cnn_backbone,   # resnet18, resnet50, efficientnet_b0, convnext_tiny
            pretrained=True, #if pretrained weights from torchvision are used - 
    
            # Image extension inside class folders
            img_ext=".png",
    
            # Training schedule
            n_epochs=7,
            W0=160,
            val_split=0.2, #the fraction of the training set used for validation
    
            # Performance
            zip=True,            # training set can be stored in a zip file, but perfomance is better if it is not
            batch_size=256,
            cache_in_ram=True,   # can be True if dataset is not too large
            num_workers=6,        # if cache_in_ram=True, cache will be duplicated for each worker 
            
    
            # Optimizer
            lr=3e-4,
            weight_decay=1e-4,
    
            # Augmentations (OFF by default)
            brightness_jitter=0.02,
            contrast_jitter=0.05,
    
            # if your training set is very inbalanced (eg you have more examples of RBCs than WBCs)    
            # Imbalance handling (robust default) 
            # imbalance_cfg=ImbalanceConfig(strategy="weighted_sampler", sampler_power=1.0),
    
            # Reproducibility
            seed=123,
        )
    
        print("Saved model:", model_path, flush=True)


if __name__ == "__main__":
    main()
