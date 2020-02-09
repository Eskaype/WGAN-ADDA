# Multi-Source
-
To run multi source domain adaptation using WGAN 

python3 trainer_dual_source/madan.py 

Change the dataset below mask paths in 
datastes/glaucoma_dataset.py 

MASK_PATHS = {"origa": "/storage/zwang/datasets/origa/",
              "drishti":"/storage/zwang/datasets/drishti/Disc_Cup_Masks/",
              "refuge": "/storage/zwang/datasets/refuge/cropped/Disc_Cup_Masks/"}
