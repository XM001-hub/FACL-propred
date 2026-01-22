# FACL-propred

##
This is the official code repository of the work "Fragment-aware contrastive learning framework for molecular property prediction".

## Quick Start
**Installation**  
Follow the below steps for dependency installation.
```
git clone https://github.com/XM001-hub/FACL-propred.git
cd FACL-propred
conda env create -f environment.yml
```
**Data**  
The raw data used for pre-training and fine-tuning are under ./dataset folder. You can also download the processed data from here.  
***FGIB Training***  
```
cd torch_fgib
python train_fgib.py
```

**Precompute**  
You can also prepare your customized dataset for pretraining.
```
python precompute.py
```


