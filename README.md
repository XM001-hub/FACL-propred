# FACL-propred

##
This is the official code repository of the work "Fragment-aware contrastive learning framework for molecular property prediction".

![示意图](https://github.com/user-attachments/assets/afec4778-ebfa-49f0-bd74-05542e2a7ffe)



## Quick Start
**Installation**  
Follow the below steps for dependency installation.
```
git clone https://github.com/XM001-hub/FACL-propred.git
cd FACL-propred
conda env create -f environment.yml
```
**Data**  
The raw data used for pre-training and fine-tuning are under ./dataset folder.

***FGIB Training***  
```
cd torch_fgib
python train_fgib.py
```

**Dataset**  

During the data preparation phase, you could prepare your customized fragments database for contrastive pairs. You can use the following code to generate fragments in different scenarios (taking the radius of context set to 0 as an example):
```
python create_frag_env_db.py -smiles_path ‘/Path/to/smiles.csv’ -radius 0
```
After obtaining the fragment data, performing pre-computations on the ZINC database is necessary, as it can significantly accelerate the pre-training process. To this end, you can run the following code to calculate molecular contrastive pairs:
```
python precompute.py -smiles_path ‘/Path/to/smiles.csv’ -fragment_path ‘/Path/to/core_fragments.db’
```

***Pre-training***  
The configuration for pre-training, including model backbone, number of layers, and etc., can all be specified in `./config/facl_config.yml` file. After the configuration file is setup, simply run the following command for training.
```
python pretrain.py 
```
The pre-trained model obtained using this command will be saved in the output directory. Additionally, the pre-trained models we provide can be downloaded [here](https://drive.google.com/drive/folders/1n9RFJVajxdUglznCqaYT9s1cOp2IHlOg?usp=sharing)

***Fine-tuning***  
After obtaining the pre-trained model, we provide two methods for fine-tuning it for downstream tasks. Without modifying the FGIB module, you can run the following command to improve the model for a downstream dataset.
```
python finetune.py
```
If fine-tuning of the FGIB module is required, run the following command. Typically, this approach yields improved prediction performance.
```
Python dual_finetune.py
```




