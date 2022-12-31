# NLPDL-final_project
This repository is for the final project of 2022-2023NLPDL Course.
The final project I choose is domain specific nlp task , and the domain I choose is computer science.
---
# Environment set up
using requirement.txt to set up environment.
### using pip
pip install -r requirements.txt

### using Conda
conda create --name <env_name> --file requirements.txt

# Data Too Big , Download From Google Drive
Download data from google drive.https://drive.google.com/drive/folders/12DScuaFbr8adLvCXx1dHbUYR7eu3sI9g?usp=share_link
and put data and the files under the same repository
## data.py

This file is an api for getting dataset

---



## train.py

the main file for training and finetuning

----



## tokenizer.py

For training a new tokenizer

---



### Example : 

1. For post training
   Just type "python train.py" then we can read the result in wandb

2. For training new tokenizer:

   Just type "python tokenizer.py"

3. For finetuning:

   change **data_collator** in **train.py** into **DataCollatorWithPadding** ,and type "python train.py"

   

   

   

   

   






