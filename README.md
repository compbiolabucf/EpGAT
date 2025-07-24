# EpGAT
This repository represents an attention-based model named EpGAT for predicting and quantifying post-transcriptional regulatory events using HiC and epigenetic data.


## Workflow
![alt text](https://github.com/compbiolabucf/EpGAT/blob/main/EpGAT_overall.png)

## Dataset
The data directory before strating the training process, should look like following:
- data
    - <cell_line1_name>
      - as_pred_data (containing quantified AS events for each chromosome)
      - apa_pred_data (containing quantified APA events for each chromosome)
      - eps_data (containing the epigenetics data for each chromosome)
    - <cell_line2_name>
      - as_pred_data
      - apa_pred_data 
      - eps_data
    - common_eps.csv (list of the common epigenetic signals' names)
    - <cell_line1_name>.hic ('HiC' file for cell_line1)
    - <cell_line2_name>.hic ('HiC' file for cell_line2)

The python scripts inside [this folder](preprocessing_data) folder can be used to prepare the data for application of the model.


## Training and testing the model
All the parts of EpGAT model can be trained simultaneously using the python script- 'main.py'. The type of experiment being conducted- cross cell line or cross chromosome should be provided as an argument. Train and test cell lines or chromosomes to be experimented with, should be specified. A number of other options can be also utilized to modify the model parameters.
Example command for cross cell line experiment:
```
python main.py --exp_type cross_cell_line --train_cell_line MCF7 --test_cell_line K562 --chromosome 1 --event as --root_directory data
```
Example command for cross chromosome experiment:
```
python main.py --exp_type cross_chromosome --cell_line MCF7 --train_chrs 1 2 3 4 5 --val_chrs 6 13 --test_chrs 7 14 --event apa --root_directory data
```
