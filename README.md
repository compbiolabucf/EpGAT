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

## Training and testing the model
All the parts of EpGAT model can be trained simultaneously using the python script- 'main.py'. The type of experiment being conducted- cross cell line or cross chromosome should be provided as an argument. Train and test cell lines or chromosomes to be experimented with, should be specified. A number of other options can be also utilized to modify the model parameters.
```
python main.py --experiment_type cross_cell_line
```
