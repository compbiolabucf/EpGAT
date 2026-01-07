# EpGAT
This repository represents an attention-based model named EpGAT for predicting and quantifying post-transcriptional regulatory events- Alternative Splicing (AS) and Alternative Polyadenylation (APA), using HiC and epigenetic data.


## Workflow
![alt text](https://github.com/compbiolabucf/EpGAT/blob/main/EpGAT_overall.png)

## Dataset
Three types of data were used in this study - epigenetic signals, HiC contact matrices, post-transcriptional regulatory events. We used two cell lines in our experiments- MCF7 and K562. Therefore, all the datasets used correspond to these cell lines.
The python scripts inside [preprocess_data](https://github.com/compbiolabucf/EpGAT/tree/main/preprocess_data) folder can be used to prepare the data for application of the model.

- Epigenetic data: We used 32 epigenetic signals for our experiments: ELF1, CTCF, CREB1, JUND, MYC, CEBPB, POLR2A, MAX, GABPA, SP1, H4K20me1, EP300, JUN, EGR1, REST, H3K4me1, H3K79me2, TAF1, SIN3A, H3K4me3, RAD21, ZBTB33, H3K4me2, H3K9ac, H2AFZ, H3K27ac, H3K36me3, H3K27me3, H3K9me3, NRF1, HDAC2, FOXA1. However, the number of epigenetic signals used can be varied using 'num_eps' hyperparameter. All of the epigenetic signals were downloaded in 'bigWig' file format from publicly avialble ENCODE project website. These files were preprocessed using 'preprocess_epigenetic_data.py' script present inside [preprocess_data](https://github.com/compbiolabucf/EpGAT/tree/main/preprocess_data) folder. 
- HiC data: The HiC data containing chromosome contact matrices were also downloaded from publicly available ENCODE project website. Intact HiC matrix was downloaded for MCF7 cell line from Experiment ID: ENCSR660LPJ, File ID: ENCFF420JTA and for K562 cell line from Experiment ID: ENCSR479XDG, File ID: ENCFF621AIY. The HiC file contained contact matrices at different resolutions (10bp, 200bp, 1000bp, 5000bp etc). These matrices were already processed by removing contacts which had mapQ (mapping quality) value of less than 30. We extracted the matrices for 200bp resolution to be used in our pipeline. 
- Post-transcriptional regulatory events: Two different post transcriptional regulatory events can be predicted by EpGAT- AS and APA. They were identified and quantified by customized pipelines developed by our lab [AS-Quant](https://github.com/compbiolabucf/AS-Quant) and [APA-Scan](https://github.com/compbiolabucf/APA-Scan), applied to RNA-seq data corresponding to the specific cell lines. The events were quantified in terms of Percent Spliced-In (PSI) for AS and Truncated Ratio (TR) for APA. These datasets were further preprocessed by 'preprocess_as_prediction_data.py' and 'preprocess_apa_prediction_data.py' scripts present inside [preprocess_data](https://github.com/compbiolabucf/EpGAT/tree/main/preprocess_data) folder to match the resolution and to make compatible for EpGAT application.

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
Example command for cross cell line experiment:
```
python main.py --exp_type cross_cell_line --train_cell_line MCF7 --test_cell_line K562 --chromosome 1 --event as --root_directory data
```
Example command for cross chromosome experiment:
```
python main.py --exp_type cross_chromosome --cell_line MCF7 --train_chrs 1 2 3 4 5 --val_chrs 6 13 --test_chrs 7 14 --event apa --root_directory data
```
