## Preprocessing data

These code scripts can be used for preparing the input and prediction data for training and inference.
- Please change the name of directory and the file names as necessary.
- The code scripts assume the epigenetics data are present in '.bigWig' file format and post transcription regulatory data in '.xlsx' file format.
- The names of the epigenetic data files are assumed to be in this format: <epigenetic_name>.bigWig. For example, 'CTCF.bigWig'.
- The contact matrix in '.hic' format can be directly fed to the model where it is processed. No separate processing was used for that.
