# Notebook for reading the JSON files from the omi-db dataset

contact: robert.marti@udg.edu

### Getting started:


Create a new conda environment

``` 
conda create -n iceberg
conda activate iceberg
```

Install the requirements:

```
pip install -r requirements.txt
```

### Preprocess data:

In order to preprocess the data and extract the ROIs and generate the corresponding csvs:

- Set the configuration file (data_processing/database_config.py) properly
- Run:
    ```
    python python data_processing/database_construction.py
    ```

