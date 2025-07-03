# Procedure
- Download dataset from https://amazon-reviews-2023.github.io/. And put the dataset into directory like `data/amazon-industrial/Industrial_and_Scientific.jsonl`
- Generated processed datasets and tokenizer
    ```bash
    python prepare --dataset_name amazon-industrial
    ```
- Transform the precessed datasets into RecBole format
    ```bash
    python transform.py --dataset_name amazon-industrial
    ```
- Train the data generator
    ```bash
    chmod +x ./train.sh
    ./train.sh
    ```
- Generate data by inference
    ```bash
    chmod +x ./generate.sh
    ./generate.sh
    ```