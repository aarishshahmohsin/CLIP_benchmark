import os
with open('./webdatasets.txt', 'r') as f:

    datasets = [line.strip() for line in f if line.strip()]
    for dataset in datasets:
        path = f'./datasets_cache/wds/{dataset.split("/")[-1]}'
        if not os.path.exists(path):
            os.makedirs(path)
            print(f"Created directory: {path}")
        else:
            print(f"Directory already exists: {path}")