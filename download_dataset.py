def main():
    import zipfile

    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()


    # Download a dataset (replace with the dataset you want)
    api.competition_download_files(
        "isic-2024-challenge", path="data/", force=True, quiet=False
    )
    # unzip the dataset

    with zipfile.ZipFile("data/isic-2024-challenge.zip", "r") as zip_ref:
        zip_ref.extractall("data/isic-2024-challenge")

if __name__ == '__main__':
    main()
    # %%
