from dataclasses import dataclass
from io import BytesIO, TextIOWrapper
from typing import Optional

import h5py
import numpy as np
import pandas as pd
import pandas.api.types
import torch
from open_clip import (
    SimpleTokenizer,
    create_model_from_pretrained,
    get_tokenizer,
)
from open_clip.tokenizer import HFTokenizer
from open_clip.transform import PreprocessCfg, image_transform_v2
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm.auto import tqdm
from transformers.convert_slow_tokenizer import Tokenizer


def generate_report(row, file: Optional[TextIOWrapper] = None, is_eval: bool = False):
    age = row["age_approx"]
    if np.isnan(age):
        age = "unknown"
    sex = row["sex"]
    if sex is None:
        sex = "unknown"
    age_sex_info = ""
    if age != "unknown" and sex != "unknown":
        age_sex_info = f" of {age:.0f} years old {sex}"
    elif age != "unknown":
        age_sex_info = f" of {age:.0f} years old patient"
    elif sex != "unknown":
        age_sex_info = f" of a {sex} patient"

    lesion_size = row["clin_size_long_diam_mm"]
    lesion_size_info = ""
    if not np.isnan(lesion_size):
        lesion_size_info = (
            f"millimeters and a diameter of {lesion_size:.3f} millimeters. "
        )

    area = row["tbp_lv_areaMM2"]
    area_info = ""
    if not np.isnan(area):
        area_info = f"The lesion has an area of {area:.3f} square "

    perimeter = row["tbp_lv_perimeterMM"]
    perimeter_info = ""
    if not np.isnan(perimeter):
        perimeter_info = f"The perimeter of the lesion is {perimeter:.3f} millimeters. "

    eccentricity = row["tbp_lv_eccentricity"]
    eccentricity_info = ""
    if not np.isnan(eccentricity):
        eccentricity_info = (
            f"eccentricity of the lesion, indicating shape irregularity, is "
            f"{eccentricity:.3f}. "
        )

    border_irregularity = row["tbp_lv_norm_border"]
    border_irregularity_info = ""
    if not np.isnan(border_irregularity):
        border_irregularity_info = (
            f"Border irregularity is rated at {border_irregularity:.3f} on a scale of 0 to "
            "10. "
        )

    color_variation = row["tbp_lv_norm_color"]
    color_variation_info = ""
    if not np.isnan(color_variation):
        color_variation_info = (
            f"Color variation within the lesion is rated at {color_variation:.3f} on a "
            "scale of 0 to 10. "
        )

    anatom_site = row["anatom_site_general"]

    report = (
        f"A photo was taken of a skin lesion located on the {anatom_site}{age_sex_info}"
        ". The patient suspects skin cancer. "
        f"{area_info}"
        f"{lesion_size_info}"
        f"{perimeter_info}"
        f"{eccentricity_info}"
        f"{border_irregularity_info}"
        f"{color_variation_info}"
    ).strip()
    if "target" in row:
        target = "malignant" if row["target"] == 1 else "benign"
        report += (
            f" Based on the image and the description, the lesion is likely {target}."
        )
    if file is not None:
        isic_id = row["isic_id"]
        file.write(f"{isic_id}\t{report}\n")
    if "target" in row and not is_eval:
        return report
    else:
        report1 = (
            report
            + " Based on the image and the description, the lesion is likely benign."
        )
        report2 = (
            report
            + " Based on the image and the description, the lesion is likely malignant."
        )
        return report1, report2


def generate_reports(metadata, path):
    with open(path, "w") as file:
        file.write("isic_id\treport\n")
        for _, row in tqdm(metadata.iterrows(), total=metadata.shape[0]):
            generate_report(row, file)


def init_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    clip_model, _ = create_model_from_pretrained(
        "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
    )  # type: ignore
    tokenizer = get_tokenizer(
        "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
    )
    pp_cfg = PreprocessCfg(**clip_model.visual.preprocess_cfg)
    aug_cfg = None
    preprocess_train = image_transform_v2(
        pp_cfg,
        is_train=True,
        aug_cfg=aug_cfg,
    )
    preprocess_val = image_transform_v2(
        pp_cfg,
        is_train=False,
    )
    return clip_model.to(device), preprocess_train, preprocess_val, tokenizer


@dataclass
class DataInfo:
    dataloader: DataLoader
    sampler = None
    shared_epoch = None

    def set_epoch(self, epoch):
        if self.shared_epoch is not None:
            self.shared_epoch.set_value(epoch)
        if self.sampler is not None and hasattr(self.sampler, "set_epoch"):
            self.sampler.set_epoch(epoch)


class IsicChallengeDataset(Dataset):
    def __init__(
        self,
        hdf5_path: str,
        metadata_or_path: str,
        tokenizer: SimpleTokenizer | HFTokenizer | Tokenizer = None,
        transform: Optional[transforms.Compose] = None,
        is_train: bool = False,
    ):
        """
        Args:
            hdf5_path (string): Path to the hdf5 file with image data.
            csv_path (string): Path to the csv file with text data.
            transform (callable, optional): Optional transform to be applied on an image.
        """
        self.hdf5_path = hdf5_path
        if isinstance(metadata_or_path, str):
            self.metadata_path = metadata_or_path
            self.text_data = pd.read_csv(metadata_or_path).set_index("isic_id")
        else:
            self.text_data = metadata_or_path
            if "isic_id" not in self.text_data.columns:
                self.text_data["isic_id"] = self.text_data.index
            self.text_data = self.text_data.set_index("isic_id")

        self.transform = transform

        self.indices = self.text_data.index
        if "target" in self.text_data.columns:
            self.targets = self.text_data["target"].values
        else:
            self.targets = None

        # Open the HDF5 file
        self.hdf5_file = h5py.File(hdf5_path, "r")
        self.tokenizer = tokenizer

        self.is_train = is_train

    def __len__(self):
        return len(self.text_data)

    def _load_image(self, idx):
        image: bytes = self.hdf5_file[idx][()]  # type: ignore

        image_bytes = BytesIO(image)
        image = Image.open(image_bytes)  # type: ignore
        if self.transform:
            image = self.transform(image)
        return image

    def _get_text(self, text: str):
        tokens = None
        if isinstance(self.tokenizer, HFTokenizer):
            tokens = self.tokenizer(text)[0]
        elif isinstance(self.tokenizer, SimpleTokenizer):
            tokens = self.tokenizer.encode(text)
        elif isinstance(self.tokenizer, Tokenizer):
            tokens = self.tokenizer.encode(text)
        else:
            raise ValueError("Tokenizer not recognized")
        return tokens

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = [self.indices[i] for i in idx.tolist()]
        else:
            idx = self.indices[idx]

        # Get image
        image = self._load_image(idx)

        # Get text
        batch = self.text_data.loc[idx]

        pos_texts = []
        neg_texts = []

        targets = []
        if isinstance(idx, list) or torch.is_tensor(idx):
            for _, row in batch.iterrows():
                if self.is_train:
                    targets.append(generate_report(row, is_eval=False))
                else:
                    neg_txt, pos_txt = generate_report(row, is_eval=True)
                    pos_texts.append(pos_txt)
                    neg_texts.append(neg_txt)
                    targets.append(row["target"])
        else:
            if self.is_train:
                targets.append(generate_report(batch, is_eval=False))
            else:
                neg_txt, pos_txt = generate_report(batch, is_eval=True)
                pos_texts.append(pos_txt)
                neg_texts.append(neg_txt)
                targets.append(batch["target"])

        if self.is_train:
            return image, self._get_text(targets)  # , self._get_text(text[1])
        else:
            return (
                image,
                self._get_text(neg_texts),
                self._get_text(pos_texts),
                torch.tensor(targets),
            )

    def close(self):
        self.hdf5_file.close()
