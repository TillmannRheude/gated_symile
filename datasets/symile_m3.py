import torch
import os

from torch.utils.data import Dataset
from transformers import AutoTokenizer


def get_language_constant(num_langs):
    """
    For the Symile-M3 experiments. Returns a list of language abbreviations
    (ISO-639 codes) based on the specified number of languages.
    """
    LANGUAGES_2 = [
        "en",   # english
        "el"    # greek
    ]
    LANGUAGES_5 = [
        "el",   # greek
        "en",   # english
        "hi",   # hindi
        "ja",   # japanese
        "uk"    # ukrainian
    ]
    LANGUAGES_10 = [
        "ar",   # arabic
        "el",   # greek
        "en",   # english
        "hi",   # hindi
        "ja",   # japanese
        "ko",   # korean
        "te",   # telugu
        "th",   # thai
        "uk",   # ukrainian
        "zh-CN" # chinese
    ]
    if num_langs == 10:
        return LANGUAGES_10
    elif num_langs == 5:
        return LANGUAGES_5
    elif num_langs == 2:
        return LANGUAGES_2


class Dataset_SymileM3(Dataset):
    def __init__(
        self, 
        data_dir: str = "/path/to/anonymized/tensors",
        split: str = "train",
        split_nr: int = 0,
        text_model_id: str = "xlm-roberta-large",
        num_langs: int = 5,
    ):
        self.split = split

        self.split_dir = os.path.join(data_dir, split)

        self.text_input_ids = torch.load(os.path.join(self.split_dir, f"text_input_ids_{split}_full.pt")).long()
        self.text_attention_mask = torch.load(os.path.join(self.split_dir, f"text_attention_mask_{split}_full.pt"))
        self.max_token_len = self.text_input_ids.shape[1]

        self.image = torch.load(os.path.join(self.split_dir, f"image_{split}_full.pt"))
        self.image_mean = torch.mean(self.image, dim=0)

        self.audio = torch.load(os.path.join(self.split_dir, f"audio_{split}_full.pt"))
        self.audio_mean = torch.mean(self.audio, dim=0)

        self.cls_id = torch.load(os.path.join(self.split_dir, f"cls_id_{split}_full.pt"))
        self.idx = torch.load(os.path.join(self.split_dir, f"idx_{split}_full.pt"))

        with open(os.path.join(self.split_dir, f"lang_{split}_full.txt"), "r") as f:
            self.lang = f.read().splitlines()
        self.lang = self.lang[0]
        lang_list = [self.lang[i:i+2] for i in range(0, len(self.lang), 2)]
        self.lang = lang_list
        
        self.languages = get_language_constant(num_langs)

        # If running an experiment that includes missingness, load the missingness tensors.
        # The missingness tensors are only used during training and validation.
        """ 
        self.txt_tokenizer = AutoTokenizer.from_pretrained(text_model_id)
        if getattr(self.args, "missingness", False) and self.split != "test":
            if self.args.missingness_prob == 0.5:
                missingness_prob_str = "50"
            elif self.args.missingness_prob == 0.6:
                missingness_prob_str = "60"
            elif self.args.missingness_prob == 0.65:
                missingness_prob_str = "65"
            elif self.args.missingness_prob == 0.7:
                missingness_prob_str = "70"
            elif self.args.missingness_prob == 0.75:
                missingness_prob_str = "75"
            else:
                raise ValueError("Missingness probability not supported.")
            self.text_missingness = torch.load(self.split_dir / f"text_missingness_prob{missingness_prob_str}_{split}.pt")
            self.image_missingness = torch.load(self.split_dir / f"image_missingness_prob{missingness_prob_str}_{split}.pt")
            self.audio_missingness = torch.load(self.split_dir / f"audio_missingness_prob{missingness_prob_str}_{split}.pt")
        """

    def __len__(self):
        return len(self.image)

    def get_missingness_text(self):
        """
        Get a tokenized representation for MISSING_TOKEN.

        Returns:
            dict: with keys "input_ids" and "attention_mask", whose values are
                  Torch.tensors with shape (self.max_token_len).
        """
        encoded_inputs = self.txt_tokenizer(text="[MISSING]",
                                            return_tensors="pt",
                                            padding="max_length",
                                            max_length=self.max_token_len)
        encoded_inputs["input_ids"] = torch.squeeze(encoded_inputs["input_ids"], dim=0)
        encoded_inputs["attention_mask"] = torch.squeeze(encoded_inputs["attention_mask"], dim=0)
        encoded_inputs["attention_mask"] = encoded_inputs["attention_mask"].to(torch.float32)
        return encoded_inputs

    def __getitem__(self, idx):
        """
        Indexes into the dataset.

        If running a missingness experiment, the text, image, and audio data may be missing.
        If the text data is missing, the text data is replaced with a tokenized representation of MISSING_TOKEN.
        If the image or audio data is missing, it is replaced with the mean image or mean audio computed from
        the training set, respectively.

        Args:
            idx (int): Index of data sample to retrieve.

        Returns:
            dict: A dictionary containing the following key-value pairs:
                - text (dict): Dictionary with keys "input_ids" and "attention_mask".
                - image (torch.Tensor): Tensor with image data.
                - audio (torch.Tensor): Tensor with audio data.
                - cls_id (torch.float32): Tensor containing the class id for the sample
                    (as determined by the image class name).
                - idx (torch.float32): Tensor containing the unique identifier for the sample.
                - lang_id (int): Integer representing the language id for the sample.
                - text_missing (torch.int32): Integer indicating whether the text data is observed (0) or missing (1).
                - image_missing (torch.int32): Integer indicating whether the image data is observed (0) or missing (1).
                - audio_missing (torch.int32): Integer indicating whether the audio data is observed (0) or missing (1).
                - all_observed (int): Integer indicating whether all modalities are observed (1) or if some modalities
                    are missing (0).
        """
        text = {"input_ids": self.text_input_ids[idx],
                "attention_mask": self.text_attention_mask[idx]}
        image = self.image[idx]
        audio = self.audio[idx]

        text_missing = 0
        image_missing = 0
        audio_missing = 0
        """ 
        # If running an experiment that includes missingness, load the missingness tensors.
        # The missingness tensors are only used during training and validation.
        if getattr(self.args, "missingness", False) and self.split != "test":
            text_missing = self.text_missingness[idx]
            image_missing = self.image_missingness[idx]
            audio_missing = self.audio_missingness[idx]

            if text_missing == 1:
                text = self.get_missingness_text()

            if image_missing == 1:
                image = self.image_mean

            if audio_missing == 1:
                audio = self.audio_mean

        if (text_missing == 0) and (image_missing == 0) and (audio_missing == 0):
            all_observed = 1
        else:
            all_observed = 0
        """
        all_observed = 1 

        return {"text": text,
                "image": image,
                "audio": audio,
                "cls_id": self.cls_id[idx],
                "idx": self.idx[idx],
                "lang_id": self.languages.index(self.lang[idx]),
                "text_missing": text_missing,
                "image_missing": image_missing,
                "audio_missing": audio_missing,
                "all_observed": all_observed}

