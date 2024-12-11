from glob import glob

import numpy as np
import pandas as pd
import torch
from flair.data import Sentence
from flair.models import SequenceTagger
from tqdm import tqdm
from transformers import RobertaForSequenceClassification, RobertaTokenizer
from torchvision import datasets, transforms
from PIL import Image
from torch.utils.data import DataLoader


@torch.no_grad()
def preprocess_image_data():
    """
    Process image data by loading, resizing, and normalizing.
    Save processed data for each stock in the specified folder.
    """
    image_paths = sorted(glob("./data/plot/*/*.jpg"))  # e.g., ./data/{ticker}/{date}.jpg
    data_list = []
    
    # Define the transformation for mean and std calculation
    basic_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    
    # Load dataset for mean and std computation
    dataset = datasets.ImageFolder('./data/plot', transform=basic_transform)
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)
    
    # Compute mean and std
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('Computing mean and std...')
    for images, _ in tqdm(loader, desc="Computing stats"):
        for i in range(3):  # Loop through each channel
            mean[i] += images[:, i, :, :].mean()
            std[i] += images[:, i, :, :].std()
    mean /= len(loader)
    std /= len(loader)

    # Define the image transformation pipeline with computed mean and std
    image_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean.numpy(), std=std.numpy())
    ])

    for img_path in tqdm(image_paths):
        temp = img_path.split("/")
        stock_name = temp[-2]  # Extract stock name from folder
        date = int(temp[-1].replace(".jpg", "").replace("-", ""))  # Extract date from filename
  
        try:
            img = Image.open(img_path).convert("RGB")  # Open and convert to RGB
            img_tensor = image_transform(img).numpy()  # Apply transformations
            data_list.append((stock_name, date, img_tensor))
        except Exception as e:
            print(f"Error processing image {img_path}: {e}")

    # Convert to DataFrame for easier manipulation
    data_list = pd.DataFrame(data_list, columns=["stock_name", "date", "image"])

    for stock_name in data_list["stock_name"].unique():
        mask = data_list["stock_name"] == stock_name
        temp = data_list.loc[mask, ["date", "image"]]
        temp.to_pickle(f"./data/my_stocknet/image/{stock_name}.pkl")  # Save as pickle for efficient storage

    return


# preprocess text data
def preprocess_sentence(text: str):
    new_text = []
    for t in text.split(" "):
        t = "@user" if t.startswith("@") and len(t) > 1 else t
        t = "http" if t.startswith("http") else t
        new_text.append(t)
    return " ".join(new_text)


@torch.no_grad()
def preprocess_text_data(batch_size: int = 512):
    tokenizer = RobertaTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")
    tokenizer.model_max_length = 512
    roberta = RobertaForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")#.to(0)
    roberta.requires_grad_(False)
    roberta.eval()

    fnames = sorted(glob("./data/stocknet-dataset/tweet/raw/*/*"))
    data_list = []
    texts = []
    sentiments = []
    embeddings = []

    for fname in fnames:
        temp = fname.split("/")
        date = int(temp[-1].replace("-", ""))
        stock_name = temp[-2]

        with open(fname, "rb") as f:
            temp = f.read().decode("utf-8")
            temp = temp.split("\n")
            temp = [text.split('"text":"')[1].split('","source"')[0] for text in temp if text != ""]
            for text in temp:
                data_list.append((stock_name, date))
                texts.append(preprocess_sentence(text))
            f.close()

    data_list = pd.DataFrame(data_list, columns=["stock_name", "date"])  # (전체 주식의 트윗 수 x 2)

    for index in tqdm(range(0, data_list.shape[0], batch_size)):
        batch: torch.Tensor
        batch = texts[index : index + batch_size]  # (-1)
        output = tokenizer(batch, padding=True, return_tensors="pt", truncation=True)#.to(0)
        batch = roberta.roberta(**output)
        del output
        batch = batch.last_hidden_state

        sentiment: torch.Tensor
        sentiment = roberta.classifier(batch)
        batch = batch.cpu()  # (-1 x 768)
        sentiment = sentiment.cpu()  # (-1 x 3)

        sentiment = sentiment.softmax(1).numpy()  # (-1 x 3)
        batch = batch[:, 0, :].numpy()  # (-1 x 768)
        sentiments.append(sentiment)
        embeddings.append(batch)

    del texts
    sentiments = np.concatenate(sentiments, axis=0)  # (~ x 3)
    embeddings = np.concatenate(embeddings, axis=0)  # (~ x 768)

    for stock_name in data_list["stock_name"].unique():
        mask = data_list["stock_name"] == stock_name
        temp = data_list.loc[mask, ["date"]].values  # (~ x 1)
        sentiment = sentiments[mask, :]  # (~ x 3)
        embedding = embeddings[mask, :]  # (~ x 768)
        temp = np.concatenate([temp, sentiment, embedding], axis=1)  # (~ x 1+3+768)
        np.save(f"./data/my_stocknet/tweet/{stock_name}.npy", temp)

    return


@torch.no_grad()
def extract_keywords(batch_size=512):
    tagger = SequenceTagger.load("flair/pos-english")
    tagger.requires_grad_(False)
    tagger.eval()

    fnames = sorted(glob("./data/stocknet-dataset/tweet/raw/*/*"))
    data_list = []
    texts = []
    keywords = []

    for fname in fnames:
        temp = fname.split("/")
        date = int(temp[-1].replace("-", ""))
        stock_name = temp[-2]

        with open(fname, "rb") as f:
            temp = f.read().decode("utf-8")
            temp = temp.split("\n")
            temp = [text.split('"text":"')[1].split('","source"')[0] for text in temp if text != ""]
            for text in temp:
                data_list.append((stock_name, date))
                texts.append(preprocess_sentence(text))
            f.close()

    data_list = pd.DataFrame(data_list, columns=["stock_name", "date"])  # (~ x 2)
    ext_sets = {"JJ", "JJR", "JJS", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ"}

    for index in tqdm(range(0, data_list.shape[0], batch_size)):
        batch = texts[index : index + batch_size]  # (-1)
        batch = [Sentence(text) for text in batch]
        tagger.predict(batch, mini_batch_size=512)

        temp = [sent.to_dict("pos")["tokens"] for sent in batch]
        temp = [
            [word_info["text"] for word_info in sent_info if word_info["labels"][0]["value"] in ext_sets]
            for sent_info in temp
        ]  # (전체 트윗 수 x ~) -> 각 트윗 단어 중 ext_sets에 해당하는 품사만 추출
        temp = [
            [word for word in sent if word not in {"is", "was", "are", "were", "be", "been", "'s"}] for sent in temp
        ]
        temp = ["\t".join(sent) for sent in temp]  # (-1)
        keywords.extend(temp)

    keywords = pd.DataFrame(keywords, columns=["keywords"])  # (~ x 1)
    keywords = pd.concat([data_list, keywords], axis=1)  # (~ x 3)

    for stock_name in data_list["stock_name"].unique():
        mask = data_list["stock_name"] == stock_name
        temp: pd.DataFrame
        temp = keywords.loc[mask, ["date", "keywords"]]  # (~ x 1)
        temp.to_parquet(f"./data/my_stocknet/keywords/{stock_name}.parquet", index=False)

    return


# preprocess stock data
def preprocess_price_data():
    fnames = sorted(glob("./data/stocknet-dataset/price/raw/*.csv"))

    for fname in fnames:
        stock_name = fname.split("/")[-1][:-4]
        data = pd.read_csv(fname)
        data.columns = data.columns.str.lower()
        data["date"] = data["date"].str.replace("-", "").astype(int)
        data["volume"] /= 1e8
        data.to_parquet(f"./data/my_stocknet/price/{stock_name}.parquet", index=False)

    return


if __name__ == "__main__":
    #preprocess_text_data()
    #extract_keywords()
    #preprocess_price_data()
    preprocess_image_data()
