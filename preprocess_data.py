from glob import glob

import numpy as np
import pandas as pd
import torch
from flair.data import Sentence
from flair.models import SequenceTagger
from tqdm import tqdm
from transformers import RobertaForSequenceClassification, RobertaTokenizer


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
    roberta = RobertaForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest").to(4)
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

    data_list = pd.DataFrame(data_list, columns=["stock_name", "date"])  # (~ x 2)

    for index in tqdm(range(0, data_list.shape[0], batch_size)):
        batch: torch.Tensor
        batch = texts[index : index + batch_size]  # (-1)
        output = tokenizer(batch, padding=True, return_tensors="pt", truncation=True).to(4)
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
        ]  # (-1 x ~)
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
    # preprocess_text_data()
    extract_keywords()
    # preprocess_price_data()
