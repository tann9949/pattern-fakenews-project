import json
import os
import re
from glob import glob

from sentence_transformers import SentenceTransformer, InputExample
from sentence_transformers import models, losses
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

# notebook lib
import pandas as pd
import matplotlib.pyplot as plt

DATA_DIR = "dataset"
NEWS_DIR = f"{DATA_DIR}/healthcare-news"
# DATASET = "raw"
DATASET = "sentencepiece"
# DATASET = "wordpiece"


def load_data(set_name):
    if set_name.lower().strip() == "wordpiece":
        json_paths = sorted(glob(f"{NEWS_DIR}/wordpiece/*.json"))
        return {
            os.path.basename(f_name).split("_")[0]: [
                json.loads(line.strip())
                for line in open(f_name).readlines()
            ]
            for f_name
            in tqdm(json_paths)
        }
    elif set_name.lower().strip() == "sentencepiece":
        return [l.strip() for l in open(f"{NEWS_DIR}/sentencepiece/sentencepiece.txt")]
    elif set_name.lower().strip() == "raw":
        return [l.strip() for l in open(f"{NEWS_DIR}/healthcare-raw.txt")]

    
def main():
    dataset = load_data(DATASET)
    for batch_size in [24]:
        for learning_rate in [3e-5]:
            for epochs in [1]:
                print(">"*30)
                print(f"Batch size: {batch_size}")
                print(f"Learning Rate: {learning_rate}")
                print(f"Epochs: {epochs}")
                print(">"*30)
                ## load model ##
                # LM models (mBERT, XLM-R, etc.)
                model_name = 'airesearch/wangchanberta-base-att-spm-uncased'
                word_embedding_model = models.Transformer(model_name, max_seq_length=416)
                pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), pooling_mode='cls')
                model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

                ## load dataset ##
                train_sentences = dataset
                # Convert train sentences to sentence pairs
                train_data = [InputExample(texts=[s, s]) for s in train_sentences]
                # DataLoader to batch your data
                train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

                # Use the denoising auto-encoder loss
                train_loss = losses.MultipleNegativesRankingLoss(model)

                ##  fit model ##
                model.fit(
                    train_objectives=[(train_dataloader, train_loss)],
                    epochs=epochs,
                    show_progress_bar=True,
                    optimizer_params={'lr': learning_rate},
                    output_path=f'weights/{DATASET}/wangchanberta-simcse-{DATASET}-bs{batch_size}-epoch{epochs}-lr{learning_rate}',
                    save_best_model=True,
                    use_amp=True,
                    checkpoint_save_steps=1000
                )

                model.save(f"weights/{DATASET}/wangchanberta-simcse-{DATASET}-bs{batch_size}-epoch{epochs}-lr{learning_rate}/final")
            

if __name__ == "__main__":
    main()
