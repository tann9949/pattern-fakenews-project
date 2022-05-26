import numpy as np
import torch
from datasets import load_metric
from transformers import (AutoModelForSequenceClassification, 
                          AutoTokenizer, DataCollatorWithPadding, 
                          TrainingArguments, Trainer)

from baseline.evaluate import calculate_score
from baseline.tokenizer import tokenize
from baseline.limesoda import LimeSodaDataset, read_limesoda

LIMESODA_DIR = "/workspace/dataset/LimeSoda"
DELIMITER = ""
BATCH_SIZE = 24
EPOCHS = 200
LEARNING_RATE = 2e-6
# model_name = 'weights/raw/wangchanberta-simcse-raw-bs24-epoch1-lr3e-05'
model_name = 'weights/sentencepiece/wangchanberta-simcse-sentencepiece-bs24-epoch1-lr3e-05'


def main():
    
    from torch.multiprocessing import Pool, Process, set_start_method
    try:
         set_start_method('spawn')
    except RuntimeError:
        pass
    
    for TRAIN_PERCENTAGE in [100, 80, 60, 40]:
        print(">"*20)
        print(f"Training with {TRAIN_PERCENTAGE}% data")
        print(">"*20)
        SAVE_DIR = f"./results/sentencepiece-simcse/wangchanberta-200epochs-{TRAIN_PERCENTAGE}percent/"
        
        TRAIN_PERCENTAGE /= 100
    
        # prepare dataset #
        dataset = read_limesoda(limesoda_dir=LIMESODA_DIR, train_percentage=TRAIN_PERCENTAGE, delimiter=DELIMITER)

        tokenizer = AutoTokenizer.from_pretrained(model_name)

        train_dataset = LimeSodaDataset(f"{LIMESODA_DIR}/../tempLimesoda/train_v1.jsonl", tokenizer)
        val_dataset = LimeSodaDataset(f"{LIMESODA_DIR}/../tempLimesoda/val_v1.jsonl", tokenizer)
        test_dataset = LimeSodaDataset(f"{LIMESODA_DIR}/../tempLimesoda/test_v1.jsonl", tokenizer)
    
        # prepare model #
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

        if torch.cuda.is_available():
            model = model.cuda()

        # train model #
        metric = load_metric("accuracy")

        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            predictions = np.argmax(logits, axis=-1)
            return metric.compute(predictions=predictions, references=labels)

        training_args = TrainingArguments(
            output_dir=SAVE_DIR,
            learning_rate=LEARNING_RATE,
            per_device_train_batch_size=BATCH_SIZE,
            per_device_eval_batch_size=16,
            num_train_epochs=EPOCHS,
            weight_decay=0.01,
            save_steps=5000,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics
        )

        trainer.train()
        model.save_pretrained(SAVE_DIR + "final")

        # evaluate
        result = trainer.predict(val_dataset)
        val_result = calculate_score(dataset["val"]["label"].values, result.predictions.argmax(-1))
        _ = val_result.pop("prediction")

        result = trainer.predict(test_dataset)
        test_result = calculate_score(dataset["test"]["label"].values, result.predictions.argmax(-1))
        _ = test_result.pop("prediction")

        print("VALIDATION RESULT")
        print(val_result)

        print("TEST RESULT")
        print(test_result)


if __name__ == "__main__":
    main()
