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
SAVE_DIR = "./results/wangchanberta-200epochs-40percent/"
BATCH_SIZE = 24
EPOCHS = 200
LEARNING_RATE = 2e-6
TRAIN_PERCENTAGE = 0.4
    
    
def main():
    
    # prepare dataset #
    dataset = read_limesoda(limesoda_dir=LIMESODA_DIR, train_percentage=TRAIN_PERCENTAGE, delimiter=DELIMITER)
    
    model_name = 'airesearch/wangchanberta-base-att-spm-uncased'
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
