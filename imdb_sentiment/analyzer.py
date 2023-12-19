import importlib.metadata as metadata

import numpy as np
import pandas as pd
import tensorflow_datasets as tfds
from transformers import Trainer, AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, \
    DataCollatorWithPadding, EarlyStoppingCallback

from imdb_sentiment.shared.utils.logger import Logger
from shared.utils.custom_dataset import CustomDataset

max_tokenization_length = 512

__version__ = metadata.version(__package__ or __name__)

tokenized_args = {
    'padding': True,
    'truncation': True,
    'max_length': max_tokenization_length
}


class Analyzer:
    def __init__(self):
        self.logger = Logger()
        model_name = 'siebert/sentiment-roberta-large-english'
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)

    def run(self):
        self.logger.info(f"Analyzing")
        train_dataset, test_dataset = self.get_dataset()
        self.pretrain(train_dataset, test_dataset)
        self.model.save_pretrained("pretrained-model")
        self.tokenizer.save_pretrained("pretrained-model")

        val_trainer = Trainer(self.model)
        test_dataset = test_dataset[:10]
        val_text_tokenized = self.tokenizer(list(test_dataset['text']), **tokenized_args)
        val_label = list(test_dataset['label'])
        val_dataset = CustomDataset(val_text_tokenized, val_label)
        correct = 0
        raw_pred, _, _ = val_trainer.predict(val_dataset)

        y_pred = np.argmax(raw_pred, axis=1)
        for index, (label, text) in enumerate(test_dataset.values):
            print(index, label, y_pred[index])
            if label == y_pred[index]:
                print("Correct")
                correct += 1
            else:
                print("Incorrect")
        print(f"Accuracy: {correct / len(test_dataset)}")

    def tokenize_function(self, example):
        return self.tokenizer(example['text'], truncation=True, max_length=max_tokenization_length)

    def pretrain(self, train_dataset, test_dataset):
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

        train_text_tokenized = self.tokenizer(list(train_dataset['text']), **tokenized_args)
        test_text_tokenized = self.tokenizer(list(test_dataset['text']), **tokenized_args)
        train_label = list(train_dataset['label'])
        test_label = list(test_dataset['label'])
        train_dataset = CustomDataset(train_text_tokenized, train_label)
        test_dataset = CustomDataset(test_text_tokenized, test_label)

        training_args = TrainingArguments("test-trainer", evaluation_strategy="steps",
                                          eval_steps=500,
                                          per_device_train_batch_size=8,
                                          per_device_eval_batch_size=8,
                                          num_train_epochs=3,
                                          seed=0,
                                          load_best_model_at_end=True, )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )
        trainer.train()
        trainer.evaluate()

    @staticmethod
    def get_dataset() -> (pd.DataFrame, pd.DataFrame):
        dataset, info = tfds.load('imdb_reviews/plain_text', with_info=True, as_supervised=True)
        train_dataset, test_dataset = dataset['train'], dataset['test']
        train_df = tfds.as_dataframe(train_dataset, info)
        test_df = tfds.as_dataframe(test_dataset, info)
        train_df['text'] = train_df['text'].astype(str)
        test_df['text'] = test_df['text'].astype(str)
        return train_df, test_df
