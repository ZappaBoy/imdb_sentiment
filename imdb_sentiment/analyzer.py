import importlib.metadata as metadata
import os.path

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_datasets as tfds
import torch
from sklearn.model_selection import train_test_split
from transformers import Trainer, AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, \
    DataCollatorWithPadding, EarlyStoppingCallback

from imdb_sentiment.shared.utils.logger import Logger
from shared.utils.custom_dataset import CustomDataset

__version__ = metadata.version(__package__ or __name__)

# Disable GPU on Tensorflow. Tensorflow is not used in this project, but it is used by the other dependencies.
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.set_visible_devices([], 'GPU')

torch.cuda.empty_cache()

seed = 0


class Analyzer:
    model_name = 'siebert/sentiment-roberta-large-english'
    ignore_original_split = True
    test_size = 0.25
    max_tokenization_length = 512
    tokenized_args = {
        'padding': True,
        'truncation': True,
        'max_length': max_tokenization_length
    }

    pretrained_model_path = 'pretrained-model'

    training_args = TrainingArguments("test-trainer", evaluation_strategy="steps", auto_find_batch_size=True,
                                      eval_steps=50, warmup_steps=2, eval_accumulation_steps=1, num_train_epochs=3,
                                      gradient_accumulation_steps=4, seed=seed, fp16=True,
                                      gradient_checkpointing=True, load_best_model_at_end=True)

    def __init__(self):
        self.logger = Logger()
        self.model = None
        self.tokenizer = None
        if os.path.exists(self.pretrained_model_path):
            self.load_model(self.pretrained_model_path, locally=True)
        else:
            self.load_model(self.model_name)

    def load_model(self, model_name: str, locally: bool = False):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=locally)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, local_files_only=locally)

    def run(self):
        self.logger.info(f"Analyzing")
        train_dataset, test_dataset = self.get_dataset()
        if not os.path.exists(self.pretrained_model_path):
            self.pretrain(train_dataset, test_dataset)
            self.model.save_pretrained("pretrained-model")
            self.tokenizer.save_pretrained("pretrained-model")

        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        val_trainer = Trainer(model=self.model, args=self.training_args, tokenizer=self.tokenizer,
                              data_collator=data_collator)

        val_dataset = self.build_tokenized_dataset(test_dataset)
        correct = 0
        raw_predictions, correct_predictions, metrics = val_trainer.predict(val_dataset)

        predictions = np.argmax(raw_predictions, axis=1)
        for index, (prediction, correct_prediction) in enumerate(zip(predictions, correct_predictions)):
            if prediction == correct_prediction:
                correct += 1
        self.logger.info(f"Accuracy: {correct / len(test_dataset)}")

    def tokenize_function(self, example):
        return self.tokenizer(example['text'], truncation=True, max_length=self.max_tokenization_length)

    def build_tokenized_dataset(self, df: pd.DataFrame) -> CustomDataset:
        label = list(df['label'])
        text_tokenized = self.tokenizer(list(df['text']), **self.tokenized_args)
        return CustomDataset(text_tokenized, label)

    def pretrain(self, train_dataset, test_dataset):
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

        train_dataset = self.build_tokenized_dataset(train_dataset)
        test_dataset = self.build_tokenized_dataset(test_dataset)

        trainer = Trainer(
            model=self.model,
            args=self.training_args,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )
        trainer.train()
        trainer.evaluate()

    def get_dataset(self) -> (pd.DataFrame, pd.DataFrame):
        dataset, info = tfds.load('imdb_reviews/plain_text', with_info=True, as_supervised=True)
        train_dataset, test_dataset = dataset['train'], dataset['test']
        train_df = tfds.as_dataframe(train_dataset, info)
        test_df = tfds.as_dataframe(test_dataset, info)

        train_df.drop_duplicates(inplace=True)
        test_df.drop_duplicates(inplace=True)

        # The original (state of the art) dataset is distributed as a 50-50 train-test split.
        # If ignore_original_split is applied the train and test parts will be joined and split using 75-25 ratio.
        # ignore_original_split option can be used to generate a more accurate model
        if self.ignore_original_split:
            df = pd.concat([train_df, test_df])
            train_df, test_df = train_test_split(df, test_size=self.test_size, random_state=seed,
                                                 stratify=df['label'])

        train_df['text'] = train_df['text'].astype(str)
        test_df['text'] = test_df['text'].astype(str)
        return train_df, test_df
