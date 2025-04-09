import pandas as pd
from datasets import Dataset
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn.model_selection import train_test_split
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from transformers import AutoTokenizer
import transformers
import accelerate

print(transformers.__version__)
print(accelerate.__version__)
model_name = "bert-base-uncased"  # Choose a pre-trained model

# Sample data (replace with your actual liquidity analysis data)
def prepare_test_dataset():
    data = {
        'text': [
            "Cash flow is strong, indicating good liquidity.",
            "High accounts receivable may lead to liquidity issues.",
            "The current ratio is below 1, signaling potential problems.",
            "Working capital is increasing, a positive sign.",
            "Quick ratio of 1.5 suggests healthy liquidity.",
            "High debt levels are a liquidity risk.",
            "Positive operating cash flow indicates good liquidity.",
            "Large inventory holdings are a liquidity concern.",
        ],
        'label': [1, 0, 0, 1, 1, 0, 1, 0],  # 1 for positive, 0 for negative liquidity
    }
    df = pd.DataFrame(data)
    # Split into training and validation sets
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    # Convert to Hugging Face datasets
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)
    return tokenized_datasets(train_dataset, val_dataset)

def load_pretrained_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return tokenizer

def define_model():
    return AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2) #model

def tokenize_function(examples):
    tokenizer = load_pretrained_and_tokenizer()
    return tokenizer(examples["text"], padding="max_length", truncation=True)

def tokenized_datasets(train_dataset, val_dataset):
    tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
    tokenized_val_dataset = val_dataset.map(tokenize_function, batched=True)
    return tokenized_train_dataset, tokenized_val_dataset

def compute_metrics(prediction):
    labels = prediction.label_ids
    predictions = prediction.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary')
    acc = accuracy_score(labels, predictions)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall,
    }

def main():
    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=10, #Increased epochs
        weight_decay=0.01,
        logging_dir="./logs",
    )

    tokenized_train_dataset, tokenized_val_dataset = prepare_test_dataset()
    model = define_model()

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_val_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    evaluation_results = trainer.evaluate()
    print(evaluation_results)
    model.save_pretrained("./fine_tuned_liquidity_model")
    tokenizer = load_pretrained_and_tokenizer()
    tokenizer.save_pretrained("./fine_tuned_liquidity_model")

if __name__ == "__main__":
    main()