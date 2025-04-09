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

def load_pretrained_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return tokenizer

class DatasetProcessor(Dataset):
    def __init__(self, df, tokenizer):
        """
        :param df:
        :param tokenizer:
        """
        self.df = df
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.df)

    def __getitems__(self, keys: list):
        row = self.df.iloc(list)

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
dataset = DatasetProcessor(df, load_pretrained_and_tokenizer())