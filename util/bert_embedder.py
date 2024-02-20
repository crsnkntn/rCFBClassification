import os
import csv
import torch
from transformers import BertTokenizer, BertModel
from tqdm import tqdm
import random

MAX_LEN = 256
MAX_CHARS = 256

def read_csv_files(directory, classes):
    labels = set()
    for class_name in classes:
        filename = class_name + '.csv'
        if filename in os.listdir(directory):
            labels.add(class_name)
    return list(labels)

def sample_balanced_dataset(labels, data, num_samples_per_class, random_seed=42):
    balanced_data = []
    random.seed(random_seed)
    for label in labels:
        if label in data:
            sampled_data = random.sample(data[label], min(num_samples_per_class, len(data[label])))
            balanced_data.extend([(label, item) for item in sampled_data])
    return balanced_data

def convert_to_bert_embeddings(text_data, model_name='bert-base-uncased'):
    def truncate_string(s, max_length):
        if len(s) > max_length:
            return s[:max_length]
        return s

    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)

    embeddings = []
    with torch.no_grad():
        for row in tqdm(text_data, desc="Processing"):
            # NOTE: INTERESTING CHOICE
            text = truncate_string(row['post_title'], int(MAX_CHARS/6)) + ' ' + truncate_string(row['post_selftext'], int(MAX_CHARS/3)) + ' ' + row['comment_body']
            inputs = tokenizer(text, return_tensors='pt', max_length=MAX_LEN, truncation=True, padding='max_length')
            outputs = model(**inputs)
            embeddings.append(outputs.last_hidden_state.squeeze().tolist())

    return embeddings

def store_embeddings_with_labels(embeddings_with_labels, output_file):
    torch.save(embeddings_with_labels, output_file)

def read_embeddings_with_labels(embeddings_file):
    return torch.load(embeddings_file)

def generate_balanced_dataset_and_embeddings(labels, input_directory, output_file, num_samples_per_class, batch_size=10):
    data = {class_name: [] for class_name in labels}

    for class_name in labels:
        with open(os.path.join(input_directory, f"{class_name}.csv"), 'r', encoding='utf-8') as file:
            csv_reader = csv.DictReader(file)
            for row in csv_reader:
                data[class_name].append(row)

    embeddings_with_labels = []

    for label in labels:
        if label in data:
            samples = random.sample(data[label], min(num_samples_per_class, len(data[label])))
            for i in range(0, len(samples), batch_size):
                batch_samples = samples[i:i + batch_size]
                embeddings = convert_to_bert_embeddings(batch_samples)
                embeddings_with_labels.extend([(label, emb) for emb in embeddings])
    store_embeddings_with_labels(embeddings_with_labels, output_file)

# Example usage
if __name__ == "__main__":
    input_directory = "datasets/combined_classdata_csv"
    output_file = "datasets/bert/dataset_mini.pt"
    labels = ["michigan", "ohiostate", "georgia", "oklahoma", "texas", "floridastate", "oregon", "alabama", "notredame", "chaos"]  # Specify the list of labels
    one_label = ["michigan"]
    num_samples_per_class = 1  # Number of samples to take from each class
    batch_size = 1  # Adjust batch size based on memory constraints

    generate_balanced_dataset_and_embeddings(one_label, input_directory, output_file, num_samples_per_class, batch_size)
