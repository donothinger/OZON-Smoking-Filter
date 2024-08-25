from transformers import ViTFeatureExtractor, ViTModel, ViTForImageClassification, AdamW, get_linear_schedule_with_warmup
from PIL import Image
import requests
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch
import pandas as pd
import os
import sys
from datetime import datetime, timedelta

from dataset import SigDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

def print_one_line(s):
    time_string = datetime.now().strftime('%H:%M:%S')
    sys.stdout.write('\r' + time_string + ' ' + s)
    sys.stdout.flush()

BATCH_SIZE=38
LR=2e-5
EPOCHS=10
MODEL_SAVE_PATH = 'model/best_rest.pt'
PRETRAINED_PATH = 'model/pretrained'
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

feature_extractor = ViTFeatureExtractor.from_pretrained(PRETRAINED_PATH)
model = ViTForImageClassification.from_pretrained(PRETRAINED_PATH)
model.classifier = nn.Linear(1280, 2, bias=True).to(DEVICE)

df = pd.read_csv('data.csv')

X_train, X_test, y_train, y_test = train_test_split(df['image_name'].to_numpy(), df['label'].to_numpy(), test_size=0.2, random_state=42)

train_dataset = SigDataset(X_train, y_train, feature_extractor)
test_dataset = SigDataset(X_test, y_test, feature_extractor)

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True) 
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

optimizer = AdamW(model.parameters(), lr=LR)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(train_dataloader) * EPOCHS)
loss_fn = nn.CrossEntropyLoss().to(DEVICE)

best_F1 = 0
for epoch in range(EPOCHS):
    start_time = datetime.now()
    train_loss, batch_idx, total, correct = 0, 0, 0, 0
    correct_labels, predicted_labels = [], []

    model.train()
    for inputs, labels in train_dataloader:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

        outputs = model(inputs)

        loss = loss_fn(outputs.logits, labels)
        train_loss += loss.item()

        correct_labels += labels.squeeze().cpu()
        predicted_labels += outputs.logits.argmax(dim=1).cpu()

        total += labels.shape[0]
        correct += (outputs.logits.argmax(dim=1) == labels).sum().item()

        batch_idx += 1

        print_one_line('Epoch {} | Loss={: .4f} | Acc={: .2f}% ({}/{})| F1={: .2f}'.format(epoch,
                                                                        train_loss/batch_idx,
                                                                        100 * accuracy_score(correct_labels, predicted_labels), correct, total,
                                                                        f1_score(correct_labels, predicted_labels)))
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
    print('')
    val_loss, batch_idx, total, correct = 0, 0, 0, 0
    correct_labels, predicted_labels = [], []
    model.eval()
    with torch.no_grad():
        for inputs, labels in test_dataloader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            outputs = model(inputs)
            loss = loss_fn(outputs.logits, labels)
            val_loss += loss.item()

            correct_labels += labels.squeeze().cpu()
            predicted_labels += outputs.logits.argmax(dim=1).cpu()

            total += labels.shape[0]
            correct += (outputs.logits.argmax(dim=1) == labels).sum().item()

            batch_idx += 1
        val_acc = 100 * accuracy_score(correct_labels, predicted_labels)
        val_loss = val_loss/batch_idx
        val_f1 = f1_score(correct_labels, predicted_labels)

        print('Validation accurancy {:.4f}'.format(val_acc))
        print('Validation loss {:.4f}'.format(val_loss))
        print('Validation F1 {:.4f}'.format(val_f1))

    if val_f1 > best_F1:
        best_F1 = val_f1
        torch.save(model, MODEL_SAVE_PATH)

