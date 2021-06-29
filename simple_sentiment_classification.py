import sentiment_utils as sent
from sentiment_utils import MovieReviews
from torch.utils.data import DataLoader
import torch
from torchtext.data import get_tokenizer
import time
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data.dataset import random_split
import datetime
import os
import time 
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device for Torch")


movie_train = MovieReviews("./movie_reviews/", train=True, transform=None)
# movie_test = MovieReviews("./movie_reviews/", train=False, transform=None)

sentiment_map = {
    0: "negative",
    1: "somewhat negative",
    2: "neutral",
    3: "somewhat positive",
    4: "positive"
}

def yield_tokens(data_iter):
    for text, _ in data_iter:
        yield tokenizer(text)

tokenizer = get_tokenizer('basic_english')
print("Building Tokenized Vocab")
vocab = build_vocab_from_iterator(yield_tokens(movie_train))

text_pipeline = lambda x: vocab.lookup_indices(tokenizer(x))
label_pipeline = lambda x: int(x)


def collate_batch(batch):
    label_list, text_list, offsets = [], [], [0]
    for (_text, _label) in batch:
        label_list.append(label_pipeline(_label))
        processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int64).to(device)
        text_list.append(processed_text)
        offsets.append(processed_text.size(0))
    label_list = torch.tensor(label_list, dtype=torch.int64).to(device)
    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0).to(device)
    text_list = torch.cat(text_list).to(device)
    return label_list.to(device), text_list.to(device), offsets.to(device)


num_classes = len(sentiment_map.keys())
vocab_size = len(vocab)
embed_size = 64
model = sent.TextClassificationModel(vocab_size, embed_size, num_classes).to(device)
#model = sent.TextLSTM(vocab_size, embed_size, num_classes, 3).to(device)


def train(dataloader):
    model.train()
    total_acc, total_count = 0, 0
    log_interval = 500
    start_time = time.time()
    for idx, (label, text, offsets) in enumerate(dataloader):
        optimizer.zero_grad()
        predited_label = model(text, offsets)
        loss = criterion(predited_label, label)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()
        total_acc += (predited_label.argmax(1) == label).sum().item()
        total_count += label.size(0)
        if idx % log_interval == 0 and idx > 0:
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches '
                  '| accuracy {:8.3f}'.format(epoch, idx, len(dataloader),
                                              total_acc/total_count))
            total_acc, total_count = 0, 0
            start_time = time.time()

def evaluate(dataloader):
    model.eval()
    total_acc, total_count = 0, 0

    with torch.no_grad():
        for idx, (label, text, offsets) in enumerate(dataloader):
            predited_label = model(text, offsets)
            loss = criterion(predited_label, label)
            total_acc += (predited_label.argmax(1) == label).sum().item()
            total_count += label.size(0)
    return total_acc/total_count


# Hyperparameters
EPOCHS = 25 # epoch
LR = 1.0  # learning rate
BATCH_SIZE = 64 # batch size for training

num_train = int(len(movie_train) * 0.95)
split_train_, split_valid_ = \
    random_split(movie_train, [num_train, len(movie_train) - num_train])

train_dataloader = DataLoader(split_train_, batch_size=BATCH_SIZE,
                              shuffle=True, collate_fn=collate_batch)
valid_dataloader = DataLoader(split_valid_, batch_size=BATCH_SIZE,
                              shuffle=True, collate_fn=collate_batch)


train_dataloader = sent.DeviceDataLoader(train_dataloader, device=device)
valid_dataloader = sent.DeviceDataLoader(valid_dataloader, device=device)


criterion = torch.nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.1)
total_accu = None


print("Begin Training\n", "*"*75)
total_start_time = time.time()
for epoch in range(1, EPOCHS + 1):
    epoch_start_time = time.time()
    train(train_dataloader)
    accu_val = evaluate(valid_dataloader)
    if total_accu is not None and total_accu > accu_val:
        scheduler.step()
    else:
        total_accu = accu_val
    print('-' * 59)
    print('| end of epoch {:3d} | time: {:5.2f}s | '
          'valid accuracy {:8.3f} '.format(epoch,
                                           time.time() - epoch_start_time,
                                           accu_val))
    print('-' * 59)
print(f"Total Training Time: {(time.time() - total_start_time) / 60:5.2f}mins")

print("Saving the model")
now = datetime.datetime.now()
model_name = "".join(["simple_sent_", str(now.year), str(now.month), 
            str(now.day), str(now.hour), str(now.minute), str(now.second)])

if not os.path.isdir('./models'):
    os.mkdir('./models')
torch.save(model, "./models/" + model_name + ".pt")


def predict(text, text_pipeline):
    with torch.no_grad():
        text = torch.tensor(text_pipeline(text)).long()
        output = model(text, torch.tensor([0]).long())
        return output.argmax(1).item()

model.to("cpu")

print("Classifying Training Data..")
true_labels = []
pred_labels = []
for text, label in movie_train:
    true_labels.append(label)
    prediction = predict(text, text_pipeline)
    pred_labels.append(prediction)

print("Writing Prediction .csv..")
pred_df = pd.DataFrame(zip(true_labels, pred_labels), columns=["Trues", "Prediction"])
pred_df.to_csv("./results/" + model_name + "_predictions_trues.csv")