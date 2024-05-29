from predictor_model.config import config
from predictor_model.models import LSTM_model
from predictor_model.processing import dataloaders


model = LSTM_model.create_model()
train_dataloader = dataloaders.create_dataloader(train_data)


def train(model, dataloader, loss_fn, optimizer):
    model.train()
    model.to(config.DEVICE)
    train_loss = 0
    train_r2 = 0
    for text, label in dataloader:
        optimizer.zero_grad()
        label, text = label.to(config.DEVICE), text.to(config.DEVICE)
        y_pred = model(text)
        loss = loss_fn(y_pred.squeeze(), label.squeeze())
        train_loss += loss
        r2 = r2_score(y_pred.squeeze(), label.squeeze())
        train_r2 += r2
        loss.backward()

        optimizer.step()

    train_loss /= len(dataloader)
    train_r2 /= len(dataloader)
    print(f"Train Loss: {train_loss}, r-squared score: {train_r2}")


def eval(model, dataloader, loss_fn):
    model.eval()

    eval_loss = 0
    eval_r2 = 0
    eval_mse = 0
    with torch.inference_mode():
        for text, label in dataloader:
            label, text = label.to(device), text.to(device)
            y_pred = model(text)
            loss = loss_fn(y_pred.squeeze(), label.squeeze())
            eval_loss += loss
            r2 = r2_score(y_pred.squeeze(), label.squeeze())
            eval_r2 += r2

        eval_loss /= len(dataloader)
        eval_r2 /= len(dataloader)
        print(f"Test Loss: {eval_loss}, r-squared score: {eval_r2}")
