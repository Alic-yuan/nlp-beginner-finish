import os
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from data import LCQMC_Dataset, load_embeddings
from utils import train, validate
from model import ESIM

def main(train_file, dev_file, vocab_file, target_dir,
         max_length=50,
         hidden_size=300,
         dropout=0.2,
         num_classes=2,
         epochs=50,
         batch_size=256,
         lr=0.0005,
         patience=5,
         max_grad_norm=10.0,
         gpu_index=0,
         checkpoint=None):
    device = torch.device("cuda:{}".format(gpu_index) if torch.cuda.is_available() else "cpu")
    print(20 * "=", " Preparing for training ", 20 * "=")
    # 保存模型的路径
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    # -------------------- Data loading ------------------- #
    print("\t* Loading training data...")
    train_data = LCQMC_Dataset(train_file, vocab_file, max_length)
    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
    print("\t* Loading validation data...")
    dev_data = LCQMC_Dataset(dev_file, vocab_file, max_length)
    dev_loader = DataLoader(dev_data, shuffle=True, batch_size=batch_size)
    # -------------------- Model definition ------------------- #
    print("\t* Building model...")
    # embeddings = load_embeddings(embeddings_file)
    model = ESIM(hidden_size, dropout=dropout,
                 num_labels=num_classes, device=device).to(device)
    # -------------------- Preparation for training  ------------------- #
    print('a')
    criterion = nn.CrossEntropyLoss()
    # 过滤出需要梯度更新的参数
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    print('b')
    # optimizer = optim.Adadelta(parameters, params["LEARNING_RATE"])
    optimizer = torch.optim.Adam(parameters, lr=lr)
    # optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max",
                                                           factor=0.85, patience=0)

    best_score = 0.0
    start_epoch = 1
    # Data for loss curves plot
    epochs_count = []
    train_losses = []
    valid_losses = []
    # Continuing training from a checkpoint if one was given as argument
    if checkpoint:
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint["epoch"] + 1
        best_score = checkpoint["best_score"]
        print("\t* Training will continue on existing model from epoch {}...".format(start_epoch))
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        epochs_count = checkpoint["epochs_count"]
        train_losses = checkpoint["train_losses"]
        valid_losses = checkpoint["valid_losses"]
    # Compute loss and accuracy before starting (or resuming) training.
    _, valid_loss, valid_accuracy, auc = validate(model, dev_loader, criterion)
    print("\t* Validation loss before training: {:.4f}, accuracy: {:.4f}%, auc: {:.4f}".format(valid_loss,
                                                                                               (valid_accuracy * 100),
                                                                                               auc))
    # -------------------- Training epochs ------------------- #
    print("\n", 20 * "=", "Training ESIM model on device: {}".format(device), 20 * "=")
    patience_counter = 0
    for epoch in range(start_epoch, epochs + 1):
        epochs_count.append(epoch)
        print("* Training epoch {}:".format(epoch))
        epoch_time, epoch_loss, epoch_accuracy = train(model, train_loader, optimizer,
                                                       criterion, epoch, max_grad_norm)
        train_losses.append(epoch_loss)
        print("-> Training time: {:.4f}s, loss = {:.4f}, accuracy: {:.4f}%"
              .format(epoch_time, epoch_loss, (epoch_accuracy * 100)))
        print("* Validation for epoch {}:".format(epoch))
        epoch_time, epoch_loss, epoch_accuracy, epoch_auc = validate(model, dev_loader, criterion)
        valid_losses.append(epoch_loss)
        print("-> Valid. time: {:.4f}s, loss: {:.4f}, accuracy: {:.4f}%, auc: {:.4f}\n"
              .format(epoch_time, epoch_loss, (epoch_accuracy * 100), epoch_auc))
        # Update the optimizer's learning rate with the scheduler.
        scheduler.step(epoch_accuracy)
        # Early stopping on validation accuracy.

        if epoch_accuracy < best_score:
            patience_counter += 1
        else:
            best_score = epoch_accuracy
            patience_counter = 0
            torch.save({"epoch": epoch,
                        "model": model.state_dict(),
                        "best_score": best_score,
                        "epochs_count": epochs_count,
                        "train_losses": train_losses,
                        "valid_losses": valid_losses},
                       os.path.join(target_dir, "best.pth.tar"))
            # Save the model at each epoch.
        torch.save({"epoch": epoch,
                    "model": model.state_dict(),
                    "best_score": best_score,
                    "optimizer": optimizer.state_dict(),
                    "epochs_count": epochs_count,
                    "train_losses": train_losses,
                    "valid_losses": valid_losses},
                   os.path.join(target_dir, "esim_{}.pth.tar".format(epoch)))

        if patience_counter >= patience:
            print("-> Early stopping: patience limit reached, stopping...")
            break


if __name__ == "__main__":
    main("atec_nlp_sim_train_all.csv", "dev.csv", "vocab.txt", "models")
