import numpy as np
from models import *
import torch
from sklearn.metrics import recall_score, confusion_matrix
from torchvision import datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib
from datasets import load_dataset
from utils import *
import seaborn as sns
matplotlib.use('TkAgg')


def test_minst(train_loader):

    # Get one batch
    image , label = next(iter(train_loader))


    # get one image from the batch
    x = image
    img = np.squeeze(image[0])

    img = img.permute(1,2,0)
    plt.imshow(img)
    plt.show()


    model = Autoencoder_CC_cifar(latent_dim=128)
    model.load_state_dict(torch.load("autoencoder_cifar_final_128.pt"))

    #x = x.view(x.size(0), -1)
    recon , latent = model(x)

    recon = recon[0]
    recon = recon.permute(1,2,0)
    recon = recon.detach().cpu().numpy()
    plt.imshow(recon)
    #plt.imshow(recon.detach().cpu().view(28,28),cmap='gray')
    plt.show()


    model_linear = nn.Sequential(
                    nn.Linear(16, 64),
                    nn.ReLU(),
                    nn.Linear(64, 10),
                    nn.LogSoftmax(dim=1)
            )
    model_linear.load_state_dict(torch.load("linear_cnn.pt"))
    y_pred = model_linear(latent).argmax(dim=1)
    print(y_pred)
    print(label)


def test_merc():
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    ds = load_dataset("blanchon/UC_Merced")
    class UCMercedDataset(torch.utils.data.Dataset):
        def __init__(self, hf_dataset, transform=None):
            self.data = hf_dataset
            self.transform = transform

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            image = self.data[idx]['image']
            label = self.data[idx]['label']  # not really used for autoencoder
            if self.transform:
                image = self.transform(image)
            return image, label
    train_data = UCMercedDataset(ds['train'], transform=transform)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=80, shuffle=True)

    # Get one batch
    image, label = next(iter(train_loader))

    # get one image from the batch
    x = image
    img = np.squeeze(image[0])
    img = img.permute(1,2,0)

    plt.subplot(1,2,1)
    plt.imshow(img)

    model = Autoencoder_CNN_RGB(latent_dim=256*4)
    model.load_state_dict(torch.load("autoencoder_merc_1024_L2.pt"))

    # x = x.view(x.size(0), -1)
    recon, latent = model(x)
    #recon = recon.permute(1,2,0)
    img = recon[0].detach().cpu()
    img = img.permute(1,2,0)

    plt.subplot(1,2,2)
    plt.imshow(img)
    plt.show()

    model_linear = nn.Sequential(
        nn.Linear(16, 64),
        nn.ReLU(),
        nn.Linear(64, 10),
        nn.LogSoftmax(dim=1)
    )
    model_linear.load_state_dict(torch.load("linear_cnn.pt"))
    y_pred = model_linear(latent).argmax(dim=1)
    print(y_pred)
    print(label)

def test_merc_iterative(latent_space, test_loader):
    model = nn.Sequential(
        nn.Linear(latent_space, 256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, 21),
        nn.LogSoftmax(dim=1)
    )
    model.load_state_dict(torch.load( 'linear_classifier_merced_'+str(latent_space)+'.pt'))

    autoencoder = Autoencoder_CNN_RGB_claude(latent_dim=latent_space)
    autoencoder.load_state_dict(torch.load('autoencoder_final_merc_' + str(latent_space)+'.pt'))

    model.eval()
    autoencoder.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for x,y in test_loader:
            _ , z = autoencoder(x)
            y_mod = model(z).argmax(dim=1)

            all_preds.append(y_mod)
            all_labels.append(y)
    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    conf_matrix = torch.zeros(21,21,dtype=torch.int32)
    for t,p in zip(all_labels, all_preds):
        conf_matrix[t,p] += 1
    #print('#### Confusion Matrix ####')
    #print(conf_matrix)

    accuracy = (all_preds == all_labels).sum().item() / len(all_labels)
    print("latent dimension: " + str(latent_space) + "accuracy: " + str(accuracy))

def test_minst_iterative(latent_space, test_loader):
    model = nn.Sequential(
        nn.Linear(latent_space, 128),
        nn.ReLU(),
        nn.Linear(128, 10),
        nn.LogSoftmax(dim=1)
    )
    model.load_state_dict(torch.load('Linear_CNN_cifar_final-'+str(latent_space)+'.pt'))


    autoencoder = Autoencoder_CC_cifar(latent_dim=latent_space)
    autoencoder.load_state_dict(torch.load('autoencoder_cifar_final_' + str(latent_space) + '.pt'))


    model.eval()
    autoencoder.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for x, y in test_loader:
            _, z = autoencoder(x)
            y_mod = model(z).argmax(dim=1)

            all_preds.append(y_mod)
            all_labels.append(y)
    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    conf_matrix = torch.zeros(10, 10, dtype=torch.int32)
    for t, p in zip(all_labels, all_preds):
        conf_matrix[t, p] += 1
    # print('#### Confusion Matrix ####')
    # print(conf_matrix)

    accuracy = (all_preds == all_labels).sum().item() / len(all_labels)

    tp = ((all_preds == 1) & (all_labels == 1)).sum().item()
    fn = ((all_preds == 0) & (all_labels == 1)).sum().item()

    recall = tp / (tp + fn + 1e-8)
    print("latent dimension: " + str(latent_space) + "accuracy: " + str(accuracy) +"   recall: "+str(recall))
    return accuracy, recall, all_preds, all_labels



if "__main__" == __name__:
    batch_size = 500
    #train, test = prep_dataLoadert("blanchon/UC_Merced", batch_size)
    #train, test =load_minst(batch_size)
    train, test = load_cifar10(batch_size)


    latent_sizes = [128, 256 , 256*2 , 256 * 4]
    #latent_sizes = [1,2,3, 4,5, 6, 10,20,40]
    latent_sizes = [4,8, 16,32, 64,128,256,512, 1024, 1024*2, 1024*4]
    latent_sizes = [1024]
    #latent_sizes=[256]
    #latent_sizes = [256]
    precision_list = []
    recall_list = []
    pred_list = []
    label_list = []
    for laten_dim in latent_sizes:
        #test_merc_iterative(laten_dim, test)
        acc, rec, pred, label = test_minst_iterative(laten_dim,test)
        precision_list.append(acc)
        recall_list.append(rec)
        pred_list.append(pred)
        label_list.append(label)


    plt.figure(figsize=(6, 4))
    plt.semilogx(latent_sizes, precision_list, marker='o', linewidth=1.2)

    # Labels and title
    plt.xlabel("Latent Size (log scale)")
    plt.ylabel("Precision")
    plt.title("Model Precision vs Latent Size")

    # Grid (light, typical for scientific plots)
    plt.grid(True, which="both", linestyle="--", linewidth=0.4, alpha=0.6)

    # Remove top/right spines for a cleaner scientific look
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Tight layout improves spacing for papers
    plt.tight_layout()

    plt.show()

    plt.figure(figsize=(6, 4))
    plt.semilogx(latent_sizes, recall_list, marker='o', linewidth=1.2)

    # Labels and title
    plt.xlabel("Latent Size (log scale)")
    plt.ylabel("Recall")
    plt.title("Model Recall vs Latent Size")

    # Grid (light, typical for scientific plots)
    plt.grid(True, which="both", linestyle="--", linewidth=0.4, alpha=0.6)

    # Remove top/right spines for a cleaner scientific look
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Tight layout improves spacing for papers
    plt.tight_layout()

    plt.show()

    class_names = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]


    pred_array = torch.cat(pred_list).numpy()
    label_array = torch.cat(label_list).numpy()

    cm = confusion_matrix(pred_array, label_array)
    plt.figure(figsize=(8,8), constrained_layout=True)
    sns.heatmap(cm , annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names , yticklabels=class_names)

    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)

    plt.xlabel("Predicted")
    plt.ylabel("Ground Truth")
    plt.title("Confusion Matrix: Linear Classification with autoencoder")
    plt.show()

"""
latent dimension: 64accuracy: 0.4365
latent dimension: 128accuracy: 0.4593
latent dimension: 256accuracy: 0.4672
latent dimension: 512accuracy: 0.4954
latent dimension: 1024accuracy: 0.5058
latent dimension: 2048accuracy: 0.5077
latent dimension: 4096accuracy: 0.5058








"""