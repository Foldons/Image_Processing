from models import *
import torch
from torchvision import datasets
import torchvision.transforms as transforms




def autoencoder_train_minst(train, input_latent):
        num_epochs = 20
        autoencoder = Autoencoder_linear(latent_dim=input_latent)
        optimizer = torch.optim.Adam(autoencoder.parameters(), lr=1e-3)
        criterion = nn.MSELoss()  # map reconstruction loss

        for epoch in range(num_epochs):
                epoch_loss = 0
                for data in train:
                        x , _ = data
                        x = x.view(x.size(0),  -1)

                        optimizer.zero_grad()

                        recon_x, _ = autoencoder(x)
                        loss = criterion(recon_x, x)

                        loss.backward()
                        optimizer.step()
                        epoch_loss += loss.item()
                print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss :.6f}")
        torch.save(autoencoder.state_dict(), 'autoencoder_minst_final_'+str(input_latent)+'.pt')

def autoencoder_train_merc(train_loader , input_latent, device):
        num_epochs = 15


        autoencoder = Autoencoder_CNN_RGB_claude(latent_dim=input_latent)
        autoencoder.to(device)
        optimizer = torch.optim.Adam(autoencoder.parameters(), lr=1e-3)
        criterion = nn.MSELoss()  # map reconstruction loss

        for epoch in range(num_epochs):
                epoch_loss = 0
                for data in train_loader:
                        x , _ = data
                        x = x.to(device)


                        optimizer.zero_grad()

                        recon_x, _ = autoencoder(x)
                        loss = criterion(recon_x, x)

                        loss.backward()
                        optimizer.step()
                        epoch_loss += loss.item()
                print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss :.6f}")
        torch.save(autoencoder.state_dict(), 'autoencoder_final_merc_' + str(input_latent)+'.pt')


def autoencoder_train_CNN_minst(train , input_latent):
        num_epochs = 15
        autoencoder = Autoencoder_CC_cifar(latent_dim=input_latent)
        optimizer = torch.optim.Adam(autoencoder.parameters(), lr=1e-3)
        criterion = nn.MSELoss()  # map reconstruction loss

        for epoch in range(num_epochs):
                epoch_loss = 0
                for data in train:
                        x , _ = data
                        #x = x.view(x.size(0),  -1)

                        optimizer.zero_grad()

                        recon_x, latent = autoencoder(x)
                        loss = criterion(recon_x, x) #+ 1e-4 * torch.sum(latent**2) # kill this if necessary

                        loss.backward()
                        optimizer.step()
                        epoch_loss += loss.item()
                print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss :.6f}")
        torch.save(autoencoder.state_dict(), 'autoencoder_cifar_final_'+str(input_latent)+'.pt')


def classifer_train_minst_cnn(train , input_latent):
        model = nn.Sequential(
                nn.Linear(input_latent, 128),
                nn.ReLU(),
                nn.Linear(128, 10),
                nn.LogSoftmax(dim=1)
        )
        autoencoder = Autoencoder_CC_cifar(latent_dim=input_latent)
        autoencoder.load_state_dict(torch.load('autoencoder_cifar_final_'+str(input_latent)+'.pt'))



        num_epochs = 15
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.NLLLoss()  # map reconstruction loss

        for epoch in range(num_epochs):
                epoch_loss = 0
                for data in train:
                        x, y = data
                        #x = x.view(x.size(0), -1)

                        optimizer.zero_grad()

                        _ , latent = autoencoder(x)
                        y_mod = model(latent)
                        #y_mod = y_mod.argmax(dim=1)
                        loss = criterion(y_mod , y)

                        loss.backward()
                        optimizer.step()
                        epoch_loss += loss.item()
                print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss :.6f}")
        torch.save(model.state_dict(), 'Linear_CNN_cifar_final-'+str(input_latent)+'.pt')




if "__main__" == __name__:
        input_latent = 256*4
        #autoencoder_train_CNN(input_latent)
        #classifer_train(input_latent)
        autoencoder_train_merc(input_latent)