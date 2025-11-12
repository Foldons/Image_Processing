from train import *
from utils import *
from test_models import *

batch_size = 512
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")


#train, test = prep_dataLoadert("blanchon/UC_Merced", batch_size)
#train, test = load_minst(batch_size)
train, test = load_cifar10(batch_size)


latent_sizes = [256]
latent_sizes = [ 4096*2 , 4096*4]
for latent_size in latent_sizes:
    print(latent_size)
    #autoencoder_train_merc(train , latent_size, device)
    #classifer_train_mercd(train, latent_size, device)

    autoencoder_train_CNN_minst(train,latent_size)
    classifer_train_minst_cnn(train, latent_size)

for latent_size in latent_sizes:
    test_merc_iterative(latent_size, test)

