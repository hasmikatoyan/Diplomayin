import torch
import torch.nn as nn
import torch.optim as optim
import gzip
import _pickle as cPickle
from torch.utils.data import DataLoader, TensorDataset


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_data_shared():
    with gzip.open('mnist.pkl.gz', 'rb') as f:
        tr_d, va_d, te_d = cPickle.load(f, encoding='latin1')
    
    def to_tensor(data):
        x = torch.tensor(data[0]).view(-1, 1, 28, 28).float()
        y = torch.tensor(data[1]).long()
        return x.to(device), y.to(device)

    return to_tensor(tr_d), to_tensor(va_d), to_tensor(te_d)

class Network3Optimized(nn.Module):
    def __init__(self):
        super(Network3Optimized, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1, 20, kernel_size=5),
            nn.ReLU(),  
            nn.MaxPool2d(2, 2), 
     
            nn.Conv2d(20, 40, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), 
            
            nn.Flatten(),
            nn.Linear(40 * 4 * 4, 100),
            nn.ReLU(),
            nn.Linear(100, 10),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        return self.layers(x)

def train_network():
    train_x, train_y = load_data_shared()[0]
    val_x, val_y = load_data_shared()[1]
    
    
    train_loader = DataLoader(TensorDataset(train_x, train_y), batch_size=64, shuffle=True)
    
    model = Network3Optimized().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001) 
    criterion = nn.NLLLoss()

    print(f"Training started ...")
    for epoch in range(10): 
        model.train()
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

       
        model.eval()
        with torch.no_grad():
            v_outputs = model(val_x)
            accuracy = (v_outputs.argmax(dim=1) == val_y).float().mean()
        
        print(f"Epoch {epoch}: validation accuracy {accuracy*100:.2f}%")

if __name__ == "__main__":
    train_network()