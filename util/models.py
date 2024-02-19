import torch
import torch.nn as nn
import torch.optim as optim

class LogisticRegression(nn.Module):
    def __init__(self, input_size):
        super(LogisticRegression, self).__init__()
        self.fc = nn.Linear(input_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc(x)
        x = self.sigmoid(x)
        return x

class DeepNeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DeepNeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x

# Training loop
def train(model, dataloader, criterion=nn.BCELoss(), n_epochs=10, lr=0.01, verbose=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)


    for epoch in range(1, n_epochs+1):
        optimizer = optim.Adam(model.parameters(), lr=lr*epoch)
        model.train()

        for x, y in dataloader:
            x, y = x.to(device), y.to(device)

            # Forward pass
            y_pred = model(x)
            loss = criterion(y_pred, y)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Print training statistics for each epoch
        if verbose:
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}")

    if verbose:
        print("Training complete")


def eval(model, dataloader):
    pass
    