from torch import nn 
from torchsummary import summary


class CNNNetwork(nn.Module):

    def __init__(self):
        super().__init__()
        # 4 conv blocks / flatten / linear / softmax
        #i am adding this comment to make a change in the code so i can a commit this new change
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=2321, 
                out_channels=16,
                kernel_size=3, 
                stride=1,
                padding=2
            ), 
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size=2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=16, 
                out_channels=32,
                kernel_size=3, 
                stride=1,
                padding=2
            ), 
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size=2)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=32, 
                out_channels=64,
                kernel_size=3, 
                stride=1,
                padding=2
            ), 
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size=2)
        )
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(
                in_channels=64, 
                out_channels=128,
                kernel_size=3, 
                stride=1,
                padding=2
            ), 
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size=2)
        )

        self.flatten = nn.Flatten()
        self.linear = nn.Linear(9600, 3)
        self.softmax = nn.Softmax(dim = 1)
    
    def forward(self, input_data):
        x = self.conv1(input_data)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        x = self.flatten(x)
        
        logists = self.linear(x)
        predictions = self.softmax(logists)

        return predictions
    

class FeedForwardNet(nn.Module): 
    def __init__(self):
        super().__init__() #call the constructor of the father class (base class)
        #store all the different layers for this

        self.flatten = nn.Flatten() # the first layer, that comes from torch as well
        self.dense_layers = nn.Sequential(
            #sequential if a facility that comes with torch, allows to pack together multiply layers sequentiatly

            nn.Linear(23133, 6912), 
            nn.ReLU(), # activation function
            nn.Linear(6912, 3456),
            nn.ReLU(),
            nn.Linear(3456, 1728), 
            nn.ReLU(),
            nn.Linear(1728, 864), 
            nn.ReLU(),
            nn.Linear(864, 432), 
            nn.ReLU(),
            nn.Linear(432, 4), 

        )
        self.softmax = nn.Softmax(dim = 1) # this will normalize the outputs of the last layer 

    def forward(self, input_data): 
        # it will allow us to tell pytorch how to proccess the data
        # how to manupulate the data
        flattened_data = self.flatten(input_data)
        logits = self.dense_layers(flattened_data)
        predictions = self.softmax(logits)
        return predictions



if __name__ == "__main__": 

    cnn = FeedForwardNet()
    summary(cnn.cuda(), (1, 64, 216))