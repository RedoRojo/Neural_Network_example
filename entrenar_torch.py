import torch

from torchvision import transforms
from demo_torch import H5Data
from torch import nn, optim
# cargamos data
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0,5), (0,5))])

carga_entrenamiento = torch.utils.data.DataLoader(H5Data("digitos.h5"), batch_size = 64, shuffle=True)
# print(carga_entrenamiento.dataset.data.shape)
carga_test = torch.utils.data.DataLoader(H5Data("digitos_test.h5"), batch_size = 64, shuffle=True)

capa_entrada = 784
capas_ocultas = [128, 64]
capa_salida = 10

# el Relu de aa abajo es la funcion de la unidad de acticacion 
modelo = nn.Sequential(nn.Linear(capa_entrada, capas_ocultas[0]), nn.ReLU(),
                        nn.Linear(capas_ocultas[0], capas_ocultas[1]), nn.ReLU(),
                        nn.Linear(capas_ocultas[1], capa_salida), nn.LogSoftmax(dim=1))

j = nn.CrossEntropyLoss()

optimizador = optim.Adam(modelo.parameters(), lr = 0.003) 

epochs = 1
print(type(carga_entrenamiento))
for e in range(epochs):
    costo = 0
    for imagen, etiqueta in carga_entrenamiento:
        imagen = imagen.view(imagen.shape[0], -1)

        optimizador.zero_grad()

        h = modelo(imagen.float())
        error = j(h, etiqueta.long())
        error.backward()
        optimizador.step()
        costo += error.item()

print(optimizador)
torch.save(modelo, "modelo.pt")