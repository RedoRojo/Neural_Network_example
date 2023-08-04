import h5py 
from torch.utils.data import Dataset

class H5Data(Dataset): 

    def __init__(self, archivo, transform=None): #el transform es una funcion que en si es lo que queremos haacer con los datos,
        self.archivo = h5py.File(archivo,'r')
        self.etiquetas = self.archivo['y']
        self.data = self.archivo['X']
        self.transform = transform # si es que quisieramos hacer una normalizacion o algo por el estilo entonces usamos esta wea 

    def __getitem__(self, index):
        datum = self.data[index]
        if(self.transform is not None):
            datum = self.transform(datum)
        
        return datum, self.etiquetas[index]

    def __len__(self): 
        return len(self.etiquetas)
    
    def close(self):
        self.archivo.close()