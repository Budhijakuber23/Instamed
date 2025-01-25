'''
In order to fienn tune the clip model, download the given libs in the python env

pip install openai-clip
pip install datasets
pip install torch
pip install tqdm

'''

# Load the dataset
from datasets import load_dataset
ds = load_dataset('roco-dataset/data/train/radiology')

# loading original clip

import clip
import torch
model, preprocess = clip.load("ViT-B/32", jit=False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Custom dataset class
from torchvision import transforms
from torch.utils.data import Dataset

# Define a custom dataset class
class CMDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image = item['image']
        text = item['caption']
        label = item['keywords'][0] + "_" + item['keywords'][1] + "_" + item['keywords'][2]
        return self.transform(image), label
    

from torch.utils.data import DataLoader
train_loader = DataLoader(CMDataset(ds), batch_size=32, shuffle=True)

import torch.nn as nn

# Modify the model to include a classifier for subcategories
class CLIPFineTuner(nn.Module):
    def __init__(self, model, num_classes):
        super(CLIPFineTuner, self).__init__()
        self.model = model
        self.classifier = nn.Linear(model.visual.output_dim, num_classes)
    
    def forward(self, x):
        with torch.no_grad():
            features = self.model.encode_image(x).float()  # Convert to float32
        return self.classifier(features)
    

model_ft = CLIPFineTuner()

import torch.optim as optim

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model_ft.classifier.parameters(), lr=1e-4)

from tqdm import tqdm

# Number of epochs for training
num_epochs = 5
for epoch in range(num_epochs):
    model_ft.train() 
    running_loss = 0.0  
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}, Loss: 0.0000") 
    
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()  
        outputs = model_ft(images)  
        loss = criterion(outputs, labels)  
        loss.backward()  
        optimizer.step() 
        
        running_loss += loss.item() 
        pbar.set_description(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader):.4f}") 

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')  
    
torch.save(model_ft.state_dict(), 'clip-vit-base-CM')  