import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms, models
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
import wandb
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.data = []
        self.labels = []
        
        class_dirs = os.listdir(root_dir)
        for label, class_dir in enumerate(class_dirs):
            class_path = os.path.join(root_dir, class_dir)
            if os.path.isdir(class_path):
                for img_file in os.listdir(class_path):
                    self.data.append(os.path.join(class_path, img_file))
                    self.labels.append(label)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data[2]
        label = self.labels[idx]
        
        img = Image.open(img_path)
        img_tensor = transforms.ToTensor()(img)
        if self.transform:
            img = self.transform(img)
        return img, label

class ContrastiveModel(nn.Module):
    def __init__(self, base_model, embedding_dim=128):
        super(ContrastiveModel, self).__init__()
        self.base_model = base_model
        self.embedding = nn.Linear(base_model.fc.in_features, embedding_dim)
        base_model.fc = nn.Identity()

    def forward(self, x):
        features = self.base_model(x)
        embeddings = F.normalize(self.embedding(features), dim=1)
        return embeddings

class InfoNCELoss(nn.Module):
    def __init__(self, temperature=0.05):
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        similarity_matrix = torch.matmul(features, features.T)
        labels = labels.contiguous().view(-1, 1)
        
        mask = torch.eq(labels, labels.T).float()
        logits = similarity_matrix / self.temperature

        exp_logits = torch.exp(logits) * mask
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True))
        return -log_prob.mean()

# Data loading
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset1 = CustomDataset(root_dir="/ocean/projects/cis220039p/pkachana/projects/disjoint_contrastive/data/cats_dogs", transform=transform)
train_dataset2 = CustomDataset(root_dir="/ocean/projects/cis220039p/pkachana/projects/disjoint_contrastive/data/horses_cows", transform=transform)

train_loader1 = DataLoader(train_dataset1, batch_size=4, shuffle=True)
train_loader2 = DataLoader(train_dataset2, batch_size=4, shuffle=True)

# Model and training setup
base_model = EncoderModel()
model = ContrastiveModel(base_model).cuda()
criterion = InfoNCELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-2)

wandb.init(project="disjoint_contrastive")

# Training loop
for step in tqdm(range(100_000)):
    model.train()
    # Train on first dataset
    images1, labels1 = next(iter(train_loader1))
    images1, labels1 = images1.cuda(), labels1.cuda()
    
    embeddings1 = model(images1)
    loss1 = criterion(embeddings1, labels1)

    optimizer.zero_grad()
    loss1.backward()
    optimizer.step()

    wandb.log({"loss1_cat_dog": loss1.item()})

    # Train on second dataset
    images2, labels2 = next(iter(train_loader2))
    images2, labels2 = images2.cuda(), labels2.cuda()

    embeddings2 = model(images2)
    loss2 = criterion(embeddings2, labels2)

    optimizer.zero_grad()
    loss2.backward()
    optimizer.step()

    wandb.log({"loss2_horse_cow": loss2.item()})

    if step % 1000 == 0:
        torch.save(model.state_dict(), f"/ocean/projects/cis220039p/pkachana/projects/disjoint_contrastive/checkpoints/model_{step}.pt")
        breakpoint()
        print("####################")
        print("Labels1:", labels1)
        similarity_matrix1 = torch.matmul(embeddings1, embeddings1.T)
        print("Similarity matrix1:", similarity_matrix1)
        similarity_image1 = wandb.Image((similarity_matrix1 * 255).detach().cpu().numpy().astype("uint8"))

        print("Labels2:", labels2)
        similarity_matrix2 = torch.matmul(embeddings2, embeddings2.T)
        print("Similarity matrix2:", similarity_matrix2)
        similarity_image2 = wandb.Image((similarity_matrix2 * 255).detach().cpu().numpy().astype("uint8"))

        cross_similarity_matrix = torch.matmul(embeddings1, embeddings2.T)
        print("Cross similarity matrix:", cross_similarity_matrix)
        cross_similarity_image = wandb.Image((cross_similarity_matrix * 255).detach().cpu().numpy().astype("uint8"))

        wandb.log({"similarity_matrix1": similarity_image1, "similarity_matrix2": similarity_image2, "cross_similarity_matrix": cross_similarity_image})


    # print(f"Epoch [{epoch + 1}/10], Loss: {total_loss / len(train_loader):.4f}")
