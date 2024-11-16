import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet18
from PIL import Image
import numpy as np

class AIImageDetector:
    def __init__(self):
        # Init model
        self.model = resnet18(pretrained=True)
        
        # final layer -> binary classification
        num_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        
        # image transformations
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(), # creating tensors
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
    
    def train(self, train_dataloader, num_epochs=10):
        # loss function and optimizer
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        
        self.model.train()
        for epoch in range(num_epochs):
            running_loss = 0.0
            correct = 0
            total = 0
            
            for images, labels in train_dataloader:
                images = images.to(self.device)
                labels = labels.float().to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs.squeeze(), labels)
                
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                predicted = (outputs.squeeze() > 0.5).float()
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            
            epoch_loss = running_loss / len(train_dataloader)
            accuracy = 100 * correct / total
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.2f}%')
    
    def predict(self, image_path):
        # evaluate
        self.model.eval()
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.model(image)
            probability = output.squeeze().item()
            prediction = "AI-Generated" if probability > 0.49 else "Real"
            confidence = max(probability, 1 - probability) * 100
            
        return prediction, confidence
    
    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
    
    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))

# usage
def main():
    # Init detector
    detector = AIImageDetector()
    # input image
    image_path = "Deepfake_Detector/imgs/7.jpg"
    #predict
    prediction, confidence = detector.predict(image_path)
    # print
    print(f"Prediction: {prediction}")
    print(f"Confidence: {confidence:.2f}%")

if __name__ == "__main__":
    main()
