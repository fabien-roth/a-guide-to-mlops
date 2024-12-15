# Import statements
import sys
from pathlib import Path
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torch.ao.quantization as quantization
from torch.utils.data import Dataset, DataLoader
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType
import numpy as np
import os
import bentoml
import tensorflow as tf

# Ajouter le répertoire principal du projet
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from a_guide_to_mlops.utils.config import PREPARED_DATA_DIR, QAT_MODEL_INTEGER_DIR
from a_guide_to_mlops.utils.config_loader import load_config
from a_guide_to_mlops.utils.seed import set_seed
from a_guide_to_mlops.model.model_builder import get_model

class CNN_Model(nn.Module):
    def __init__(self, input_channels, num_classes):
        super(CNN_Model, self).__init__()
        self.quant = quantization.QuantStub()  # Quantization layer
        
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=2)
        self.bn1 = torch.nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2)
        self.bn2 = torch.nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        """
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=2)
        self.bn3 = torch.nn.BatchNorm2d(256)
        self.relu3 = nn.ReLU()
        """
        self.flatten = nn.Flatten()  # Permet de simplifier la mise à plat
        self.fc = None  # La couche fully connected sera définie dynamiquement
        self.dequant = quantization.DeQuantStub()  # Dequantization layer
        
        self.num_classes = num_classes

    def forward(self, x):
        x = self.quant(x)  # Quantization
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        #x = self.relu3(self.bn3(self.conv3(x)))
        
        if self.fc is None:
            num_features = x.size(1) * x.size(2) * x.size(3)
            self.fc = nn.Linear(num_features, self.num_classes).to(x.device, dtype=x.dtype)
        
        x = x.reshape(x.size(0), -1)
        x = self.flatten(x)
        x = self.dequant(x)  # Dequantization
        x = self.fc(x)
        
        return x

class CustomDataset(Dataset):
    def __init__(self, X, y, transform=None):
        self.X = torch.tensor(X, dtype=torch.float32)  # Convertir en tensor
        self.y = torch.tensor(y, dtype=torch.long)     # Convertir en tensor
        self.transform = transform  # Optionnel : Transformations (normalisation, data augmentation, etc.)

    def __len__(self):
        return len(self.X)  # Taille du dataset

    def __getitem__(self, idx):
        image = self.X[idx]
        label = self.y[idx]

        # Appliquer les transformations si elles sont définies
        if self.transform:
            image = self.transform(image)

        return image, label

def main():
    print("Script started...", flush=True)

    # Load parameters
    config = load_config()
    prepare_params = config["prepare"]
    train_params = config["train"]

    prepared_dataset_folder = PREPARED_DATA_DIR
    model_folder = QAT_MODEL_INTEGER_DIR
    model_folder.mkdir(parents=True, exist_ok=True)

    image_size = prepare_params["image_size"]
    grayscale = prepare_params["grayscale"]
    input_channels = 1 if grayscale else 3
    num_classes = train_params["output_classes"]

    set_seed(train_params["seed"])

    # Load datasets
    print("Loading datasets...", flush=True)

    # Charger les datasets avec TensorFlow
    ds_train_tf = tf.data.Dataset.load(str(prepared_dataset_folder / "train"))
    #ds_test_tf = tf.data.Dataset.load(str(prepared_dataset_folder / "test"))

    # conversion tfwrapper => torchwrapper
    X_train_list = []
    y_train_list= []
    for features, labels in ds_train_tf:
        X_train_list.append(features.numpy())  
        y_train_list.append(labels.numpy())
    
    X_train_list = [arr.reshape(-1, arr.shape[1], arr.shape[2], 1) for arr in X_train_list]
    X_train = np.concatenate(X_train_list, axis=0)
    X_train = np.transpose(X_train, (0, 3, 1, 2))

    y_train = np.concatenate(y_train_list, axis=0)

    train_dataset = CustomDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # Define the PyTorch model
    print("Defining PyTorch model...", flush=True)
    model = CNN_Model(input_channels, num_classes)
    model.eval()
    model.qconfig = quantization.get_default_qat_qconfig("x86")  # Configure QAT

    model_fp32_fused = torch.ao.quantization.fuse_modules(model,
    [["conv1", "bn1", "relu1"], ["conv2", "bn2", "relu2"]])

    #quantization.prepare_qat(model, inplace=True)
    model_fp32_prepared = torch.ao.quantization.prepare_qat(model_fp32_fused.train())
    
    # Define optimizer and loss
    optimizer = optim.Adam(model_fp32_prepared.parameters(), lr=train_params["lr"])
    criterion = nn.CrossEntropyLoss()

    # Train the model
    print("Starting QAT training...", flush=True)
    model_fp32_prepared.train()
    EPOCHS = 100
    #EPOCHS = train_params["epochs"]:
    for epoch in range(EPOCHS):
        running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0

        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model_fp32_prepared(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_predictions += labels.size(0)

        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = correct_predictions / total_predictions
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")



    # Convert the model to a quantized version
    print("Converting model to quantized version...", flush=True)
    model_fp32_prepared.eval()
    model_int8 = torch.ao.quantization.convert(model_fp32_prepared)
    

    # Save the quantized model
    model_path = model_folder / "qat_int8.pt"
    torch.save(model_int8.state_dict(), model_path)
    # Save the scripted version
    model_scripted_path = model_folder / "qat_int8_scripted.pt"
    model_scripted = torch.jit.script(model_int8)
    model_scripted.save(model_scripted_path)
    print(f"Quantized PyTorch model saved at {model_path}", flush=True)

    model_int8.eval()

    # Save the trained model using BentoML with a unique name for tracking
    print("Saving the model using BentoML...", flush=True)
    bentoml_model_name = "QAT_int8_torch"  # Unique name to identify this variant
    bentoml.pytorch.save_model(
        bentoml_model_name,
        model_int8
    )

    # Export the BentoML model to the specified folder for deployment
    print("Exporting the model...", flush=True)
    bentoml.models.export_model(
        f"{bentoml_model_name}:latest",
        str(model_folder / "QAT_int8_torch.bentomodel")
    )

if __name__ == "__main__":
    main()
