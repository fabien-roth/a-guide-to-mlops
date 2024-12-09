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
from onnxruntime.quantization import quantize_static, QuantType
from onnxruntime.transformers.float16 import convert_float_to_float16
import numpy as np
import os
import bentoml
import tensorflow as tf

# Ajouter le répertoire principal du projet
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from a_guide_to_mlops.utils.config import PREPARED_DATA_DIR, QAT_MODEL_FLOAT16_DIR
from a_guide_to_mlops.utils.config_loader import load_config
from a_guide_to_mlops.utils.seed import set_seed
from a_guide_to_mlops.model.model_builder import get_model

# Define a CNN PyTorch model
class CNN_Model(nn.Module):
    def __init__(self, input_channels, num_classes):
        super(CNN_Model, self).__init__()
        self.quant = quantization.QuantStub()  # Quantization layer
        self.conv = nn.Conv2d(input_channels, 32, kernel_size=3, stride=2)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()  # Permet de simplifier la mise à plat
        self.fc = None  # La couche fully connected sera définie dynamiquement
        self.dequant = quantization.DeQuantStub()  # Dequantization layer
        self.num_classes = num_classes

    def forward(self, x):
        x = self.quant(x)  # Quantization
        x = self.relu(self.conv(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool2(x)
        if self.fc is None:
            num_features = x.size(1) * x.size(2) * x.size(3)
            self.fc = nn.Linear(num_features, self.num_classes).to(x.device, dtype=x.dtype)
        x = x.reshape(x.size(0), -1)
        x = self.flatten(x)
        x = self.fc(x)
        x = self.dequant(x)  # Dequantization
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
    model_folder = QAT_MODEL_FLOAT16_DIR
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
    ds_test_tf = tf.data.Dataset.load(str(prepared_dataset_folder / "test"))

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
    model.qconfig = quantization.get_default_qat_qconfig("fbgemm")  # Configure QAT
    #quantization.prepare_qat(model, inplace=True)
    quantization.prepare_qat(model)
    
    # Define optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=train_params["lr"])
    criterion = nn.CrossEntropyLoss()

    # Train the model
    print("Starting QAT training...", flush=True)
    model.train()
    EPOCHS = 100
    #EPOCHS = train_params["epochs"]:
    for epoch in range(EPOCHS):
        running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0

        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
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
    model.eval()
    model = quantization.convert(model, inplace=True)
    
    # Vérifier le type des poids après la conversion
    for name, param in model.named_parameters():
        print(f"Paramètre: {name}, Type: {param.dtype}")

    # Hook pour capturer les types d'activations
    activation_dtype = {}

    def activation_hook(module, input, output):
        activation_dtype[module] = output.dtype

    # Ajouter des hooks à chaque module
    hooks = []
    for name, module in model.named_modules():
        hooks.append(module.register_forward_hook(activation_hook))

    # Effectuer un passage en avant
    with torch.no_grad():
        dummy_input = torch.randn(1, input_channels, *image_size, dtype=torch.float32)
        model(dummy_input)

    # Afficher les types d'activations capturés
    for module, dtype in activation_dtype.items():
        print(f"Module: {module}, Activation type: {dtype}")

    # Retirer les hooks après inspection
    for hook in hooks:
        hook.remove()
    
    

    # Save the quantized model
    model_path = model_folder / "celestial_bodies_classifier_qat.pt"
    torch.save(model.state_dict(), model_path)
    print(f"Quantized PyTorch model saved at {model_path}", flush=True)

    # Convert to ONNX format
    print("Converting model to ONNX format...", flush=True)
    onnx_path = model_folder / "celestial_bodies_classifier_qat_float_16.onnx"
    dummy_input = torch.randn(1, input_channels, *image_size, dtype=torch.float32)
    print(f"dummy_input dtype: {dummy_input.dtype}")
    print(model)
    torch.onnx.export(
        model, dummy_input, str(onnx_path),
        input_names=["input"], output_names=["output"],
        opset_version=13
    )
    print(f"ONNX model saved at {onnx_path}", flush=True)

    # Quantize the ONNX model to float16
    print("Applying float16 quantization to ONNX model...", flush=True)
    quantized_onnx_path = model_folder / "celestial_bodies_classifier_qat_float16_quantized.onnx"

    # Convert the model to float16
    model_float16=convert_float_to_float16(
        model=str(onnx_path),
    )
    onnx.save(model_float16, str(quantized_onnx_path))

    print(f"Float16 quantized ONNX model saved at {quantized_onnx_path}", flush=True)

    # Save quantized ONNX model with BentoML
    print("Saving ONNX model with BentoML...", flush=True)
    quantized_model = onnx.load(str(quantized_onnx_path))
    bentoml_model_name = "celestial_bodies_classifier_qat_float16"
    bentoml.onnx.save_model(
        bentoml_model_name,
        quantized_model,
        signatures={"run": {"batchable": True}},
        labels={"model": "quantized", "framework": "onnx"},
        metadata={"description": "Quantized ONNX model of celestial bodies classifier"}
    )
    print("Quantized ONNX model successfully saved to BentoML store.", flush=True)

if __name__ == "__main__":
    main()
