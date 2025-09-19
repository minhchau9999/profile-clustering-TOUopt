import requests
import zipfile
import os

def download_uci_energy_data():
    """Download the UCI energy consumption dataset"""
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00235/household_power_consumption.zip"
    
    print("Downloading UCI Individual Household Electric Power Consumption dataset...")
    
    # Download the file
    response = requests.get(url)
    with open("household_power_consumption.zip", "wb") as f:
        f.write(response.content)
    
    # Extract the file
    with zipfile.ZipFile("household_power_consumption.zip", 'r') as zip_ref:
        zip_ref.extractall(".")
    
    print("Dataset downloaded and extracted successfully!")
    print("File: household_power_consumption.txt")

if __name__ == "__main__":
    download_uci_energy_data()