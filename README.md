# Federated Learning Model

## Overview
This project implements a **Federated Learning Model** using flower library. The model enables decentralized learning across multiple edge devices.

## Installation
### Prerequisites
Ensure you have the following installed:
- Python 3.8+
- PyTorch  (choose one based on implementation)
- `flwr` (Flower Framework for Federated Learning)
- `numpy`, `pandas`, `matplotlib` (for data processing and visualization)

### Setup
Clone the repository:
```sh
git clone https://github.com/your-repo/federated-learning.git
cd federated-learning
```

## Usage
### Start the Server
```sh
python server.py
```

### Start Clients
Run the following command on each client device:
```sh
python client.py <DEVICE_ID>
```

### Monitor Training Progress
The training logs and performance metrics will be displayed appropriately.

## Configuration
Modify the `config.py` file to customize the following parameters:
- `num_rounds`: Number of federated training rounds.
- `num_clients`: Number of participating clients.
- `learning_rate`: Learning rate for local training.
- `batch_size`: Batch size for training on local devices.
- `and many more`

## Results
After training, results will be available in:
- `models/` - Saved model weights.
- `plots/` - Visualization of training progress.

## Contributing
1. Fork the repository.
2. Create a new branch (`feature-branch`).
3. Commit your changes.
4. Push to your branch and submit a pull request.

## License
This project is licensed under the MIT License.

## Contact
For questions or collaboration, contact: `mridulmaikhuri1234@gmail.com`

