import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import flwr
from flwr.server import start_server
from config import DEVICE, history, SERVER_ADDRESS, fit_config, fit_metrics_aggregation_fn, fraction_fit, fraction_evaluate, min_fit_clients, min_evaluate_clients, min_available_clients
from utils.plot import plot_dashboard
from utils.customFedAvg import CustomFedAvg
from utils.model import Net, test
from utils.load_dataset import load_datasets

strategy = CustomFedAvg(
    fraction_fit=fraction_fit,
    fraction_evaluate=fraction_evaluate,
    min_fit_clients=min_fit_clients,
    min_evaluate_clients=min_evaluate_clients,
    min_available_clients=min_available_clients,
    on_fit_config_fn=fit_config,
    fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
)

if __name__ == "__main__":
    start_server(server_address=SERVER_ADDRESS, config=flwr.server.ServerConfig(num_rounds=5), strategy=strategy)
    _, _, testloader = load_datasets(0)
    loss, accuracy = test(Net().to(DEVICE), testloader)
    print(f"Final test Results\n Loss = {loss}\n Accuracy = {accuracy}")
    plot_dashboard(history)
