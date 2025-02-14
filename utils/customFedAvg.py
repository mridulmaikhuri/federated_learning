from flwr.server.strategy import FedAvg
from config import history, update_history

class CustomFedAvg(FedAvg):
    def aggregate_fit(self, server_round, results, failures):
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)

        # Count the number of participating clients
        num_clients = len(results)

        # Extract metrics and find their average
        train_losses = [fit_res.metrics["train_loss"] for _, fit_res in results if "train_loss" in fit_res.metrics]
        train_accuracies = [fit_res.metrics["train_accuracy"] for _, fit_res in results if "train_accuracy" in fit_res.metrics]
        avg_train_loss = sum(train_losses) / len(train_losses) if train_losses else 0
        avg_train_accuracy = sum(train_accuracies) / len(train_accuracies) if train_accuracies else 0

        print(f"Training Round {server_round}\n No of Clients = {num_clients}\n Train Loss = {avg_train_loss}\n Train Acc = {avg_train_accuracy}")
        update_history(server_round, avg_train_loss, avg_train_accuracy, None, None)

        return aggregated_parameters, aggregated_metrics

    def aggregate_evaluate(self, server_round, results, failures):
        aggregated_loss, aggregated_metrics = super().aggregate_evaluate(server_round, results, failures)

        # Count the number of participating clients
        num_clients = len(results)

        # Extract metrics and find their average
        val_losses = [eval_res.metrics["val_loss"] for _, eval_res in results if "val_loss" in eval_res.metrics]
        val_accuracies = [eval_res.metrics["accuracy"] for _, eval_res in results if "accuracy" in eval_res.metrics]
        avg_val_loss = sum(val_losses) / len(val_losses) if val_losses else 0
        avg_val_accuracy = sum(val_accuracies) / len(val_accuracies) if val_accuracies else 0

        print(f"Validation Round {server_round}\n No of Clients = {num_clients}\n Val Loss = {avg_val_loss}\n Val Acc = {avg_val_accuracy}")
        update_history(server_round, None, None, avg_val_loss, avg_val_accuracy)

        return aggregated_loss, aggregated_metrics

if __name__ == "__main__":
    pass