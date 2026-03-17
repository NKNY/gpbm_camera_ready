"""Early stopping callback for training loops."""


class EarlyStopping:
    """Stop training when monitored metric stops improving.

    :param patience: Number of epochs to wait after last improvement (default: 10).
    :param min_delta: Minimum change to qualify as improvement (default: 0.0).
    :param monitor: Metric name to monitor, e.g. "loss" or "ndcg@5" (default: "loss").
    :param verbose: Print message when stopping (default: True).

    Usage:
        early_stop = EarlyStopping(patience=5, monitor="ndcg@5")
        for epoch in range(n_epochs):
            metrics = evaluate(model, val_loader)
            if early_stop(epoch, metrics):
                break
    """

    def __init__(self, patience=10, min_delta=0.0, monitor="loss", verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.monitor = monitor
        self.verbose = verbose
        self.higher_is_better = "loss" not in monitor
        self.best_val = float("-inf") if self.higher_is_better else float("inf")
        self.best_epoch = 0
        self.counter = 0
        self.should_stop = False

    def __call__(self, epoch, metrics):
        current = metrics[self.monitor]

        if self.higher_is_better:
            is_improvement = current > self.best_val + self.min_delta
        else:
            is_improvement = current < self.best_val - self.min_delta

        if is_improvement:
            self.best_val = current
            self.best_epoch = epoch
            self.counter = 0
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                if self.verbose:
                    print(
                        f"Early stopping at epoch {epoch + 1}. Best {self.monitor}: {self.best_val:.4f} at epoch {self.best_epoch + 1}"
                    )
                return True
        return False
