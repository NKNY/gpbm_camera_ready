"""MLflow logging utilities."""

import os
import sys
import io
from dataclasses import asdict

try:
    import boto3
except ImportError:
    boto3 = None

try:
    import mlflow
except ImportError:
    mlflow = None


class Logger:
    """Wrapper for MLflow logging that gracefully handles missing mlflow.

    If mlflow is None, all logging methods are no-ops.
    """

    def __init__(self, mlflow=None, run=None, tracking_uri=None):
        self._mlflow = mlflow
        self._run = run
        self._tracking_uri = tracking_uri

    def log_param(self, key, value):
        """Log a single parameter."""
        if self._mlflow:
            self._mlflow.log_param(key, value)

    def log_metrics(self, metrics, step, prefix=""):
        """Log metrics dict, sanitizing @ in keys to _at_."""
        if self._mlflow:
            sanitized = {
                f"{prefix}{k.replace('@', '_at_')}": v for k, v in metrics.items()
            }
            self._mlflow.log_metrics(sanitized, step=step)

    def close(self):
        """End the MLflow run."""
        if not self._mlflow:
            return
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        try:
            self._mlflow.end_run()
        finally:
            sys.stdout = old_stdout


def _get_mlflow_url(tracking_uri, experiment_id, run_id):
    """Build MLflow UI URL from tracking URI and run info."""
    if not tracking_uri:
        return None
    if tracking_uri.startswith("arn:"):
        try:
            parts = tracking_uri.split(":")
            region = parts[3]
            server_name = parts[5].split("/")[1]
            client = boto3.client("sagemaker", region_name=region)
            response = client.describe_mlflow_tracking_server(
                TrackingServerName=server_name
            )
            domain = response["TrackingServerUrl"].rstrip("/").replace("https://", "")
        except Exception:
            return None
    else:
        domain = tracking_uri.rstrip("/").replace("https://", "").replace("http://", "")
    return f"https://{domain}/#/experiments/{experiment_id}/runs/{run_id}"


def create_logger(config, tags=None):
    """Create Logger instance, initializing MLflow if enabled in config.

    :param config: Training config with mlflow_* fields.
    :param tags: Optional additional tags dict.
    :return: Logger instance (no-op if mlflow disabled or not installed).
    """
    if not config.mlflow_tracking:
        return Logger()
    if mlflow is None:
        print("Warning: mlflow not installed, skipping tracking")
        return Logger()

    os.environ["MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR"] = "false"
    if config.mlflow_system_metrics:
        os.environ["MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING"] = "true"
        mlflow.enable_system_metrics_logging()
    if config.mlflow_tracking_uri:
        mlflow.set_tracking_uri(config.mlflow_tracking_uri)
    if config.mlflow_experiment:
        mlflow.set_experiment(config.mlflow_experiment)

    run = mlflow.start_run(
        run_name=config.mlflow_run_name, log_system_metrics=config.mlflow_system_metrics
    )

    # Log config params
    params = {}
    for k, v in asdict(config).items():
        if isinstance(v, (int, float, str, bool, type(None))):
            params[k] = v
        elif k == "loss_kwargs":
            for kk, vv in v.items():
                params[f"loss_{kk}"] = vv
    mlflow.log_params(params)

    # Tags
    all_tags = {**config.mlflow_tags, **(tags or {})}
    if config.mlflow_note and "mlflow.note.content" not in all_tags:
        all_tags["mlflow.note.content"] = config.mlflow_note
    if all_tags:
        mlflow.set_tags(all_tags)

    url = _get_mlflow_url(
        config.mlflow_tracking_uri, run.info.experiment_id, run.info.run_id
    )
    if url:
        print(f"🏃 View run at: {url}")

    return Logger(mlflow, run, config.mlflow_tracking_uri)
