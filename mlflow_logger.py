# --- MLflow drop-in replacement for TensorBoard SummaryWriter ---
import os, io, contextlib, mlflow
from datetime import datetime
from PIL import Image
import numpy as np

class MLflowWriter:
    """
    Minimal shim so existing `writer.add_scalar(...)` and (one-off) `add_images(...)`
    keep working. Uses MLflow metrics & artifacts under the hood.
    """
    def __init__(self, 
                 experiment="multi_error_env", 
                 experiment_id=None,
                 run_name=None, 
                 params: dict=None,
                 tracking_uri="databricks"):
        mlflow.set_tracking_uri(tracking_uri)

        if experiment_id:
            mlflow.set_experiment(experiment_id=experiment_id)
        elif experiment:
            mlflow.set_experiment(experiment)
        else:
            raise ValueError("Must provide experiment path or experiment_id")
            
        # Autoname the run if not provided
        self._run = mlflow.start_run(run_name=run_name or datetime.now().strftime("%m%d_%H%M%S"))
        if params:
            # Make sure all params are JSON/str-friendly
            safe_params = {}
            for k,v in params.items():
                try:
                    _ = str(v)
                    safe_params[k] = v
                except Exception:
                    safe_params[k] = repr(v)
            mlflow.log_params(safe_params)

    # mirrors SummaryWriter.add_scalar(tag, scalar_value, global_step)
    def add_scalar(self, tag: str, value, step: int):
        try:
            val = float(value)
        except Exception:
            # fallback: try tensor -> python number
            val = float(getattr(value, "item", lambda: value)())
        mlflow.log_metric(tag, val, step=step)

    # very lightweight replacement for your single usage of add_images(...)
    # img_tensor expected as (N, C, H, W) with C=1 or 3
    def add_images(self, tag: str, img_tensor, global_step: int, dataformats='NCHW', max_images: int=16):
        if dataformats != 'NCHW':
            raise ValueError("This shim expects dataformats='NCHW'.")

        imgs = img_tensor
        if hasattr(imgs, "detach"):  # torch tensor -> numpy
            imgs = imgs.detach().cpu().numpy()

        N, C, H, W = imgs.shape
        N = min(N, max_images)
        base_path = f"{tag}/step_{global_step}"

        for i in range(N):
            arr = imgs[i]
            # Expect either (1, H, W) or (3, H, W)
            if C == 1:
                arr = arr[0]  # (H, W) float
                # mlflow accepts float [0,1] or uint8 [0,255]. Ensure [0,1].
                arr = np.clip(arr, 0.0, 1.0).astype(np.float32)  # (H, W)
                # You can pass numpy directly:
                mlflow.log_image(arr, artifact_file=f"{base_path}/img_{i:03d}.png")
            elif C == 3:
                arr = np.transpose(arr, (1, 2, 0))  # (H, W, 3)
                arr = np.clip(arr, 0.0, 1.0).astype(np.float32)
                mlflow.log_image(arr, artifact_file=f"{base_path}/img_{i:03d}.png")
            else:
                # Skip unexpected channel counts
                continue


    # convenience to log files or whole folders you already create (your scatter3d htmls)
    def log_artifact(self, path: str, artifact_path: str=None):
        if os.path.isdir(path):
            mlflow.log_artifacts(path, artifact_path=artifact_path)
        else:
            mlflow.log_artifact(path, artifact_path=artifact_path)

    def close(self):
        # End the run cleanly
        try:
            mlflow.end_run()
        except Exception:
            pass
# ----------------------------------------------------------------

if __name__ == "__main__":

    mlflow.set_tracking_uri("databricks")        # or "databricks://mlflow" if using profile
    mlflow.set_experiment(experiment_id="1490651313414470")

    with mlflow.start_run(run_name="sanity"):
        mlflow.log_metric("ping", 1.0, step=0)

    print("OK")
