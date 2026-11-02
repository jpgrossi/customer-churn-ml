import mlflow
from mlflow.tracking import MlflowClient

# Run elegido: logreg_balanced (ver comparacion en el README)
RUN_ID = "22a9815163114e4592c049c3b77c0130"
MODEL_NAME = "CustomerChurnModel"


def main():
    client = MlflowClient()
    run = client.get_run(RUN_ID)

    resultado = mlflow.register_model(model_uri="runs:/" + RUN_ID + "/model", name=MODEL_NAME)
    version = resultado.version

    client.set_model_version_tag(MODEL_NAME, version, "run_name", run.data.tags["mlflow.runName"])
    client.set_model_version_tag(MODEL_NAME, version, "git_sha", run.data.tags["git_sha"])
    client.set_model_version_tag(MODEL_NAME, version, "dataset_md5", run.data.tags["dataset_md5"])
    client.set_model_version_tag(MODEL_NAME, version, "primary_metric", "recall=" + str(round(run.data.metrics["recall"], 3)))
    client.set_model_version_tag(MODEL_NAME, version, "positive_class", "Yes")
    client.set_model_version_tag(MODEL_NAME, version, "owner", "Grupo Customer Churn ISTEA")

    client.update_model_version(
        name=MODEL_NAME,
        version=version,
        description="LogisticRegression con class_weight=balanced. Elegido por mayor recall (0.723) y ROC-AUC (0.812) frente a rf_balanced, con menor complejidad. Precision 0.481."
    )

    client.set_registered_model_alias(MODEL_NAME, "candidate", version)

    print("Modelo registrado:", MODEL_NAME, "- version", version, "- alias candidate")


if __name__ == "__main__":
    main()

