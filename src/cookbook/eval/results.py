import os
from dataclasses import dataclass
from typing import Any

import requests

from cookbook.cli.utils import (
    PythonEnv,
    add_aws_flags,
    add_secret_to_beaker_workspace,
    install_oe_eval,
    make_eval_run_name,
    escape_datalake_tags
)


@dataclass(frozen=True)
class DatalakeApi:
    base_url: str = "https://oe-eval-datalake.allen.ai"

    def health(self) -> bool:
        """Check API health"""
        url = f"{self.base_url}/health"
        response = requests.get(url)
        return response.status_code == 204

    # Greenlake endpoints
    def upload_to_greenlake(self, experiment_id: str, **kwargs) -> dict[str, Any]:
        """Uploads an eval experiment's results from Beaker to Greenlake."""
        url = f"{self.base_url}/greenlake/upload/{experiment_id}"
        response = requests.post(url, params=kwargs)
        response.raise_for_status()
        return response.json()

    def inspect_greenlake(self, experiment_id: str, resulttype: str | None = None) -> dict[str, Any]:
        """Checks if an eval experiment's results have been collected in Greenlake."""
        url = f"{self.base_url}/greenlake/inspect/{experiment_id}"
        params = {"resulttype": resulttype} if resulttype else {}
        response = requests.get(url, params=params)
        response.raise_for_status()
        return response.json()

    def get_exp_metrics(
        self, experiment_id: str, workspace: str | None = None, beaker_info: bool = False
    ) -> list[dict[str, Any]]:
        """Returns all tasks metrics of an eval experiment as list of dictionaries."""
        url = f"{self.base_url}/greenlake/metrics-all/{experiment_id}"
        params = {}
        if workspace:
            params["workspace"] = workspace
        if beaker_info:
            params["beaker_info"] = beaker_info
        response = requests.get(url, params=params)
        response.raise_for_status()
        return response.json()

    def get_exp_metadata(self, experiment_id: str, workspace: str | None = None) -> list[dict[str, Any]]:
        """Returns the metadata of an eval experiment."""
        url = f"{self.base_url}/greenlake/metadata/{experiment_id}"
        params = {"workspace": workspace} if workspace else {}
        response = requests.get(url, params=params)
        response.raise_for_status()
        return response.json()

    def get_exp_result(
        self, experiment_id: str, workspace: str | None = None, resulttype: str = "ALL_METRICS", task_idx: int = 0
    ) -> list[dict[str, Any]]:
        """Returns the result of an eval experiment."""
        url = f"{self.base_url}/greenlake/get-result/{experiment_id}"
        params = {"resulttype": resulttype, "task_idx": task_idx}
        if workspace:
            params["workspace"] = workspace
        response = requests.get(url, params=params)
        response.raise_for_status()
        return response.json()

    def download_exp_result(
        self,
        experiment_id: str,
        workspace: str | None = None,
        resulttype: str = "ALL_METRICS",
        task_idx: int = 0,
        newfilename: str | None = None,
    ) -> bytes:
        """Downloads a result file for an eval experiment."""
        url = f"{self.base_url}/greenlake/download-result/{experiment_id}"
        params = {"resulttype": resulttype, "task_idx": task_idx}
        if workspace:
            params["workspace"] = workspace
        if newfilename:
            params["newfilename"] = newfilename
        response = requests.get(url, params=params)
        response.raise_for_status()
        return response.content

    def find_experiment(
        self,
        checkpoint_path: str,
        add_bos_token: bool = False,
        from_created_dt: str = "2024-07-01",
        limit: int | None = 1000,
        **kwargs,
    ) -> list[str]:
        """Finds experiments."""
        url = f"{self.base_url}/bluelake/find-experiments/"

        def parse_and_validate(response: requests.Response) -> list[str]:
            results = response.json()
            assert (
                isinstance(results, list) and all("experiment_id" in result for result in results)
            ), f"Expected a list of results with experiment_id key, got {results}--have API changed?"
            return [result["experiment_id"] for result in results]

        # we first try to see if model_path yields to any results by specifying it as tag
        model_path_tag = escape_datalake_tags({"model_path": checkpoint_path})["model_path"]
        params = {"tags": model_path_tag, "from_created_dt": from_created_dt, "limit": limit}
        response = requests.get(url, params=params)
        response.raise_for_status()

        if len(results := parse_and_validate(response)) > 0:
            return results

        # if no results are found, we try to find the experiment by model_path in the experiment name
        run_name = make_eval_run_name(checkpoint_path=checkpoint_path, add_bos_token=add_bos_token)

        params = {"model_name": run_name, "from_created_dt": from_created_dt, "limit": limit}
        response = requests.get(url, params=params)
        response.raise_for_status()
        return parse_and_validate(response)


    def find_experiment_results(
        self, resulttype: str = "ALL_METRICS", from_created_dt: str = "2024-07-01", **kwargs
    ) -> list[dict[str, Any]]:
        """Finds experiments and fetches their results."""
        url = f"{self.base_url}/bluelake/find-experiments-result/"
        params = {"resulttype": resulttype, "from_created_dt": from_created_dt, **kwargs}
        response = requests.get(url, params=params)
        response.raise_for_status()
        return response.json()

    def add_experiment_tags(self, experiment_id: str, tags: str, overwrite: bool = False) -> dict[str, Any]:
        """Add tags to an experiment."""
        url = f"{self.base_url}/bluelake/add-experiment-tags/"
        params = {"experiment_id": experiment_id, "tags": tags, "overwrite": overwrite}
        response = requests.put(url, params=params)
        response.raise_for_status()
        return response.json()

    def get_task_config(self, task_hash: str, return_full_dict: bool = False) -> dict[str, Any]:
        """Fetches the full task configs for a task_hash."""
        url = f"{self.base_url}/bluelake/get-task-config/{task_hash}"
        params = {"return_full_dict": return_full_dict}
        response = requests.get(url, params=params)
        response.raise_for_status()
        return response.json()

    def get_model_config(self, model_hash: str, return_full_dict: bool = False) -> dict[str, Any]:
        """Fetches the full model configs for a model_hash."""
        url = f"{self.base_url}/bluelake/get-model-config/{model_hash}"
        params = {"return_full_dict": return_full_dict}
        response = requests.get(url, params=params)
        response.raise_for_status()
        return response.json()

    def get_model_task_eval(
        self, model_hash: str, task_hash: str, return_fields: str | None = None, return_full_dict: bool = False
    ) -> list[dict[str, Any]]:
        """Fetches the full existing eval metrics for a model_hash/task_hash pair."""
        url = f"{self.base_url}/bluelake/get-model-task-eval/"
        params = {
            "model_hash": model_hash,
            "task_hash": task_hash,
        }
        if return_fields:
            params["return_fields"] = return_fields
        if return_full_dict:
            params["return_full_dict"] = return_full_dict
        response = requests.get(url, params=params)
        response.raise_for_status()
        return response.json()



if __name__ == "__main__":

    client = DatalakeApi()
    exp = client.find_experiment(
        checkpoint_path="weka://oe-training-default/ai2-llm/checkpoints/OLMo-medium/peteish7/step59000-hf"
    )
    print(exp)
