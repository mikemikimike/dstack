import math
import time
from datetime import timedelta
from typing import Any, Dict, List, Optional

import requests
from gpuhunt.providers.runpod import RunpodProvider
from requests import Response

from dstack._internal.core.errors import BackendError, BackendInvalidCredentialsError
from dstack._internal.utils.common import get_current_datetime

API_URL = "https://api.runpod.io/v2"


class RunpodApiClientError(BackendError):
    errors: List[Dict]

    def __init__(self, errors: List[Dict]):
        self.errors = errors
        super().__init__(errors)


class RunpodApiClient:
    def __init__(self, api_key: str):
        self._session = requests.Session()
        self._session.headers.update({"Authorization": f"Bearer {api_key}"})

    def validate_api_key(self) -> bool:
        try:
            self.get_user_details()
        except BackendInvalidCredentialsError:
            return False
        return True

    def get_user_details(self) -> Dict:
        # The v2 API has no account-details endpoint. Listing pods is an
        # authenticated, account-scoped request and is sufficient to validate
        # the key without creating or changing a resource.
        return self._make_request("GET", "/pods").json()

    def create_pod(
        self,
        name: str,
        image_name: str,
        gpu_type_id: str,
        cloud_type: str,
        support_public_ip: bool = True,
        start_ssh: bool = True,
        data_center_id: Optional[str] = None,
        country_code: Optional[str] = None,
        gpu_count: int = 1,
        volume_in_gb: int = 0,
        container_disk_in_gb: Optional[int] = None,
        min_vcpu_count: int = 1,
        min_memory_in_gb: int = 1,
        docker_args: str = "",
        ports: Optional[str] = None,
        volume_mount_path: Optional[str] = None,
        env: Optional[Dict[str, Any]] = None,
        template_id: Optional[str] = None,
        network_volume_id: Optional[str] = None,
        allowed_cuda_versions: Optional[List[str]] = None,
        bid_per_gpu: Optional[float] = None,
        container_registry_auth_id: Optional[str] = None,
    ) -> Dict:
        del support_public_ip, bid_per_gpu
        gpu = {
            "id": gpu_type_id,
            "count": gpu_count,
            "minVcpuCountPerGpu": max(1, math.ceil(min_vcpu_count / gpu_count)),
            "minRamPerGpu": max(1, math.ceil(min_memory_in_gb / gpu_count)),
        }
        if allowed_cuda_versions is not None:
            gpu["allowedCudaVersions"] = allowed_cuda_versions
        else:
            gpu["minCudaVersion"] = RunpodProvider.MIN_CUDA_VERSION
        payload = self._build_container_payload(
            name=name,
            image_name=image_name,
            cloud_type=cloud_type,
            data_center_id=data_center_id,
            country_code=None,
            start_ssh=start_ssh,
            container_disk_in_gb=container_disk_in_gb,
            docker_args=docker_args,
            ports=ports,
            volume_mount_path=volume_mount_path,
            volume_in_gb=volume_in_gb,
            env=env,
            template_id=template_id,
            network_volume_id=network_volume_id,
            container_registry_auth_id=container_registry_auth_id,
        )
        payload["gpu"] = gpu
        if country_code is not None:
            payload["dataCenterIds"] = self._resolve_data_center_ids(
                country_code=country_code,
                resource_id=gpu_type_id,
                availability_key="gpuAvailability",
            )
        return self._make_request("POST", "/pods", payload).json()

    def create_cpu_pod(
        self,
        name: str,
        image_name: str,
        instance_id: str,
        cloud_type: str,
        deploy_cost: float,
        vcpu_count: int = 2,
        start_ssh: bool = True,
        data_center_id: Optional[str] = None,
        container_disk_in_gb: Optional[int] = None,
        docker_args: Optional[str] = None,
        ports: Optional[str] = None,
        volume_mount_path: Optional[str] = None,
        env: Optional[Dict[str, Any]] = None,
        template_id: Optional[str] = None,
        network_volume_id: Optional[str] = None,
        container_registry_auth_id: Optional[str] = None,
    ) -> Dict:
        del deploy_cost
        payload = self._build_container_payload(
            name=name,
            image_name=image_name,
            cloud_type=cloud_type,
            data_center_id=data_center_id,
            country_code=None,
            start_ssh=start_ssh,
            container_disk_in_gb=container_disk_in_gb,
            docker_args=docker_args,
            ports=ports,
            volume_mount_path=volume_mount_path,
            env=env,
            template_id=template_id,
            network_volume_id=network_volume_id,
            container_registry_auth_id=container_registry_auth_id,
        )
        payload["cpu"] = {"id": instance_id, "vcpuCount": vcpu_count}
        return self._make_request("POST", "/pods", payload).json()

    def update_pod_container_registry_auth(
        self,
        pod_id: str,
        container_registry_auth_id: str,
    ) -> str:
        response = self._make_request(
            "PATCH", f"/pods/{pod_id}", {"registry": container_registry_auth_id}
        )
        return response.json()["id"]

    def get_pod(self, pod_id: str) -> Dict:
        return self._make_request("GET", f"/pods/{pod_id}").json()

    def terminate_pod(self, pod_id: str) -> Dict:
        response = self._make_request("DELETE", f"/pods/{pod_id}")
        return response.json() if response.content else {}

    def get_container_registry_auths(self) -> List[Dict]:
        return self._make_request("GET", "/registries").json()["registries"]

    def add_container_registry_auth(self, name: str, username: str, password: str) -> str:
        return self._make_request(
            "POST",
            "/registries",
            {"name": name, "username": username, "password": password},
        ).json()["id"]

    def delete_container_registry_auth(self, auth_id: str) -> None:
        self._make_request("DELETE", f"/registries/{auth_id}")

    def get_network_volume(self, volume_id: str) -> Optional[Dict]:
        try:
            return self._make_request("GET", f"/network-volumes/{volume_id}").json()
        except RunpodApiClientError as exc:
            if any(error.get("status") == requests.codes.not_found for error in exc.errors):
                return None
            raise

    def create_network_volume(self, name: str, region: str, size: int) -> str:
        return self._make_request(
            "POST",
            "/network-volumes",
            {"name": name, "size": size, "dataCenter": region},
        ).json()["id"]

    def delete_network_volume(self, volume_id: str) -> None:
        self._make_request("DELETE", f"/network-volumes/{volume_id}")

    def create_cluster(
        self,
        cluster_name: str,
        gpu_type_id: str,
        pod_count: int,
        gpu_count_per_pod: int,
        image_name: str,
        deploy_cost: str,
        template_id: Optional[str] = None,
        cluster_type: str = "TRAINING",
        network_volume_id: Optional[str] = None,
        volume_in_gb: Optional[int] = None,
        throughput: Optional[int] = None,
        allowed_cuda_versions: Optional[List[str]] = None,
        volume_key: Optional[str] = None,
        data_center_id: Optional[str] = None,
        start_jupyter: bool = False,
        start_ssh: bool = False,
        container_disk_in_gb: Optional[int] = None,
        docker_args: Optional[str] = None,
        env: Optional[Dict[str, Any]] = None,
        volume_mount_path: Optional[str] = None,
        ports: Optional[str] = None,
    ) -> Dict:
        del deploy_cost, throughput, volume_key, allowed_cuda_versions
        payload = self._build_container_payload(
            name=cluster_name,
            image_name=image_name,
            cloud_type=None,
            data_center_id=data_center_id,
            country_code=None,
            start_ssh=start_ssh,
            container_disk_in_gb=container_disk_in_gb,
            docker_args=docker_args,
            ports=ports,
            volume_mount_path=volume_mount_path,
            volume_in_gb=volume_in_gb,
            env=env,
            template_id=template_id,
            network_volume_id=network_volume_id,
            container_registry_auth_id=None,
        )
        payload.update(
            {
                "type": cluster_type,
                "compute": {
                    "gpuTypeId": gpu_type_id,
                    "gpuCountPerPod": gpu_count_per_pod,
                    "podCount": pod_count,
                },
                "startJupyter": start_jupyter,
            }
        )
        response = self._make_request("POST", "/clusters", payload).json()
        pods = self._make_request("GET", f"/clusters/{response['id']}/pods").json().get("pods", [])
        for pod in pods:
            pod.setdefault("clusterIp", pod.get("cluster", {}).get("ip"))
        response["pods"] = pods
        return response

    def delete_cluster(self, cluster_id: str) -> bool:
        self._make_request("DELETE", f"/clusters/{cluster_id}")
        return True

    def _build_container_payload(
        self,
        name: str,
        image_name: str,
        cloud_type: Optional[str],
        data_center_id: Optional[str],
        country_code: Optional[str],
        start_ssh: bool,
        container_disk_in_gb: Optional[int],
        docker_args: Optional[str],
        ports: Optional[str],
        volume_mount_path: Optional[str],
        env: Optional[Dict[str, Any]],
        template_id: Optional[str],
        network_volume_id: Optional[str],
        container_registry_auth_id: Optional[str] = None,
        volume_in_gb: Optional[int] = None,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"name": name, "image": image_name}
        if cloud_type is not None:
            payload["cloud"] = cloud_type
        if data_center_id is not None:
            payload["dataCenterIds"] = [data_center_id]
        elif country_code is not None:
            # Country-code offers are resolved to v2 data-center IDs by the
            # resource-specific create method before the request is sent.
            del country_code
        if start_ssh:
            payload["startSsh"] = True
        if container_disk_in_gb is not None:
            payload["disk"] = container_disk_in_gb
        if docker_args is not None:
            payload["args"] = docker_args
        if ports is not None:
            payload["ports"] = [port.strip() for port in ports.split(",")]
        if env is not None:
            payload["env"] = {key: str(value) for key, value in env.items()}
        if template_id is not None:
            payload["templateId"] = template_id
        if container_registry_auth_id is not None:
            payload["registry"] = container_registry_auth_id
        if network_volume_id is not None:
            if volume_mount_path is None:
                raise ValueError("volume_mount_path is required for a network volume")
            payload["mounts"] = {
                "network": [{"volumeId": network_volume_id, "path": volume_mount_path}]
            }
        elif volume_in_gb:
            if volume_mount_path is None:
                raise ValueError("volume_mount_path is required for persistent storage")
            payload["mounts"] = {"persistent": {"size": volume_in_gb, "path": volume_mount_path}}
        return payload

    def _resolve_data_center_ids(
        self,
        country_code: str,
        resource_id: str,
        availability_key: str,
    ) -> List[str]:
        response = self._make_request(
            "GET",
            "/catalog/datacenters",
            params={"include": availability_key.replace("Availability", "_AVAILABILITY").upper()},
        ).json()
        data_center_ids = []
        country_code = country_code.upper()
        for data_center in response["dataCenters"]:
            if not data_center["id"].upper().startswith(f"{country_code}-"):
                continue
            if any(
                resource["id"] == resource_id and resource["availability"] != "NONE"
                for resource in data_center.get(availability_key, [])
            ):
                data_center_ids.append(data_center["id"])
        if not data_center_ids:
            raise RunpodApiClientError(
                [
                    {
                        "message": (
                            f"No {resource_id} capacity was found in Runpod country "
                            f"code {country_code}"
                        ),
                        "status": requests.codes.bad_request,
                    }
                ]
            )
        return data_center_ids

    def _make_request(
        self,
        method: str,
        path: str,
        data: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> Response:
        response = self._session.request(
            method=method,
            url=f"{API_URL}{path}",
            json=data,
            params=params,
            timeout=120,
        )
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            if response.status_code in (requests.codes.forbidden, requests.codes.unauthorized):
                raise BackendInvalidCredentialsError(response.text) from exc
            try:
                problem = response.json()
            except ValueError:
                problem = {}
            detail = problem.get("detail") or problem.get("message") or response.text
            error = {
                "message": detail,
                "title": problem.get("title"),
                "status": response.status_code,
                "extensions": {"code": problem.get("title")},
            }
            if problem.get("errors"):
                error["validation_errors"] = problem["errors"]
            raise RunpodApiClientError([error]) from exc
        return response

    def wait_for_instance(self, instance_id: str) -> Optional[Dict]:
        start = get_current_datetime()
        wait_for_instance_interval = 5
        # To change the status to "running," the image must be pulled and then started.
        # We have to wait for 20 minutes while a large image is pulled.
        while get_current_datetime() < (start + timedelta(minutes=20)):
            pod = self.get_pod(instance_id)
            if pod["runtime"] is not None:
                return pod
            time.sleep(wait_for_instance_interval)
        return None
