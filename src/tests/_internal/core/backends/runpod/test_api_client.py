import pytest

from dstack._internal.core.backends.runpod.api_client import (
    API_URL,
    RunpodApiClient,
    RunpodApiClientError,
)
from dstack._internal.core.errors import BackendInvalidCredentialsError


def test_create_gpu_pod_uses_v2_nested_request(requests_mock):
    requests_mock.post(f"{API_URL}/pods", json={"id": "pod-1"}, status_code=201)

    response = RunpodApiClient("test").create_pod(
        name="gpu-test",
        image_name="python:3.11-slim",
        gpu_type_id="NVIDIA A10G",
        cloud_type="SECURE",
        gpu_count=2,
        min_vcpu_count=8,
        min_memory_in_gb=32,
        container_disk_in_gb=50,
        data_center_id="US-KS-2",
        docker_args='{"cmd":["echo hi"]}',
        ports="22/tcp, 8080/http",
        volume_mount_path="/workspace",
        network_volume_id="vol-1",
        env={"HELLO": "WORLD"},
        allowed_cuda_versions=["12.8"],
        container_registry_auth_id="cred-1",
    )

    assert response == {"id": "pod-1"}
    assert requests_mock.last_request.json() == {
        "name": "gpu-test",
        "image": "python:3.11-slim",
        "cloud": "SECURE",
        "dataCenterIds": ["US-KS-2"],
        "startSsh": True,
        "disk": 50,
        "args": '{"cmd":["echo hi"]}',
        "ports": ["22/tcp", "8080/http"],
        "env": {"HELLO": "WORLD"},
        "registry": "cred-1",
        "mounts": {"network": [{"volumeId": "vol-1", "path": "/workspace"}]},
        "gpu": {
            "id": "NVIDIA A10G",
            "count": 2,
            "minVcpuCountPerGpu": 4,
            "minRamPerGpu": 16,
            "allowedCudaVersions": ["12.8"],
        },
    }


def test_create_gpu_pod_resolves_country_code_to_v2_data_centers(requests_mock):
    requests_mock.get(
        f"{API_URL}/catalog/datacenters",
        json={
            "dataCenters": [
                {
                    "id": "US-TX-3",
                    "gpuAvailability": [{"id": "NVIDIA A10G", "availability": "HIGH"}],
                },
                {
                    "id": "CA-MTL-1",
                    "gpuAvailability": [{"id": "NVIDIA A10G", "availability": "HIGH"}],
                },
            ]
        },
    )
    requests_mock.post(f"{API_URL}/pods", json={"id": "pod-1"}, status_code=201)

    RunpodApiClient("test").create_pod(
        name="community-test",
        image_name="python:3.11-slim",
        gpu_type_id="NVIDIA A10G",
        cloud_type="COMMUNITY",
        country_code="US",
    )

    assert requests_mock.request_history[0].qs == {"include": ["gpu_availability"]}
    assert requests_mock.last_request.json()["dataCenterIds"] == ["US-TX-3"]


def test_create_cpu_pod_uses_v2_cpu_request(requests_mock):
    requests_mock.post(f"{API_URL}/pods", json={"id": "cpu-pod-1"}, status_code=201)

    response = RunpodApiClient("test").create_cpu_pod(
        name="cpu-test",
        image_name="python:3.11-slim",
        instance_id="cpu3g-2-8",
        cloud_type="SECURE",
        deploy_cost=0.08,
        vcpu_count=8,
        data_center_id="AP-JP-1",
        container_disk_in_gb=5,
        docker_args='{"cmd":["echo hi"]}',
        ports="22/tcp, 8080/http",
        volume_mount_path="/workspace",
        env={"HELLO": "WORLD"},
        template_id="runpod-ubuntu",
        network_volume_id="vol-1",
        container_registry_auth_id="cred-1",
    )

    assert response["id"] == "cpu-pod-1"
    assert requests_mock.last_request.json()["cpu"] == {
        "id": "cpu3g-2-8",
        "vcpuCount": 8,
    }
    assert requests_mock.last_request.json()["mounts"] == {
        "network": [{"volumeId": "vol-1", "path": "/workspace"}]
    }


def test_get_and_terminate_pod_use_v2_paths(requests_mock):
    requests_mock.get(
        f"{API_URL}/pods/pod-1",
        json={"id": "pod-1", "runtime": {"ports": []}},
    )
    requests_mock.delete(f"{API_URL}/pods/pod-1", status_code=204)
    client = RunpodApiClient("test")

    assert client.get_pod("pod-1")["id"] == "pod-1"
    assert client.terminate_pod("pod-1") == {}
    assert [request.method for request in requests_mock.request_history] == ["GET", "DELETE"]


def test_validate_api_key_lists_pods(requests_mock):
    requests_mock.get(f"{API_URL}/pods", json={"pods": []})

    assert RunpodApiClient("test").validate_api_key() is True
    assert requests_mock.last_request.headers["Authorization"] == "Bearer test"


def test_unauthorized_v2_response_is_invalid_credentials(requests_mock):
    requests_mock.get(
        f"{API_URL}/pods",
        status_code=401,
        json={"title": "Unauthorized", "status": 401, "detail": "bad key"},
    )

    with pytest.raises(BackendInvalidCredentialsError):
        RunpodApiClient("bad").get_user_details()


def test_v2_problem_response_is_runpod_error(requests_mock):
    requests_mock.post(
        f"{API_URL}/pods",
        status_code=422,
        json={
            "title": "Unprocessable Entity",
            "status": 422,
            "detail": "invalid pod",
            "errors": ["$.gpu is required"],
        },
    )

    with pytest.raises(RunpodApiClientError) as exc_info:
        RunpodApiClient("test").create_cpu_pod(
            name="cpu-test",
            image_name="python:3.11-slim",
            instance_id="cpu3g-2-8",
            cloud_type="SECURE",
            deploy_cost=0.08,
        )

    assert exc_info.value.errors[0]["message"] == "invalid pod"
    assert exc_info.value.errors[0]["validation_errors"] == ["$.gpu is required"]


def test_registry_and_volume_methods_use_wrapped_v2_lists(requests_mock):
    requests_mock.get(
        f"{API_URL}/registries", json={"registries": [{"id": "reg-1", "name": "test"}]}
    )
    requests_mock.post(f"{API_URL}/registries", json={"id": "reg-2"}, status_code=201)
    requests_mock.delete(f"{API_URL}/registries/reg-2", status_code=204)
    requests_mock.get(
        f"{API_URL}/network-volumes/vol-1",
        json={"id": "vol-1", "size": 50},
    )
    requests_mock.post(f"{API_URL}/network-volumes", json={"id": "vol-2"}, status_code=201)
    requests_mock.delete(f"{API_URL}/network-volumes/vol-2", status_code=204)
    client = RunpodApiClient("test")

    assert client.get_container_registry_auths() == [{"id": "reg-1", "name": "test"}]
    assert client.add_container_registry_auth("test", "user", "password") == "reg-2"
    client.delete_container_registry_auth("reg-2")
    volume = client.get_network_volume("vol-1")
    assert volume is not None
    assert volume["size"] == 50
    assert client.create_network_volume("data", "US-KS-2", 50) == "vol-2"
    client.delete_network_volume("vol-2")


def test_cluster_creation_reads_v2_member_pods(requests_mock):
    requests_mock.post(
        f"{API_URL}/clusters",
        json={"id": "cluster-1", "name": "cluster"},
        status_code=201,
    )
    requests_mock.get(
        f"{API_URL}/clusters/cluster-1/pods",
        json={"pods": [{"id": "pod-1", "cluster": {"ip": "10.0.0.2"}}]},
    )

    response = RunpodApiClient("test").create_cluster(
        cluster_name="cluster",
        gpu_type_id="NVIDIA A10G",
        pod_count=2,
        gpu_count_per_pod=1,
        image_name="python:3.11-slim",
        deploy_cost="1.00",
        data_center_id="US-KS-2",
        container_disk_in_gb=20,
        volume_mount_path="/workspace",
        network_volume_id="vol-1",
    )

    assert response["pods"] == [
        {"id": "pod-1", "cluster": {"ip": "10.0.0.2"}, "clusterIp": "10.0.0.2"}
    ]
    assert requests_mock.request_history[0].json()["compute"] == {
        "gpuTypeId": "NVIDIA A10G",
        "gpuCountPerPod": 1,
        "podCount": 2,
    }
