import yaml
from kubernetes import client, config
from os import path

JOB_NAME = "fed-moe"
NAMESPACE = "second-carrier-prediction"
IMAGE = "registry.ailab.rnd.ki.sw.ericsson.se/second-carrier-prediction/main/fl-moe"

def create_job_object():

    resources = client.V1ResourceRequirements(
        limits={"cpu": "20", "memory": "64Gi"},
        requests={"cpu": "20", "memory": "64Gi"})

    volumeMounts = client.V1VolumeMount(
        name="projectdisk",
        mount_path="/proj/second-carrier-prediction/")

    container = client.V1Container(
        name="fed-moe",
        image=IMAGE,
        command=["perl", "-Mbignum=bpi", "-wle", "print bpi(2000)"],
        volume_mounts=[volumeMounts],
        resources=resources)

    secrets = client.V1LocalObjectReference()
    node_selector = {"nvidia.com/gpu": "true"}

    tolerations = client.V1Toleration(
        key="nvidia.com/gpu",
        operator="Exists",
        effect="NoSchedule")

    pvc = client.V1PersistentVolumeClaimVolumeSource(
        claim_name="cephfs-second-carrier-prediction")

    volumes = client.V1Volume(name="projectdisk", persistent_volume_claim=pvc)

    template = client.V1PodTemplateSpec(
        metadata=client.V1ObjectMeta(labels={"app": "fed-moe"}),
        spec=client.V1PodSpec(
            restart_policy="Never",
            containers=[container],
            tolerations=[tolerations],
            node_selector=node_selector,
            image_pull_secrets=[secrets],
            volumes=[volumes]
        ))

    spec = client.V1JobSpec(
        template=template,
        backoff_limit=4)

    job = client.V1Job(
        api_version="batch/v1",
        kind="Job",
        metadata=client.V1ObjectMeta(name=JOB_NAME),
        spec=spec)

    return job


def create_job(api_instance, job):
    api_response = api_instance.create_namespaced_job(
        body=job,
        namespace=NAMESPACE)
    print("Job created. status='%s'" % str(api_response.status))

def main():

    config.load_kube_config()
    batch_v1 = client.BatchV1Api()
    job = create_job_object()

    create_job(batch_v1, job)


if __name__ == '__main__':
    main()
