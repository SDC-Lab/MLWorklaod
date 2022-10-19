from kubernetes import client, config # pip install kubernetes
from typing import Dict, List, Union
from shlex import quote  # Prevent command injection
from datetime import datetime
from icmplib import multiping # pip install icmplib
import subprocess
import csv
import requests
import time
import re

INTERVAL = 15  # The minimum for this value is 15 seconds - this is the fastest the Kubernetes metrics server will update usage metrics
START_TIME = time.time() # Current time
GATEWAY_IP = "10.0.0.169"
GATEWAY_PORT = "16443"
PROMETHEUS_IP = "<INSERT IP>"
PROMETHEUS_PORT = "9090"
OPENFAAS_NAMESPACE = "openfaas-fn"
OPENFAAS_USERNAME = "admin"
OPENFAAS_PASSWORD = "<INSERT PASSWORD>"
KUBECONFIG_PATH = "/home/k8s-master/.kube/config"


def get_kube_api_object(config_object: config, filepath: str = None):
    config_object.load_kube_config(config_file=filepath)
    return client.CoreV1Api()

kube_api = get_kube_api_object(config, KUBECONFIG_PATH)
api = client.CustomObjectsApi()

nodes_resource_usage = api.list_cluster_custom_object("metrics.k8s.io", "v1beta1", "nodes")




def get_sleep_duration():
    return INTERVAL - ((time.time() - START_TIME) % INTERVAL)


def get_all_nodes(kube_api: client.CoreV1Api):
    return list(kube_api.list_node().items)


def get_all_pods(kube_api: client.CoreV1Api, namespace: str = None):
    if (namespace is None):
        return list(kube_api.list_pod_for_all_namespaces().items)

    return list(kube_api.list_namespaced_pod(namespace).items)


def get_nodes_matching_label_values(kube_api: client.CoreV1Api, label_name: str, label_values: Union[str, List[str]]):
    if isinstance(label_values, str):
        matching_nodes = kube_api.list_node(
            label_selector="%s=%s" % (label_name, label_values))
    else:
        matching_nodes = kube_api.list_node(label_selector="%s in (%s)" % (
            label_name, str(label_values).replace("[", "").replace("]", "").replace("'", "")))

    return list(matching_nodes.items)


def get_nodes_matching_labels_list(kube_api: client.CoreV1Api, labels: Dict[str, List[str]]):
    result = {}
    label_names = list(labels.keys())

    for label_name in label_names:
        matching_nodes = get_nodes_matching_label_values(
            kube_api, label_name, labels[label_name])

        for node in matching_nodes:
            # Ensure returned list does not contain duplicate results.
            result[node.metadata.name] = node

    return list(result.values())


def set_node_labels(kube_api: client.CoreV1Api, node_name: str, labels: Dict[str, Union[str, None]]):
    body = {
        "metadata": {
            "labels": labels
        }
    }

    api_response = kube_api.patch_node(node_name, body)
    return api_response


def get_resource_usage_percentage(used: str, allocatable: str):
    decimal_units = ["n", "u", "m", None, "k", "M", "G", "T", "P", "E"]
    binary_units = [None, "Ki", "Mi", "Gi", "Ti", "Pi", "Ei"]

    targets = [used, allocatable]

    results = []

    for target in targets:
        target_unit = None
        target_value = None

        regex_match = re.search(r"([A-Z])\w*", target, flags=re.IGNORECASE)

        if regex_match is not None:
            startIndex = regex_match.start()
            target_value = target[0:startIndex]
            target_unit = target[startIndex:]
        else:
            if len(target) > 0:
                target_value = target

        multiplier = None
        if target_unit in decimal_units:
            multiplier = 1000 ** (decimal_units.index(target_unit) - 3)
        elif target_unit in binary_units:
            multiplier = 1024 ** binary_units.index(target_unit)

        if target_value is not None and multiplier is not None:
            results.append(float(target_value) * multiplier)

    if len(results) != 2:
        return None

    return results[0] / results[1] * 100


def get_pods():
    output = subprocess.getoutput("kubectl get pods -n openfaas-fn")
    output_lines = output.split('\n')[1:]
    count = 0
    for output_line in output_lines:
        if "Running" not in output_line:
            continue
        #if "har-cnn-infer" not in output_line:
        #    continue
        count+=1
    return count

def get_node_resource_usage(kube_api: client.CoreV1Api, kube_custom_object_api: client.CustomObjectsApi):
    nodes = get_all_nodes(kube_api)
    node_allocatable = {}

    for node in nodes:
        node_allocatable[node.metadata.name] = {
            "cpu": node.status.allocatable["cpu"],
            "memory": node.status.allocatable["memory"]
        }

    # nodes_resource_usage = kube_custom_object_api.list_cluster_custom_object("metrics.k8s.io", "v1beta1", "nodes")
    
    result = {}
    output = subprocess.getoutput("kubectl top nodes")
    output_lines = output.split('\n')[1:]
    for output_line in output_lines:
        stats = output_line.split()
        if len(stats) != 5:
            print("Error stats - %s" % output_line)
            return result
        print(stats)
        node_name = stats[0]
        result[node_name] = {
            "node": node_name,
            "allocatable_cpu": node_allocatable[node_name]["cpu"],
            "used_cpu": stats[1],
            "pct_used_cpu": get_resource_usage_percentage(stats[1], node_allocatable[node_name]["cpu"]),
            "allocatable_memory": node_allocatable[node_name]["memory"],
            "used_memory": stats[3],
            "pct_used_memory": get_resource_usage_percentage(stats[3], node_allocatable[node_name]["memory"]),
        }

    # result = {}
    # for stats in nodes_resource_usage["items"]:
    #     node_name = stats["metadata"]["name"]
    #     result[node_name] = {
    #         "node": node_name,
    #         "allocatable_cpu": node_allocatable[node_name]["cpu"],
    #         "used_cpu": stats["usage"]["cpu"],
    #         "pct_used_cpu": get_resource_usage_percentage(stats["usage"]["cpu"], node_allocatable[node_name]["cpu"]),
    #         "allocatable_memory": node_allocatable[node_name]["memory"],
    #         "used_memory": stats["usage"]["memory"],
    #         "pct_used_memory": get_resource_usage_percentage(stats["usage"]["memory"], node_allocatable[node_name]["memory"]),
    #     }

    return result


def get_pod_resource_usage(kube_api: client.CoreV1Api, kube_custom_object_api: client.CustomObjectsApi):
    pods = get_all_pods(kube_api, OPENFAAS_NAMESPACE)
    pod_allocatable = {}

    for pod in pods:
        # Currently, this method only returns data for first container in pod
        pod_allocatable[pod.metadata.name] = {
            "cpu": pod.spec.containers[0].resources.limits["cpu"],
            "memory": pod.spec.containers[0].resources.limits["memory"]
        }

    pods_resource_usage = kube_custom_object_api.list_cluster_custom_object(
        "metrics.k8s.io", "v1beta1", "pods")

    result = {}
    for stats in pods_resource_usage["items"]:
        node_name = stats["metadata"]["name"]

        if node_name not in pod_allocatable:
            continue

        result[node_name] = {
            "node": node_name,
            "cpu_limit": pod_allocatable[node_name]["cpu"],
            "used_cpu": stats["containers"][0]["usage"]["cpu"],
            "pct_used_cpu": get_resource_usage_percentage(stats["containers"][0]["usage"]["cpu"], pod_allocatable[node_name]["cpu"]),
            "memory_limit": pod_allocatable[node_name]["memory"],
            "used_memory": stats["containers"][0]["usage"]["memory"],
            "pct_used_memory": get_resource_usage_percentage(stats["containers"][0]["usage"]["memory"], pod_allocatable[node_name]["memory"]),
        }

    return result


def get_function_average_execution_time(function_name: str, time_window: int):
    query = "(rate(gateway_functions_seconds_sum[{}s]) / rate(gateway_functions_seconds_count[{}s]))".format(
        time_window, time_window)
    response = requests.get(
        "http://{}:{}/api/v1/query?query={}".format(PROMETHEUS_IP, PROMETHEUS_PORT, query))

    for result in response.json()['data']['result']:
        if result['metric']['function_name'] == "{}.{}".format(function_name, OPENFAAS_NAMESPACE):
            return result['value'][1]

    return None


def deploy_openfaas_function(function_name: str, image_name: str, handler_name: str, function_lang: str, labels: Dict[str, str] = None, constraints: Dict[str, str] = None):
    command = "faas-cli deploy --image={} --name={} --handler={} --gateway=http://{}:{} --lang={}".format(quote(
        image_name), quote(function_name), quote(handler_name), quote(GATEWAY_IP), quote(GATEWAY_PORT), quote(function_lang))

    label_names = list(labels.keys())
    for label_name in label_names:
        command += " --label={}={}".format(quote(label_name),
                                           quote(labels[label_name]))

    constraint_names = list(constraints.keys())
    for constraint_name in constraint_names:
        command += " --constraint={}={}".format(
            quote(constraint_name), quote(constraints[constraint_name]))

    subprocess.run("faas-cli login --username {} --password={} && {}".format(
        quote(OPENFAAS_USERNAME), quote(OPENFAAS_PASSWORD), command), shell=True)


def delete_openfaas_function(function_name: str):
    command = "faas-cli delete {}".format(quote(function_name))
    subprocess.run("faas-cli login --username {} --password={} && {}".format(
        quote(OPENFAAS_USERNAME), quote(OPENFAAS_PASSWORD), command), shell=True)


def get_ip_address_of_nodes(kube_api: client.CoreV1Api):
    nodes = get_all_nodes(kube_api)
    ip_addresses = {}
    for node in nodes:
        hostname = None
        ip = None
        for address in node.status.addresses:
            if address.type == 'Hostname':
                hostname = address.address
            if address.type == 'InternalIP':
                ip = address.address
        if (hostname is not None and ip is not None):
            ip_addresses[hostname] = ip
    return ip_addresses


def icmp_ping_test(ip_addresses):
    results = multiping(ip_addresses, count=3, interval=0.33)
    return {result.address: result.avg_rtt for result in results}


if __name__ == "__main__":
    kube_api = get_kube_api_object(config, KUBECONFIG_PATH)
    api = client.CustomObjectsApi()

    ip_addresses = get_ip_address_of_nodes(kube_api)

    num_iterations = 0
    startTime = datetime.now()

    ping_header_written = False
    cpu_usage_header_written = False
    mem_usage_header_written = False
    pods_header_written = False

    while True:
        currentTime = datetime.now()
        print("{} --- Iteration #{} @ {}".format(startTime,
              num_iterations, currentTime))

        try:
            latencies = icmp_ping_test(ip_addresses.values())

            ping_results = [{'seconds_elapsed': INTERVAL * num_iterations, **
                            {key: latencies[ip_addresses[key]] for key in ip_addresses.keys()}}]

            # Log ping metrics for each node in cluster - need to make sure a directory named 'ping' exists before running the script
            with open('ping/ping_{}.csv'.format(startTime.strftime('%Y-%m-%d_%H-%M-%S')), 'a', newline='') as output_file:
                dict_writer = csv.DictWriter(output_file, ping_results[0].keys())

                if not ping_header_written:
                    dict_writer.writeheader()

                ping_header_written = True
                dict_writer.writerows(ping_results)
            
            usage = get_node_resource_usage(kube_api, api)

            cpu_usages = [{'seconds_elapsed': INTERVAL * num_iterations,
                        **{key: usage[key]['pct_used_cpu'] for key in usage.keys()}}]

            # Log CPU usage metrics for each node in cluster - need to make sure a directory named 'cpu' exists before running the script
            with open('cpu/cpu_{}.csv'.format(startTime.strftime('%Y-%m-%d_%H-%M-%S')), 'a', newline='') as output_file:
                dict_writer = csv.DictWriter(output_file, cpu_usages[0].keys())

                if not cpu_usage_header_written:
                    dict_writer.writeheader()

                cpu_usage_header_written = True
                dict_writer.writerows(cpu_usages)

            mem_usages = [{'seconds_elapsed': INTERVAL * num_iterations, **
                        {key: usage[key]['pct_used_memory'] for key in usage.keys()}}]

            # Log memory usage metrics for each node in cluster - need to make sure a directory named 'mem' exists before running the script
            with open('mem/mem{}.csv'.format(startTime.strftime('%Y-%m-%d_%H-%M-%S')), 'a', newline='') as output_file:
                dict_writer = csv.DictWriter(output_file, mem_usages[0].keys())

                if not mem_usage_header_written:
                    dict_writer.writeheader()

                mem_usage_header_written = True
                dict_writer.writerows(mem_usages)

            pods = [{'seconds_elapsed': INTERVAL * num_iterations, 'numOfPods': get_pods()}]
            with open('pods/pods{}.csv'.format(startTime.strftime('%Y-%m-%d_%H-%M-%S')), 'a', newline='') as output_file:
                dict_writer = csv.DictWriter(output_file, pods[0].keys())

                if not pods_header_written:
                    dict_writer.writeheader()

                pods_header_written = True
                dict_writer.writerows(pods)
                print("Pods", pods)
        except Exception as e:
            print("Exception ", e)
        num_iterations += 1
        time.sleep(get_sleep_duration())

### Sample code: ###

## Deploy/destroy OpenFaaS functions programatically 
# deploy_openfaas_function("fn_name", "dockerusername/img_name:tag", "./handlername", "dockerfile", {"com.openfaas.scale.min": "3", "com.openfaas.scale.max": "3"}, {"kubernetes.io/arch": "amd64"})
# delete_openfaas_function("fn_name")


## Gets average execution time for a particular function in a particular time window, e.g. time window of 15 means execution time between now up to and including 15 seconds before now
# print(get_function_average_execution_time("fn_name", INTERVAL))


## Gets number of nodes in the Kubernetes cluster
# print(len(get_all_nodes(kube_api)))


## Gets a list of nodes that match a particular label
# print(get_nodes_matching_label_values(kube_api, "kubernetes.io/hostname", "rpi1"))


## Gets a list of nodes that match any of several labels
# print(len(get_nodes_matching_labels_list(kube_api, {"kubernetes.io/hostname": ["node1", "node2"], "beta.kubernetes.io/arch": ["arm64"]})))


## Gets information about the maximum CPU, memory, storage and number of pods for each node
# nodes_list = kube_api.list_node()
# for i in nodes_list.items:
#     print(i.metadata.name, i.metadata.labels["beta.kubernetes.io/arch"], i.status.allocatable)


## Gets information about the current CPU and memory usage for each node
# api = client.CustomObjectsApi()
# node_usages = get_node_resource_usage(kube_api, api)
# for node in node_usages.keys():
#     print(node, node_usages[node])


## Get and print basic information about each pod in the specified namespace
# pods = get_all_pods(kube_api, OPENFAAS_NAMESPACE)
# for pod in pods:
#     print("%s\t%s\t%s\t%s\t%s\t%s\t%s" % (pod.spec.node_name, pod.status.host_ip, pod.status.pod_ip,
#                                           pod.metadata.namespace, pod.status.phase, pod.metadata.name, pod.metadata.labels["faas_function"]))