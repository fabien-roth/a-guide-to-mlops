import requests

# Prometheus API URL (Update with the correct external Prometheus IP address)
PROMETHEUS_URL = "http://34.65.186.74:9090/api/v1/query"

# Queries from the updated rule set
queries = {
    # HTTP Requests Metrics
    "Total HTTP Requests per model_type": "sum(rate(bentoml_service_request_total[5m])) by (model_type)",

    # Latency Metrics
    "95th Percentile Latency per model_type": "histogram_quantile(0.95, sum(rate(bentoml_service_request_duration_seconds_bucket[5m])) by (le, model_type))",
    "90th Percentile Latency per model_type": "histogram_quantile(0.90, sum(rate(bentoml_service_request_duration_seconds_bucket[5m])) by (le, model_type))",
    "Average Latency per model_type": "sum(rate(bentoml_service_request_duration_seconds_sum[5m])) by (model_type) / sum(rate(bentoml_service_request_duration_seconds_count[5m])) by (model_type)",
    "Request Latency Distribution per model_type": "sum(rate(bentoml_service_request_duration_seconds_bucket[5m])) by (le, model_type)",

    # Per-job Metrics
    "Total HTTP Requests per job": "sum(rate(bentoml_service_request_total[5m])) by (job)",
    "Total Latency per job": "sum(rate(bentoml_service_request_duration_seconds_sum[5m])) by (job)",

    # Additional Queries for Baseline
    "Baseline: Total HTTP Requests": "sum(rate(bentoml_service_request_total{model_type='baseline'}[5m]))",
    "Baseline: 95th Percentile Latency": "histogram_quantile(0.95, sum(rate(bentoml_service_request_duration_seconds_bucket{model_type='baseline'}[5m])) by (le))",

    # Additional Queries for ptq_integer
    "PTQ Integer: Total HTTP Requests": "sum(rate(bentoml_service_request_total{model_type='ptq_integer'}[5m]))",
    "PTQ Integer: 95th Percentile Latency": "histogram_quantile(0.95, sum(rate(bentoml_service_request_duration_seconds_bucket{model_type='ptq_integer'}[5m])) by (le))",

    # Additional Queries for ptq_float16
    "PTQ Float16: Total HTTP Requests": "sum(rate(bentoml_service_request_total{model_type='ptq_float16'}[5m]))",
    "PTQ Float16: 95th Percentile Latency": "histogram_quantile(0.95, sum(rate(bentoml_service_request_duration_seconds_bucket{model_type='ptq_float16'}[5m])) by (le))",

    # Additional Queries for ptq_dynamic
    "PTQ Dynamic: Total HTTP Requests": "sum(rate(bentoml_service_request_total{model_type='ptq_dynamic'}[5m]))",
    "PTQ Dynamic: 95th Percentile Latency": "histogram_quantile(0.95, sum(rate(bentoml_service_request_duration_seconds_bucket{model_type='ptq_dynamic'}[5m])) by (le))",

    # Memory and CPU Metrics
    "Memory Usage (percent)": "(1 - (node_memory_free_bytes / node_memory_total_bytes)) * 100",
    "CPU Usage (percent) per core": "100 * (1 - rate(node_cpu_seconds_total{mode='idle'}[5m]))",
    "Total CPU Usage (percent) across all cores": "100 * (1 - avg(rate(node_cpu_seconds_total{mode='idle'}[5m])))",
}


def test_queries(prometheus_url, queries):
    print("Testing Prometheus Queries...")
    failed_queries = []

    for description, query in queries.items():
        print(f"\nTesting Query: {description}")
        print(f"Query String: {query}")  # Log the query string for debugging
        try:
            response = requests.get(prometheus_url, params={"query": query})
            if response.status_code == 200:
                result = response.json().get('data', {}).get('result', [])
                if result:
                    print(f"✅ Query succeeded. Results: {result}")
                else:
                    print(f"⚠️ Query returned no data. Check if the labels and rules are correct.")
                    failed_queries.append((description, "No data returned"))
            else:
                print(f"❌ Query failed. HTTP Status: {response.status_code}, Response: {response.text}")
                failed_queries.append((description, f"HTTP Error: {response.status_code}"))
        except Exception as e:
            print(f"❌ Query failed with error: {e}")
            failed_queries.append((description, str(e)))

    # Summary
    print("\n=== Query Test Summary ===")
    if failed_queries:
        print(f"{len(failed_queries)} query(ies) failed:")
        for desc, error in failed_queries:
            print(f"- {desc}: {error}")
    else:
        print("All queries succeeded!")


if __name__ == "__main__":
    test_queries(PROMETHEUS_URL, queries)
