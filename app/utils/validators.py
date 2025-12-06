def validate_generate_output(output: dict) -> bool:
    # Basic validation for required keys
    required = ["text", "model", "latency_ms"]
    return all(k in output for k in required)
