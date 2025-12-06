import time, uuid

def start_trace():
    return {"trace_id": str(uuid.uuid4()), "start": time.time()}

def end_trace(trace):
    trace["end"] = time.time()
    trace["duration_ms"] = (trace["end"]-trace["start"])*1000
    return trace
