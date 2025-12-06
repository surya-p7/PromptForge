import logging, sys, time, uuid, json

logger = logging.getLogger("llm_service")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

def request_logger(request_id=None, **extra):
    if request_id is None:
        request_id = str(uuid.uuid4())
    def _log(level, msg, **kwargs):
        payload = {"rid": request_id, "msg": msg}
        payload.update(extra)
        if kwargs: payload.update(kwargs)
        logger.log(level, json.dumps(payload))
    return _log
