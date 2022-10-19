import time
import logging
import os
from functools import partial
from multiprocessing.pool import Pool
import csv
from funcx.sdk.client import FuncXClient
fxc = FuncXClient(asynchronous=False)
fxc.throttling_enabled = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logging.getLogger('requests').setLevel(logging.CRITICAL)
logger = logging.getLogger(__name__)

def myecho(val):
  return val

def testing(val):
  return time.time()

def run_test(val):
  hello_function = fxc.register_function(myecho)
  # hello_function = "d29fde73-22cd-4728-bb41-e4ab18262cb1"
  endpoint_id = "a52bacff-fe38-436a-9f6c-61026bd47894" # tqlap089@gmail.com - k8s
  # endpoint_id = "174c2042-ffb4-4d37-ba1c-9fbf1d211b73" # tqlap@apcs.vn - node 3
  # endpoint_id = '4b116d3c-1703-4f8f-9f6f-39921e5864df' # Public tutorial endpoint
  try:
    start = time.time()
    res = fxc.run('Hello World', endpoint_id=endpoint_id, function_id=hello_function)
    result = fxc.get(f"tasks/{res}")
    while result['status'] != 'success':
        time.sleep(1)
        result = fxc.get(f"tasks/{res}")
    completion_time = result['completion_t']
    exec_time = (float(completion_time) - start) * 1000
    return exec_time
  except Exception as e:
    logger.error("EXCEPTION - %s" % e)
