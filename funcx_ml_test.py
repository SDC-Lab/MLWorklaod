

import time
import logging
from functools import partial
from multiprocessing.pool import Pool
import pandas as pd
import os.path as path

from funcx_ml_run import run_test
# from funcx_lstm_run import run_test

from funcx.sdk.client import FuncXClient
fxc = FuncXClient(asynchronous=False)
fxc.throttling_enabled = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logging.getLogger('requests').setLevel(logging.NOTSET)
logger = logging.getLogger(__name__)

configs = [
    # {'thread': 5, 'period': 1, 'loop': 200},
    # {'thread': 10, 'period': 1, 'loop': 100},
    # {'thread': 15, 'period': 1, 'loop': 67},
    # {'thread': 20, 'period': 1, 'loop': 50},
    {'thread': 1, 'period': 1, 'loop': 1000},
]
ITERATION = 4000



def restart():
  def min_func(a, b):
    return a - b
  func_min_uuid = fxc.register_function(min_func)
  tutorial_endpoint = 'a52bacff-fe38-436a-9f6c-61026bd47894'
  res_add = fxc.run(5, 10, function_id=func_min_uuid, endpoint_id=tutorial_endpoint)
  res_add = fxc.run(5, 10, function_id=func_min_uuid, endpoint_id=tutorial_endpoint)
  res_add = fxc.run(5, 10, function_id=func_min_uuid, endpoint_id=tutorial_endpoint)
  res_add = fxc.run(5, 10, function_id=func_min_uuid, endpoint_id=tutorial_endpoint)
  res_add = fxc.run(5, 10, function_id=func_min_uuid, endpoint_id=tutorial_endpoint)
  result = fxc.get(f"tasks/{res_add}")
  while result['status'] != 'success':
      time.sleep(1)
      result = fxc.get(f"tasks/{res_add}")



result_file = './result/funcx-ml-autoscale-1pod-result.csv'
if not path.exists(result_file):
  pd.DataFrame({}).to_csv(result_file)
  df_csv = pd.read_csv(result_file, header=0)
  df_csv['Testing'] = [0]*5000
  df_csv.to_csv(result_file, header=True, index=False)

restart()
for config in configs:
  with Pool(config['thread']) as p:
    logger.info("Start Thread %s" % config)
    data_points = []
    throughputs = []
    timestamps = []
    iter=1
    ts = time.time()   
    index=0
    for i in range(0, ITERATION):
      index+=1
      logger.info('data_points: %s' % data_points)
      logger.info('timestamps: %s' % timestamps)
      if throughputs:
        logger.info('Throughput: %s' % (sum(throughputs)/len(throughputs)))
      logger.info('Index: %s' % index)
      i_time = time.time()
      results = p.map(run_test, ['']*config['thread'])
      throughput = config['thread']/((time.time()-i_time)*1000)
      throughputs.append(throughput)
      for result in results:
        if result:
          data_points.append(result)
          timestamps.append(i_time)
      if len(data_points) > (1000*iter):
        iter+=1
        field_name = "loop_%s" % (config['loop'])
        df = pd.read_csv(result_file, header=0)
        df[field_name] = pd.Series(data_points)
        add_field_name = field_name + '-timestamp'
        df[add_field_name] = pd.Series(timestamps)
        add_field_name = field_name + '-average'
        df[add_field_name] = pd.Series([sum(data_points)/len(data_points)])
        add_field_name = field_name + '-throughput'
        df[add_field_name] = pd.Series([sum(throughputs)/len(throughputs)])
        df.to_csv(result_file, header=True, index=False)
        logger.info("Sleeping between interation-%s..." % iter)
        time.sleep(1000)
        if iter>3:
          break
        restart()
    logging.info('Took %s seconds', time.time() - ts)


field_name = "loop_%s" % (configs[0]['loop'])
df = pd.read_csv(result_file, header=0)
df[field_name] = pd.Series(data_points)
add_field_name = field_name + '-timestamp'
df[add_field_name] = pd.Series(timestamps)
df.to_csv(result_file, header=True, index=False)

timestamps = timestamps[:len(data_points)]



  

# result_file = './result/funcx-result.csv'
# with open(result_file, mode='w') as csv_file:
#   writer = csv.DictWriter(csv_file, fieldnames=['time'])
#   writer.writeheader()
#   with Pool(15) as p:
#     results = p.map(testing, ['hello']*100)
#   for result in results:
#     if result:
#       writer.writerow({'time': result})
