

import time
import logging
import os
from functools import partial
from multiprocessing.pool import Pool
import csv
import pandas as pd
from xlwt import Workbook
import os.path as path
from funcx_run import run_test, myecho, testing

from funcx.sdk.client import FuncXClient
fxc = FuncXClient(asynchronous=False)
fxc.throttling_enabled = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logging.getLogger('requests').setLevel(logging.CRITICAL)
logger = logging.getLogger(__name__)

configs = [
    # {'thread': 1, 'period': 1, 'loop': 1000},
    # {'thread': 5, 'period': 1, 'loop': 200},
    # {'thread': 10, 'period': 1, 'loop': 100},
    # {'thread': 15, 'period': 1, 'loop': 67},
    {'thread': 20, 'period': 1, 'loop': 50},
]
ITERATION = 3
REQUESTS = 1000



# result_file = './result/funcx-result.csv'
# if not path.exists(result_file):
#   pd.DataFrame({}).to_csv(result_file)

# for config in configs:
#   logger.info("Start Thread %s" % config)
#   data_points = []
#   throughputs = []
#   ts = time.time()    
#   for i in range(0, ITERATION):
#     i_time = time.time()
#     with Pool(config['thread']) as p:
#         results = p.map(run_test, ['']*REQUESTS)
#     throughput = REQUESTS/((time.time()-i_time)*1000)
#     throughputs.append(throughput)
#     data_points += [result for result in results if result]
#   field_name = "loop_%s" % (config['loop'])
#   df_csv = pd.read_csv(result_file, header=0)
#   df_csv[field_name] = data_points
#   field_name += '-average'
#   df_csv[field_name] = sum(data_points)/len(data_points)
#   field_name += '-throughput'
#   df_csv[field_name] = sum(throughputs)/len(throughputs)
#   df_csv.to_csv(result_file, header=True, index=False)
#   logging.info('Took %s seconds', time.time() - ts)





result_file = './result/funcx-result.xlsx'
if not path.exists(result_file):
  pd.DataFrame({}).to_excel(result_file,index=False)

for config in configs:
  logger.info("Start Thread %s" % config)
  data_points = []
  throughputs = []
  ts = time.time()    
  for i in range(0, ITERATION):
    i_time = time.time()
    with Pool(config['thread']) as p:
        results = p.map(run_test, ['']*REQUESTS)
    throughput = REQUESTS/((time.time()-i_time)*1000)
    throughputs.append(throughput)
    data_points += [result for result in results if result]
  field_name = "loop_%s" % (config['loop'])
  df = pd.read_excel(result_file)
  df[field_name] = pd.Series(data_points)
  field_name += '-average'
  df[field_name] = pd.Series([sum(data_points)/len(data_points)])
  field_name += '-throughput'
  df[field_name] = pd.Series([sum(throughputs)/len(throughputs)])
  df.to_excel(result_file,index=False)
  logging.info('Took %s seconds', time.time() - ts)



# result_file = './result/funcx-result.csv'
# with open(result_file, mode='w') as csv_file:
#   writer = csv.DictWriter(csv_file, fieldnames=['time'])
#   writer.writeheader()
#   with Pool(15) as p:
#     results = p.map(testing, ['hello']*100)
#   for result in results:
#     if result:
#       writer.writerow({'time': result})
