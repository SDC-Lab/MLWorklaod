

import os
import csv
import matplotlib.pyplot as plt
import xlwt
import pandas as pd
from xlwt import Workbook
import os.path as path


servers = [

    ## python - helloworld
    # {'name': 'openfaas', 'host': '10.0.0.169', 'port': 31112, 'path': 'function/myecho'},
    # {'name': 'kubeless', 'host': 'hello.10.0.0.169.nip.io', 'port': 80, 'path': 'kube'},
    # {'name': 'fission', 'host': '10.0.0.169', 'port': 32461, 'path': 'hello'},

    ## python - machine learning cnn
    {'name': 'kubeless', 'host': 'har-cnn-infer.10.0.0.169.nip.io', 'port': 80, 'path': 'har-cnn-infer-kube'},
    {'name': 'openfaas', 'host': '10.0.0.169', 'port': 31112, 'path': 'function/har-cnn-infer'},
    {'name': 'fission', 'host': '10.0.0.169', 'port': 32461, 'path': 'har-cnn-infer'},

    ## python - machine learning lstm
    # {'name': 'kubeless', 'host': 'har-lstm-infer.10.0.0.169.nip.io', 'port': 80, 'path': 'har-lstm-infer-kube'},
    # {'name': 'openfaas', 'host': '10.0.0.169', 'port': 31112, 'path': 'function/har-lstm-infer'}
    # {'name': 'fission', 'host': '10.0.0.169', 'port': 32461, 'path': 'har-lstm-infer'},

    ## javascript - helloword
    # {'name': 'openfaas', 'host': '10.0.0.169', 'port': 31112, 'path': 'function/myechojs'},
    # {'name': 'kubeless', 'host': 'hellojs.10.0.0.169.nip.io', 'port': 80, 'path': 'jskube'},
    # {'name': 'fission', 'host': '10.0.0.169', 'port': 32461, 'path': 'hellojs'},

    # {'name': 'funcx', 'host': '10.0.0.169', 'port': 80, 'path': '', 'test_path': './bin/templates/Funcx-Request.jmx'}
]

configs = [
    {'thread': 1, 'period': 1, 'loop': 1000},
    {'thread': 5, 'period': 1, 'loop': 200},
    # {'thread': 10, 'period': 1, 'loop': 100},
    # {'thread': 15, 'period': 1, 'loop': 67},
    # {'thread': 20, 'period': 1, 'loop': 50},
]

setup_mode = 'lstm-autoscale-1pod-50'

execute_path = "./bin/jmeter"
output_path = "./output/output-%(name)s-loop_%(config)s-%(num)s-%(mode)s.jtl"
test_path = "./bin/templates/HTTP-Serverless-Request.jmx"
report_path = "./result/report-%(name)s-loop_%(config)s-%(num)s-%(mode)s.csv"
average_result_path = './result/average-result-%(name)s-loop_%(config)s-%(mode)s.csv'
image_path = './result/image-result-%(name)s-loop_%(config)s-%(mode)s.png'
jlt_result_file = './result/jlt-result.csv'
field_names = ['Average', 'Throughput', 'Error %', 'Std. Dev.']

cum_plot_sheet = 'plot-result-%(name)s-loop%(config)s-%(mode)s'
cum_plot_path = './result/plot-result-%(mode)s.xls'

ITER_RANGE = 3

def generate_average_report(average_result_file, reports):
    sum_report = {}
    for report in reports:
        with open(report, mode='r') as csv_file:
            csv_reader = csv.DictReader(csv_file)
            for row in csv_reader:
                for field_name in field_names:
                    val = row[field_name]
                    if  '%' in val:
                        val = val.replace('%', '')
                    sum_report[field_name] = sum_report.get(field_name, 0.0) + float(val)
                break
    total_report = len(reports)
    average_report = {key: round(sum_report[key]/total_report, 2) for key in sum_report}
    with open(average_result_file, mode='w') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=field_names)
        writer.writeheader()
        writer.writerow(average_report)


def generage_bar_chart():
    labels = []
    report_data = {}
    for server in servers:
        labels.append(server['name'])
    for field_name in field_names:
        report_data[field_name] = {}
        for config in configs:
            report_data[field_name][config['loop']] = []
            data = []
            for server in servers:
                average_result_file = average_result_path % {'name': server['name'], 'config': config['loop'], 'mode': setup_mode}
                with open(average_result_file, mode='r') as csv_file:
                    csv_reader = csv.DictReader(csv_file)
                    for row in csv_reader:
                        data.append(float(row[field_name]))
                        break
            image_file = image_path % {'name': field_name, 'config': config['loop'], 'mode': setup_mode}
            fig, ax = plt.subplots()
            ax.bar(labels, data, align='center', alpha=0.5, ecolor='black', capsize=10)
            ax.set_ylabel(field_name)
            ax.set_xticks(labels)
            ax.set_xticklabels(labels)
            ax.set_title('Users: %s - Loop: %s' % (str(config['thread']), str(config['loop'])))
            ax.yaxis.grid(True)
            # Save the figure and show
            plt.tight_layout()
            plt.savefig(image_file)
            plt.show()



def generate_worksheet():
    # Workbook is created
    wb = Workbook()
    for server in servers:
        result_file = cum_plot_sheet % {'name': server['name'], 'config': configs[0]['loop'], 'mode': setup_mode}
        sheet1 = wb.add_sheet(result_file)
        data_points = []
        for i in range(0,ITER_RANGE):
            report_path_i = report_path % {'name': server['name'], 'config': configs[0]['loop'], 'num': i, 'mode': setup_mode}
            with open(report_path_i, mode='r') as csv_file:
                csv_reader = csv.DictReader(csv_file)
                for row in csv_reader:
                    data_points.append(float(row[field_names[0]]))
                    break
        data_points.sort()
        prev = 0.0
        index = 0
        for data_point in data_points:
            sheet1.write(index, 0, data_point)
            freq = (1/len(data_points)) + prev
            sheet1.write(index, 1, freq)
            prev = freq
            index+=1
    result_file_path = cum_plot_path % {'mode': setup_mode}
    wb.save(result_file_path)


def generate_rawsheet():
    if not path.exists(jlt_result_file):
        pd.DataFrame({}).to_csv(jlt_result_file)
        df_csv = pd.read_csv(jlt_result_file, header=0)
        df_csv['Testing'] = [0]*5000
        df_csv.to_csv(jlt_result_file, header=True, index=False)
    for server in servers:
        for config in configs:
            data_points = []
            timestamp_points = []
            for i in range(0,ITER_RANGE):
                output_path_i = output_path % {'name': server['name'], 'config': config['loop'], 'num': i, 'mode': setup_mode}
                with open(output_path_i, mode='r') as csv_file:
                    csv_reader = csv.DictReader(csv_file)
                    for row in csv_reader:
                        data_points.append(float(row['Latency']))
                        timestamp_points.append(row['timeStamp'])
            field_name = "%s-loop_%s-%s" % (server['name'], config['loop'], setup_mode)
            df_csv = pd.read_csv(jlt_result_file, header=0)
            df_csv[field_name] = pd.Series(data_points)
            field_name = "%s-loop_%s-timestamp-%s" % (server['name'], config['loop'], setup_mode)
            df_csv[field_name] = pd.Series(timestamp_points)
            df_csv.to_csv(jlt_result_file, header=True, index=False)
            

for server in servers:
    for config in configs:
        server_params = "-Jhost=%s -Jport=%s -Jpath=%s -Jprotocol=http" % (server['host'], server['port'], server['path'])
        config_params = "-Jthread=%s -Jperiod=%s -Jloop=%s" % (config['thread'], config['period'], config['loop'])
        average_result_file = average_result_path % {'name': server['name'], 'config': config['loop'], 'mode': setup_mode}
        reports = []
        test_plan = test_path
        if 'test_path' in server:
            test_plan = server['test_path']
        for i in range(0,ITER_RANGE):
            output_path_i = output_path % {'name': server['name'], 'config': config['loop'], 'num': i, 'mode': setup_mode}
            report_path_i = report_path % {'name': server['name'], 'config': config['loop'], 'num': i, 'mode': setup_mode}
            reports.append(report_path_i)
            execute_test_command = '%s -n -f -t %s -l %s %s %s && exit' % (execute_path, test_plan, output_path_i, server_params, config_params)
            print("## Start ## - %s" % execute_test_command)
            # execute the jmeter file
            os.system(execute_test_command)
            # generate reports
            os.system('./bin/JMeterPluginsCMD.sh --generate-csv %s --input-jtl %s --plugin-type AggregateReport' % (report_path_i, output_path_i))
        generate_average_report(average_result_file, reports)
    generate_rawsheet()


# generage_bar_chart()





