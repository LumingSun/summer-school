from http.server import HTTPServer, BaseHTTPRequestHandler
import json
import random

import sys 
sys.path.append("..") 
from lab1.range_query import ParsedRangeQuery
import lab1.statistics as stats
# from lab2.cost_learning import estimate_learning
# from lab2.plan import Plan
from lab2.cost_calibration import estimate_plan
host = ('localhost', 8888)


class Resquest(BaseHTTPRequestHandler):
    def handle_cardinality_estimate(self, req_data):
        # YOUR CODE HERE: use your model in lab1
        # print("cardinality_estimate post_data: " + str(req_data))
        req = json.loads(req_data)
        range_query = self.construct_query(req)
        
        stats_json_file = '/home/sunluming/summer-school/lab1/data/title_stats.json'
        columns = ['kind_id', 'production_year', 'imdb_id', 'episode_of_id', 'season_nr', 'episode_nr']
        
        table_stats = stats.TableStats.load_from_json_file(stats_json_file, columns)
        
        est_avi = stats.AVIEstimator.estimate(range_query, table_stats)# * table_stats.row_count
        
        # print("est_avi: ", est_avi)
        res = {"sel":est_avi}
        return json.dumps(res)

    def construct_query(self,req_data:list):
        col_left, col_right = {}, {}
        for item in req_data:
            if item[:2] == "gt":
                p = item[3:-1].split(", ")
                col = p[0].split(".")[2]
                val = int(p[1])
                col_left[col] = int(val)
                
            elif item[:2] =="lt":
                p = item[3:-1].split(", ")
                col = p[0].split(".")[2]
                val = int(p[1])
                col_right[col] = int(val)
        return ParsedRangeQuery("None", "imdb", "kind_id", col_left, col_right)
                    
    def handle_cost_estimate(self, req_data):
        req = json.loads(req_data)
        print("req: ",req)
        # YOUR CODE HERE: use your model in lab2
        # train_json_file = 'lab2/data/train_plans.json'
        # test_json_file = 'lab2/data/test_plans.json'
        # train_plans, test_plans = [], []
        # with open(train_json_file, 'r') as f:
        #     train_cases = json.load(f)
        # for case in train_cases:
        #     train_plans.append(Plan.parse_plan(case['query'], case['plan']))
        # _, _, est_learning_costs, act_learning_times = estimate_learning(train_plans, test_plans)
        # factors = {
        # "cpu": 1,
        # "scan": 1,
        # "net": 1,
        # "seek": 1,
        # }
        # cost = estimate_plan(req, factors, factors)
        # print("cost_estimate post_data: " + str(cost))
        res = {"cost":random.randint(1,100)}
        return json.dumps(res)

    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        req_data = self.rfile.read(content_length)
        resp_data = ""
        if self.path == "/cardinality":
            resp_data = self.handle_cardinality_estimate(req_data)
        elif self.path == "/cost":
            resp_data = self.handle_cost_estimate(req_data)

        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        # print("response data: ",resp_data)
        self.wfile.write(resp_data.encode())


if __name__ == '__main__':
    server = HTTPServer(host, Resquest)
    print("Starting server, listen at: %s:%s" % host)
    server.serve_forever()
