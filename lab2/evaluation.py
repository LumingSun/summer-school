import json

from cost_calibration import estimate_calibration
from cost_learning import estimate_learning
from evaluation_utils import draw_act_est_figure
from plan import Plan


if __name__ == '__main__':
    train_json_file = 'lab2/data/train_plans.json'
    test_json_file = 'lab2/data/test_plans.json'
    train_plans, test_plans = [], []

    with open(train_json_file, 'r') as f:
        train_cases = json.load(f)
    for case in train_cases:
        train_plans.append(Plan.parse_plan(case['query'], case['plan']))

    with open(test_json_file, 'r') as f:
        test_cases = json.load(f)
    for case in test_cases:
        test_plans.append(Plan.parse_plan(case['query'], case['plan']))

    act_times = []
    tidb_costs = []
    for p in test_plans:
        act_times.append(p.exec_time_in_ms())
        tidb_costs.append(p.tidb_est_cost())

    # _, _, est_learning_costs, act_learning_times = estimate_learning(train_plans, test_plans)
    est_calibration_costs = estimate_calibration(train_plans, test_plans)

    for i in range(len(est_calibration_costs)):
        if act_times[i] > 3500 and est_calibration_costs[i] < 1.2e9:
            print(test_plans[i].query)

    draw_act_est_figure("tidb", tidb_costs, act_times, True)
    # draw_act_est_figure("learning", est_learning_costs, act_learning_times, True)
    draw_act_est_figure("calibration", est_calibration_costs, act_times, True)
