def estimate_plan(operator, factors, weights):
    cost = 0.0
    for child in operator.children:
        cost += estimate_plan(child, factors, weights)

    if operator.is_hash_agg():
        # YOUR CODE HERE: design your cost formula for HashAgg
        # cpuCost = inputRows * sessVars.CPUFactor * aggFuncFactor
        input_roww_cnt = 0
        for child in operator.children:
            input_roww_cnt += child.est_row_counts()
        cost += input_roww_cnt * factors["cpu"]
        weights["cpu"] += input_roww_cnt

    elif operator.is_hash_join():
        # hash_join_cost = (build_hashmap_cost + probe_and_pair_cost)
        #   = (build_row_cnt * cpu_fac) + (output_row_cnt * cpu_fac)
        output_row_cnt = operator.est_row_counts()
        build_side = int(operator.children[1].is_build_side())
        build_row_cnt = operator.children[build_side].est_row_counts()

        cost += (build_row_cnt + output_row_cnt) * factors["cpu"]
        weights["cpu"] += build_row_cnt + output_row_cnt

    elif operator.is_sort():
        # cost of sort is:
	    # CPUFactor * batchSize * Log2(batchSize) * (indexRows / batchSize)
        output_row_cnt = operator.est_row_counts()
        cost += output_row_cnt * factors["cpu"]
        weights["cpu"] += output_row_cnt

    elif operator.is_selection():
        # YOUR CODE HERE: design your cost formula for Selection
        #  selection cost: rows * cpu-factor
        output_row_cnt = operator.est_row_counts()
        cost += output_row_cnt * factors["cpu"]
        weights["cpu"] += output_row_cnt

    elif operator.is_projection():
        # YOUR CODE HERE: design your cost formula for Projection
        # cpuCost := count * sessVars.CPUFactor
        output_row_cnt = operator.est_row_counts()
        cost += output_row_cnt * factors["cpu"]
        weights["cpu"] += output_row_cnt


    elif operator.is_table_reader():
        #  (indexRows / batchSize) * batchSize * CPUFactor
        # YOUR CODE HERE: design your cost formula for TableReader
        pass

    elif operator.is_table_scan():
        # YOUR CODE HERE: design your cost formula for TableScan
        pass

    elif operator.is_index_reader():
        # YOUR CODE HERE: design your cost formula for IndexReader
        pass

    elif operator.is_index_scan():
        # YOUR CODE HERE: design your cost formula for IndexScan
        pass

    elif operator.is_index_lookup():
        # index_lookup_cost = net_cost + seek_cost
        #   = (build_row_cnt * build_row_size + probe_row_cnt * probe_row_size) * net_fac +
        #     (build_row_cnt / batch_size) * seek_fac
        build_side = int(operator.children[1].is_build_side())
        build_row_cnt = operator.children[build_side].est_row_counts()
        build_row_size = operator.children[build_side].row_size()
        probe_row_cnt = operator.children[1 - build_side].est_row_counts()
        probe_row_size = operator.children[1 - build_side].row_size()
        batch_size = operator.batch_size()

        cost += (
            build_row_cnt * build_row_size + probe_row_cnt * probe_row_size
        ) * factors["net"]
        weights["net"] += (
            build_row_cnt * build_row_size + probe_row_cnt * probe_row_size
        )

        cost += (build_row_cnt / batch_size) * factors["seek"]
        weights["seek"] += build_row_cnt / batch_size

    else:
        print(operator.id)
        assert 1 == 2  # unknown operator
    return cost


def estimate_calibration(train_plans, test_plans):
    # init factors
    factors = {
        "cpu": 1,
        "scan": 1,
        "net": 1,
        "seek": 1,
    }

    # get training data: factor weights and act_time
    est_costs_before = []
    act_times = []
    weights = []
    for p in train_plans:
        w = {"cpu": 0, "scan": 0, "net": 0, "seek": 0}
        cost = estimate_plan(p.root, factors, w)
        weights.append(w)
        act_times.append(p.exec_time_in_ms())
        est_costs_before.append(cost)

    # YOUR CODE HERE
    # calibrate your cost model with regression and get the best factors
    # factors * weights ==> act_time
    new_factors = {}
    print("--->>> regression cost factors: ", new_factors)

    # evaluation
    est_costs = []
    for p in test_plans:
        w = {"cpu": 0, "scan": 0, "net": 0, "seek": 0}
        cost = estimate_plan(p.root, new_factors, w)
        est_costs.append(cost)

    return est_costs
