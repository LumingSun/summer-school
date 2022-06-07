import torch
import torch.nn as nn

from plan import Operator

operators = [
    "Projection",
    "Selection",
    "Sort",
    "HashAgg",
    "HashJoin",
    "IndexHashJoin",
    "TableScan",
    "IndexScan",
    "TableRowIDScan",
    "IndexRangeScan",
    "IndexFullScan",
    "TableFullScan",
    "IndexLookUp",
    "TableReader",
    "IndexReader",
]


class PlanFeatureCollector:
    def __init__(self):
        # YOUR CODE HERE: you can add features extracted from plan.
        # self.op_count = 0
        self.count_per_op = [0] * len(operators)
        self.rows_per_op = [0] * len(operators)
        self.est_cost_per_op = [0] * len(operators)
        pass

    def add_operator(self, op: Operator):
        # YOUR CODE HERE: update features by op
        # pass
        op_id = op.id.split("_")[0]
        op_idx = operators.index(op_id)
        self.count_per_op[op_idx] += 1
        self.rows_per_op[op_idx] += float(op.est_rows)
        self.est_cost_per_op[op_idx] += float(op.est_cost)

    def walk_operator_tree(self, op: Operator):
        self.add_operator(op)
        for child in op.children:
            self.walk_operator_tree(child)
        return self.est_cost_per_op + self.rows_per_op + self.est_cost_per_op
        # YOUR CODE HERE: concat features as a vector
        # return [self.op_count] + self.count_per_op + self.rows_per_op


class PlanDataset(torch.utils.data.Dataset):
    def __init__(self, plans):
        super().__init__()
        self.data = []
        for plan in plans:
            collector = PlanFeatureCollector()
            features = torch.Tensor(collector.walk_operator_tree(plan.root))
            exec_time = torch.Tensor([plan.exec_time_in_ms()])
            self.data.append((features, exec_time))

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


# Define your model for cost estimation
class YourModel(nn.Module):
    def __init__(self):
        super().__init__()
        # YOUR CODE HERE
        self.seq_model = nn.Sequential(
            nn.Linear(len(operators) * 3, 64),
            nn.ReLU(),
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.seq_model(x)

    def init_weights(self):
        # YOUR CODE HERE
        pass


def estimate_learning(train_plans, test_plans):
    train_dataset = PlanDataset(train_plans)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=10, shuffle=True, num_workers=4
    )
    model = YourModel()
    optm = torch.optim.Adam(model.parameters(), lr=0.001)
    # model.init_weights()

    def loss_fn(est_time, act_time):
        # YOUR CODE HERE: define loss function
        return torch.mean(abs(act_time - est_time) / act_time)

    # YOUR CODE HERE: complete training loop
    num_epoch = 50
    for epoch in range(num_epoch):
        print(f"epoch {epoch} start")
        running_loss = 0.0
        for i, data in enumerate(train_loader):
            x, y = data[0], data[1]
            optm.zero_grad()
            outputs = model(x)
            loss = loss_fn(outputs, y)
            loss.backward()
            optm.step()
            running_loss += loss.item()
            if i % 20 == 0:
                print(
                    f"[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.5f}"
                )
                running_loss = 0.0

    train_est_times, train_act_times = [], []
    for i, data in enumerate(train_loader):
        # YOUR CODE HERE: evaluate on train data
        x, y = data[0], data[1]
        outputs = model(x)
        train_est_times += outputs.tolist()
        train_act_times += y.tolist()

    test_dataset = PlanDataset(test_plans)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=10, shuffle=True, num_workers=1
    )

    test_est_times, test_act_times = [], []
    for i, data in enumerate(test_loader):
        x, y = data[0], data[1]
        outputs = model(x)
        test_est_times += outputs.tolist()
        test_act_times += y.tolist()

    return train_est_times, train_act_times, test_est_times, test_act_times
