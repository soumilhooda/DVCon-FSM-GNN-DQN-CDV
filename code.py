# Import necessary libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data
from torch_geometric.nn import GINConv
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
from collections import defaultdict
import random
import torch.multiprocessing as mp

###############################
# Part 1: Enhanced GNN-Based Formal Verification of a Complex FSM
###############################

# Generate a complex FSM with 5000 states and intricate transitions
num_states = 5000
states = list(range(num_states))

# Randomly generate transitions between states to simulate a complex FSM
np.random.seed(42)
transitions = []
for _ in range(num_states * 5):  # Each state has approximately 5 outgoing transitions
    src = np.random.choice(states)
    dst = np.random.choice(states)
    if src != dst:
        transitions.append((src, dst))

# Remove duplicate transitions
transitions = list(set(transitions))

# Create edge index for PyG
edges = torch.tensor(transitions, dtype=torch.long).t().contiguous()

# Generate node features (could be state encodings or properties)
node_features = torch.randn((num_states, 128))  # Increased feature size for complexity

###############################
# Injecting Specific Faults and Preparing Data
###############################

# Initialize lists to store faulty transitions and their fault types
faulty_transitions = []
faulty_labels = []

# Fault 1: Deadlocks - transitions to non-existent states (creating new nodes)
num_deadlocks = int(0.05 * len(transitions))  # 5% of transitions
deadlock_transitions = []
deadlock_states = []
for _ in range(num_deadlocks):
    src = np.random.choice(states)
    dst = num_states + len(deadlock_states)  # New non-existent state
    deadlock_states.append(dst)
    deadlock_transitions.append((src, dst))
    faulty_transitions.append((src, dst))
    faulty_labels.append(1)  # Label 1 for deadlocks

# Update the number of states and node features to include deadlock states
num_states += len(deadlock_states)
states.extend(deadlock_states)
deadlock_node_features = torch.randn((len(deadlock_states), node_features.shape[1]))
node_features = torch.cat([node_features, deadlock_node_features], dim=0)

# Fault 2: Unreachable states - new states with no incoming transitions
num_unreachable = int(0.05 * num_states)  # 5% of states
unreachable_states = list(range(num_states, num_states + num_unreachable))
num_states += num_unreachable
states.extend(unreachable_states)
unreachable_node_features = torch.randn((num_unreachable, node_features.shape[1]))
node_features = torch.cat([node_features, unreachable_node_features], dim=0)
# No transitions leading to these states

# Fault 3: Unintended loops - create loops in states that shouldn't have them
num_loops = int(0.05 * len(transitions))  # 5% of transitions
loop_transitions = []
for _ in range(num_loops):
    src = np.random.choice(states)
    loop_transitions.append((src, src))  # Self-loop
    faulty_transitions.append((src, src))
    faulty_labels.append(3)  # Label 3 for unintended loops

# Combine all faulty transitions
# Note: Faulty labels are already collected during fault injection

# Combine original transitions and faulty transitions
all_transitions = transitions + faulty_transitions
all_labels = [0] * len(transitions) + faulty_labels  # Label 0 for valid transitions

# Create edge index for PyG
all_edges = torch.tensor(all_transitions, dtype=torch.long).t().contiguous()

# Recalculate degrees and edge features for all transitions
degrees = np.zeros(num_states)
in_degrees = np.zeros(num_states)
out_degrees = np.zeros(num_states)
for src, dst in all_transitions:
    degrees[src] += 1
    degrees[dst] += 1
    out_degrees[src] += 1
    in_degrees[dst] += 1

# Create edge features for all transitions
edge_features = []
for src, dst in all_transitions:
    src_degree = degrees[src]
    dst_degree = degrees[dst]
    src_out_degree = out_degrees[src]
    dst_in_degree = in_degrees[dst]
    is_self_loop = 1 if src == dst else 0
    edge_features.append([src_degree, dst_degree, src_out_degree, dst_in_degree, is_self_loop])
edge_features = torch.tensor(edge_features, dtype=torch.float)

# Shuffle the data
perm = torch.randperm(len(all_labels))
all_edges = all_edges[:, perm]
edge_features = edge_features[perm]
all_labels = np.array(all_labels)[perm.numpy()]

# Update FSM data with all edges and edge features
fsm_faulty_data = Data(x=node_features, edge_index=all_edges, edge_attr=edge_features)

# Split data into training and testing sets
edge_indices = np.arange(all_edges.shape[1])
train_indices, test_indices, train_labels, test_labels = train_test_split(
    edge_indices,
    all_labels,
    test_size=0.2,
    stratify=all_labels,
    random_state=42
)

train_edges = all_edges[:, train_indices]
train_edge_features = edge_features[train_indices]
train_labels = torch.tensor(train_labels, dtype=torch.long)
test_edges = all_edges[:, test_indices]
test_edge_features = edge_features[test_indices]
test_labels = torch.tensor(test_labels, dtype=torch.long)

###############################
# Define the Enhanced EdgeFeatureGNN Model
###############################

class EnhancedEdgeFeatureGNN(nn.Module):
    def __init__(self, input_dim, edge_input_dim, hidden_dim, num_classes):
        super(EnhancedEdgeFeatureGNN, self).__init__()
        self.node_emb = nn.Linear(input_dim, hidden_dim)
        self.edge_emb = nn.Linear(edge_input_dim, hidden_dim)
        nn.init.xavier_uniform_(self.node_emb.weight)
        nn.init.xavier_uniform_(self.edge_emb.weight)

        self.conv1 = GINConv(
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
            ), train_eps=True
        )
        self.conv2 = GINConv(
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
            ), train_eps=True
        )
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, x, edge_index, edge_attr):
        x = self.node_emb(x)
        edge_attr = self.edge_emb(edge_attr)

        # First GINConv layer
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)

        # Second GINConv layer
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)

        # Edge feature concatenation
        edge_src = x[edge_index[0]]
        edge_dst = x[edge_index[1]]
        edge_features = torch.cat([edge_src, edge_dst, edge_attr], dim=1)
        edge_logits = self.edge_mlp(edge_features)
        return edge_logits

# Initialize the enhanced GNN model
input_dim = node_features.shape[1]
edge_input_dim = edge_features.shape[1]
hidden_dim = 256  # Increased hidden dimension for better capacity
num_classes = 4  # 0: Valid, 1: Deadlock, 2: Unreachable, 3: Loop
edge_gnn_model = EnhancedEdgeFeatureGNN(input_dim=input_dim, edge_input_dim=edge_input_dim, hidden_dim=hidden_dim, num_classes=num_classes)
optimizer_edge_gnn = optim.Adam(edge_gnn_model.parameters(), lr=0.0005, weight_decay=1e-5)
criterion_edge_gnn = nn.CrossEntropyLoss()

# Learning rate scheduler
scheduler = optim.lr_scheduler.StepLR(optimizer_edge_gnn, step_size=20, gamma=0.5)

###############################
# Training the Enhanced EdgeFeatureGNN Model
###############################

epochs = 100  # Increased to 100 epochs
batch_size = 8192

edge_gnn_model.train()
for epoch in range(epochs):
    epoch_loss = 0
    num_batches = int(np.ceil(train_edges.shape[1] / batch_size))
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, train_edges.shape[1])
        batch_edges = train_edges[:, start_idx:end_idx]
        batch_edge_features = train_edge_features[start_idx:end_idx]
        batch_labels = train_labels[start_idx:end_idx]

        optimizer_edge_gnn.zero_grad()
        out = edge_gnn_model(node_features, batch_edges, batch_edge_features)
        loss = criterion_edge_gnn(out, batch_labels)
        loss.backward()
        optimizer_edge_gnn.step()

        epoch_loss += loss.item() * batch_labels.size(0)

    epoch_loss /= train_edges.shape[1]
    scheduler.step()

    if (epoch + 1) % 5 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.6f}')

###############################
# Evaluating the Enhanced EdgeFeatureGNN Model
###############################

edge_gnn_model.eval()
with torch.no_grad():
    out = edge_gnn_model(node_features, test_edges, test_edge_features)
    _, predictions = torch.max(out, 1)

    accuracy = accuracy_score(test_labels.numpy(), predictions.numpy())
    conf_matrix = confusion_matrix(test_labels.numpy(), predictions.numpy(), labels=[0,1,2,3])
    class_names = ['Valid', 'Deadlock', 'Unreachable', 'Loop']
    class_report = classification_report(test_labels.numpy(), predictions.numpy(), labels=[0,1,2,3], target_names=class_names, zero_division=0)

    print("\nEnhanced EdgeFeatureGNN Formal Verification Results:")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print("Confusion Matrix:")
    print(conf_matrix)
    print("\nClassification Report:")
    print(class_report)

    # Functional Coverage Metrics per Fault Type
    for i, fault_name in enumerate(class_names[1:], start=1):
        fault_indices = test_labels.numpy() == i
        if fault_indices.sum() > 0:
            correct_faults = (predictions.numpy() == i) & fault_indices
            fault_coverage = correct_faults.sum() / fault_indices.sum() * 100
            print(f"Fault Coverage for {fault_name}: {fault_coverage:.2f}%")
        else:
            print(f"No instances of {fault_name} in test set.")

###############################
# Part 2: RL-Based Coverage-Driven Verification (CDV) on Complex FSM
###############################

# Define the environment for the RL agent
class ComplexFSMEnvironment:
    def __init__(self, states, transitions, max_steps=1000):
        self.states = states
        self.transitions = transitions
        self.transition_dict = defaultdict(list)
        for src, dst in transitions:
            self.transition_dict[src].append(dst)
        self.num_states = len(states)
        self.current_state = None
        self.total_steps = 0
        self.max_steps = max_steps
        self.coverage = set()
        self.valid_start_states = [state for state in self.states if state in self.transition_dict]

    def reset(self):
        self.current_state = random.choice(self.valid_start_states)
        self.total_steps = 0
        self.coverage = set()
        return self.current_state

    def step(self, action):
        self.total_steps += 1
        if action in self.transition_dict.get(self.current_state, []):
            self.current_state = action
            reward = 1 if self.current_state not in self.coverage else 0
            self.coverage.add(self.current_state)
            done = self.total_steps >= self.max_steps
            return self.current_state, reward, done
        else:
            # Invalid action
            reward = -1
            done = True
            return self.current_state, reward, done

    def get_valid_actions(self, state=None):
        if state is None:
            state = self.current_state
        return self.transition_dict.get(state, [])

# Use Q-learning with multi-processing to handle the complexity

def train_agent(agent_id, q_table, env, num_episodes, epsilon, learning_rate, discount_factor, results):
    for episode in range(num_episodes):
        state = env.reset()
        done = False

        while not done:
            if random.uniform(0, 1) < epsilon:
                # Explore: select a random valid action
                valid_actions = env.get_valid_actions()
                if not valid_actions:
                    break
                action = random.choice(valid_actions)
            else:
                # Exploit: select the action with max Q-value
                valid_actions = env.get_valid_actions()
                if not valid_actions:
                    break
                q_values = [q_table.get((state, a), 0.0) for a in valid_actions]
                max_q = max(q_values)
                max_actions = [a for a, q in zip(valid_actions, q_values) if q == max_q]
                action = random.choice(max_actions)

            next_state, reward, done = env.step(action)

            # Update Q-value
            best_next_actions = env.get_valid_actions(next_state)
            if best_next_actions:
                best_next_q = max([q_table.get((next_state, a), 0.0) for a in best_next_actions])
            else:
                best_next_q = 0
            q_table[(state, action)] = q_table.get((state, action), 0.0) + learning_rate * (reward + discount_factor * best_next_q - q_table.get((state, action), 0.0))

            state = next_state

        if (episode + 1) % 100 == 0:
            coverage = len(env.coverage)
            results.put((agent_id, episode + 1, coverage))

# Initialize shared Q-table
manager = mp.Manager()
q_table = manager.dict()

# Training parameters
num_agents = 4  # Number of parallel agents
num_episodes_per_agent = 500  # Increased for better learning
epsilon = 0.1
learning_rate = 0.05
discount_factor = 0.99

# Initialize environment
# Exclude faulty transitions for the RL agent
valid_transitions = transitions  # Only valid transitions
env = ComplexFSMEnvironment(states, valid_transitions, max_steps=5000)

# Initialize multiprocessing
processes = []
results = manager.Queue()
for agent_id in range(num_agents):
    p = mp.Process(target=train_agent, args=(agent_id, q_table, env, num_episodes_per_agent, epsilon, learning_rate, discount_factor, results))
    processes.append(p)
    p.start()

for p in processes:
    p.join()

# Collect results
while not results.empty():
    agent_id, episode, coverage = results.get()
    print(f"Agent {agent_id}, Episode {episode}, Coverage: {coverage}/{num_states}")

# Evaluate the agent
state = env.reset()
done = False
visited_states = set()
total_rewards = 0

while not done:
    valid_actions = env.get_valid_actions()
    if not valid_actions:
        break
    q_values = [q_table.get((state, a), 0.0) for a in valid_actions]
    max_q = max(q_values)
    max_actions = [a for a, q in zip(valid_actions, q_values) if q == max_q]
    action = random.choice(max_actions)
    next_state, reward, done = env.step(action)
    visited_states.add(next_state)
    total_rewards += reward
    state = next_state

coverage_percentage = (len(visited_states) / num_states) * 100
print(f"\nCoverage Achieved by Q-learning Agent: {coverage_percentage:.2f}%")
print(f"Total Rewards Collected: {total_rewards}")
