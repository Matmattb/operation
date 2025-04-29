import numpy as np
from collections import deque
import time
import random
import matplotlib.pyplot as plt

# Function to read flow network from .txt file
def read_flow_network(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    n = int(lines[0].strip())
    capacity = np.zeros((n, n), dtype=int)
    for i in range(n):
        row = list(map(int, lines[i + 1].strip().split()))
        for j in range(n):
            capacity[i][j] = row[j]
    return n, capacity

# Function to display a matrix (capacity, cost, or Bellman’s table)
def display_matrix(matrix, title, labels=None):
    print(f"\n{title}:")
    n = matrix.shape[0]
    if labels:
        header = "    " + " ".join(f"{lbl:>4}" for lbl in labels)
        print(header)
        for i, row in enumerate(matrix):
            row_str = f"{labels[i]:>2} |" + " ".join(f"{x:>4}" if x != float('inf') else "  inf" for x in row)
            print(row_str)
    else:
        for row in matrix:
            print(" ".join(f"{x:>2}" for x in row))

# Ford-Fulkerson with Edmonds-Karp (BFS for shortest augmenting paths)
def ford_fulkerson(n, capacity, trace_file):
    flow = np.zeros((n, n), dtype=int)
    residual = capacity.copy()
    max_flow = 0
    vertices = [chr(ord('s') if i == 0 else ord('t') if i == n-1 else ord('a') + i - 1) for i in range(n)]

    with open(trace_file, 'w', encoding='utf-8') as f:
        f.write("Ford-Fulkerson Execution Trace\n")
        display_matrix(residual, "Capacity table display", vertices)
        f.write("\nCapacity table display:\n")
        for row in residual:
            f.write(" ".join(f"{x:>2}" for x in row) + "\n")
        f.write("\nThe initial residual graph is the starting graph.\n")

        iteration = 1
        while True:
            parent = [-1] * n
            parent[0] = -2
            queue = deque([0])
            path_flow = float('inf')
            path = []
            
            while queue:
                u = queue.popleft()
                for v in range(n):
                    if parent[v] == -1 and residual[u][v] > 0:
                        parent[v] = u
                        queue.append(v)
                        if v == n - 1:
                            break
            
            if parent[n - 1] == -1:
                break
            
            f.write(f"\n* Iteration {iteration}:\n\nBreadth-first search:\n")
            queue = deque([0])
            visited = {0}
            bfs_trace = []
            while queue:
                u = queue.popleft()
                neighbors = []
                for v in range(n):
                    if residual[u][v] > 0 and v not in visited:
                        neighbors.append(vertices[v])
                        visited.add(v)
                        queue.append(v)
                if neighbors:
                    bfs_trace.append(f"{vertices[u]}{''.join(neighbors)};")
                    for v in range(n):
                        if residual[u][v] > 0 and parent[v] == u:
                            f.write(f"    Π({vertices[v]})={vertices[u]}\n")
            
            f.write("\n".join(bfs_trace) + "\n")
            
            v = n - 1
            while v != 0:
                u = parent[v]
                path_flow = min(path_flow, residual[u][v])
                path.append((u, v))
                v = u
            path.reverse()
            
            for u, v in path:
                residual[u][v] -= path_flow
                residual[v][u] += path_flow
                flow[u][v] += path_flow
            
            max_flow += path_flow
            path_str = "".join(vertices[u] for u, _ in path) + vertices[path[-1][1]]
            f.write(f"Detection of an improving chain: {path_str} with a flow {path_flow}.\n")
            f.write("\nModifications to the residual graph:\n")
            display_matrix(residual, f"Residual graph after iteration {iteration}", vertices)
            for row in residual:
                f.write(" ".join(f"{x:>2}" for x in row) + "\n")
            
            iteration += 1
        
        f.write("\n* Max flow display:\n")
        flow_display = np.zeros((n, n), dtype=object)
        for i in range(n):
            for j in range(n):
                flow_display[i][j] = f"{flow[i][j]}/{capacity[i][j]}" if capacity[i][j] > 0 else "0"
        display_matrix(flow_display, "Max flow display", vertices)
        for row in flow_display:
            f.write(" ".join(f"{x:>5}" for x in row) + "\n")
        f.write(f"\nValue of the max flow = {max_flow}\n")
    
    return max_flow, flow

# Push-Relabel algorithm
def push_relabel(n, capacity, trace_file):
    flow = np.zeros((n, n), dtype=int)
    excess = [0] * n
    height = [0] * n
    height[0] = n
    residual = capacity.copy()
    vertices = [chr(ord('s') if i == 0 else ord('t') if i == n-1 else ord('a') + i - 1) for i in range(n)]
    
    with open(trace_file, 'w', encoding='utf-8') as f:
        f.write("Push-Relabel Execution Trace\n")
        display_matrix(capacity, "Initial capacity table", vertices)
        f.write("\nInitial capacity table:\n")
        for row in capacity:
            f.write(" ".join(f"{x:>2}" for x in row) + "\n")
        
        for v in range(n):
            if capacity[0][v] > 0:
                flow[0][v] = capacity[0][v]
                residual[0][v] = 0
                residual[v][0] = capacity[0][v]
                excess[v] = capacity[0][v]
                excess[0] -= capacity[0][v]
        
        f.write("\nAfter initialization:\n")
        f.write(f"Excess: {excess}\n")
        f.write(f"Heights: {height}\n")
        
        iteration = 1
        while any(e > 0 for e in excess[1:n-1]):
            max_height = -1
            u = -1
            for i in range(1, n-1):
                if excess[i] > 0 and height[i] > max_height:
                    max_height = height[i]
                    u = i
                elif excess[i] > 0 and height[i] == max_height and u != -1:
                    if vertices[i] < vertices[u]:
                        u = i
            if u == -1:
                break
            
            pushed = False
            for v in [n-1] + sorted(range(n), key=lambda x: vertices[x]):
                if v != u and residual[u][v] > 0 and height[u] == height[v] + 1:
                    push_flow = min(excess[u], residual[u][v])
                    flow[u][v] += push_flow
                    residual[u][v] -= push_flow
                    residual[v][u] += push_flow
                    excess[u] -= push_flow
                    excess[v] += push_flow
                    pushed = True
                    f.write(f"\nIteration {iteration}: Push {push_flow} from {vertices[u]} to {vertices[v]}\n")
                    f.write(f"Excess: {excess}\n")
                    f.write(f"Heights: {height}\n")
                    break
            
            if not pushed:
                min_height = float('inf')
                for v in range(n):
                    if residual[u][v] > 0:
                        min_height = min(min_height, height[v])
                if min_height != float('inf'):
                    height[u] = 1 + min_height
                    f.write(f"\nIteration {iteration}: Relabel {vertices[u]} to height {height[u]}\n")
                    f.write(f"Excess: {excess}\n")
                    f.write(f"Heights: {height}\n")
            
            iteration += 1
        
        max_flow = sum(flow[i][n-1] for i in range(n))
        f.write("\nFinal flow:\n")
        display_matrix(flow, "Final flow", vertices)
        for row in flow:
            f.write(" ".join(f"{x:>2}" for x in row) + "\n")
        f.write(f"\nMaximum flow value: {max_flow}\n")
    
    return max_flow, flow

# Minimum-Cost Flow using Bellman-Ford on Residual Graph
def min_cost_flow(n, capacity, cost, target_flow, trace_file):
    flow = np.zeros((n, n), dtype=int)
    residual = capacity.copy()
    total_flow = 0
    total_cost = 0
    # Ensure cost matrix matches shape, handle potential non-edges
    # Cost should be high for non-existent edges if not already handled
    # We assume cost[i][j] = 0 if capacity[i][j] = 0 initially
    vertices = [chr(ord('s') if i == 0 else ord('t') if i == n-1 else ord('a') + i - 1) for i in range(n)]

    with open(trace_file, 'w', encoding='utf-8') as f:
        f.write("Minimum-Cost Flow Execution Trace (Corrected Bellman-Ford)\n")
        # Display initial state (optional but good for trace)
        f.write("\nInitial Capacity:\n")
        # Use display_matrix for consistent output (if desired, or keep simple write)
        for r in capacity: f.write(" ".join(map(str, r)) + "\n")
        f.write("\nInitial Cost:\n")
        for r in cost: f.write(" ".join(map(str, r)) + "\n")
        f.write("\nTarget Flow: {}\n".format(target_flow))

        iteration = 1
        while total_flow < target_flow:
            # Bellman-Ford on the residual graph to find min-cost path
            dist = [float('inf')] * n
            parent = [-1] * n
            edge_from = [-1] * n # Track the edge u used to reach v: parent[v] = u
            dist[0] = 0
            # Bellman-Ford Table (optional for detailed trace)
            # bellman_table = [[float('inf')] * n for _ in range(n)]
            # bellman_table[0] = dist.copy()

            # Run Bellman-Ford for n-1 iterations
            for k in range(n - 1):
                updated = False
                for u in range(n):
                    if dist[u] == float('inf'): continue # Skip unreachable nodes

                    for v in range(n):
                        # Check forward edge (u, v) in residual graph
                        if residual[u][v] > 0 and dist[u] + cost[u][v] < dist[v]:
                            dist[v] = dist[u] + cost[u][v]
                            parent[v] = u
                            # edge_from[v] = u # Store edge if needed separately
                            updated = True

                        # Check backward edge (v, u) in residual graph
                        # This corresponds to flow previously pushed u -> v
                        # Residual capacity exists on (v, u), cost is -cost[u][v]
                        if residual[v][u] > 0 and dist[v] - cost[u][v] < dist[u]:
                             # This relaxation seems wrong, Bellman-Ford relaxes edges *from* reachable nodes
                             # Let's reconsider the relaxation structure. Iterate through all potential edges.
                             pass # Re-evaluating the standard Bellman-Ford relaxation below

            # Corrected Bellman-Ford Relaxation Loop
            dist = [float('inf')] * n
            parent = [-1] * n
            dist[0] = 0
            bellman_table = [[float('inf')] * n for _ in range(n)] # For trace
            bellman_table[0] = dist.copy()

            for k in range(1, n): # Iterate n-1 times
                bellman_table[k] = bellman_table[k-1].copy() # Start with previous distances
                updated_in_iter = False
                for u in range(n):
                    for v in range(n):
                        # Edge (u, v) - Forward in original graph
                        if residual[u][v] > 0 and bellman_table[k-1][u] != float('inf'):
                            new_dist_v = bellman_table[k-1][u] + cost[u][v]
                            if new_dist_v < bellman_table[k][v]:
                                bellman_table[k][v] = new_dist_v
                                parent[v] = u
                                updated_in_iter = True

                        # Edge (u, v) - Backward in original graph (means edge v->u used)
                        # Residual capacity exists on u->v if flow was pushed v->u
                        # Cost is -cost[v][u]
                        # Need original cost[v][u]
                        if residual[u][v] > 0 and capacity[u][v] == 0 and capacity[v][u] > 0: # Check if it's purely a backward edge residual
                           if bellman_table[k-1][u] != float('inf'):
                               # Make sure cost[v][u] exists if accessing it
                               neg_cost_vu = -cost[v][u] if cost[v][u] != 0 else float('inf') # Handle potential 0 cost for non-edge
                               if neg_cost_vu != float('inf'):
                                   new_dist_v = bellman_table[k-1][u] + neg_cost_vu
                                   if new_dist_v < bellman_table[k][v]:
                                       bellman_table[k][v] = new_dist_v
                                       parent[v] = u
                                       updated_in_iter = True

                if not updated_in_iter: # Optimization: stop if no changes
                    # Fill rest of table for display if needed
                    for fill_k in range(k + 1, n):
                        bellman_table[fill_k] = bellman_table[k]
                    break
            # Final distances are in bellman_table[k] or bellman_table[n-1]
            final_dist = bellman_table[min(k, n-1)] # Use distances from last effective iteration

            # Write Bellman table to trace
            f.write(f"\n--- Iteration {iteration} ---\n")
            f.write("Bellman-Ford Distances (Rows: Iteration k, Cols: Nodes):\n")
            # Use display_matrix if adapted, otherwise simple print
            f.write("     " + " ".join(f"{lbl:>5}" for lbl in vertices) + "\n")
            for k_row, row_data in enumerate(bellman_table[:min(k, n-1)+1]):
                 f.write(f"k={k_row:<2} [" + " ".join(f"{x:5.0f}" if x != float('inf') else "  inf" for x in row_data) + " ]\n")

            # Check for negative cycles (optional but good practice)
            # Run one more iteration check
            # ...

            # If sink is unreachable, break
            if final_dist[n-1] == float('inf'):
                f.write("\nSink node t is unreachable. No more augmenting paths.\n")
                break

            # Reconstruct path from parent array
            path = []
            curr = n - 1
            while curr != 0:
                prev = parent[curr]
                if prev == -1:
                    # Should not happen if final_dist[n-1] is not inf and s=0
                    f.write("\nError reconstructing path.\n")
                    path = None # Indicate error
                    break
                path.append((prev, curr))
                curr = prev
            
            if path is None: break # Stop if path reconstruction failed
            path.reverse()

            # Find path flow capacity and cost
            path_flow = float('inf')
            path_cost_per_unit = final_dist[n-1] # Bellman-Ford gives shortest path cost

            for u, v in path:
                path_flow = min(path_flow, residual[u][v])

            # Limit flow by remaining target flow
            path_flow = min(path_flow, target_flow - total_flow)

            if path_flow <= 0: # Should not happen if target_flow not met and path exists
                f.write("\nPath found but path flow is zero. Stopping.\n")
                break

            # Augment flow and update residual graph/costs
            total_flow += path_flow
            total_cost += path_flow * path_cost_per_unit # Use cost from Bellman-Ford

            path_str = "".join(vertices[u] for u, _ in path) + vertices[path[-1][1]]
            f.write(f"\nFound Min-Cost Path: {path_str}")
            f.write(f"\nPath Cost per unit: {path_cost_per_unit}, Path Flow: {path_flow}")
            f.write(f"\nTotal Flow so far: {total_flow}, Total Cost so far: {total_cost}")

            for u, v in path:
                residual[u][v] -= path_flow
                residual[v][u] += path_flow

                # Update net flow matrix (careful: flow[u][v] is net flow u->v)
                if capacity[u][v] > 0: # Edge (u,v) exists in original graph
                    flow[u][v] += path_flow
                else: # Edge (u,v) must be backward residual of original (v,u)
                    flow[v][u] -= path_flow # Decrease flow on original edge v->u

            f.write("\nUpdated Residual Graph:\n")
            # Use display_matrix or simple print
            for r in residual: f.write(" ".join(map(str, r)) + "\n")

            iteration += 1
            # Safety break for potential infinite loops if target_flow is huge or unreachable
            if iteration > n * n : # Heuristic limit
                 f.write("\nWarning: Exceeded iteration limit. Stopping.\n")
                 break

        f.write("\n--- Algorithm End ---\n")
        f.write(f"\nFinal Net Flow Matrix:\n")
        # Use display_matrix or simple print
        for r in flow: f.write(" ".join(map(str, r)) + "\n")
        f.write(f"\nAchieved Total Flow: {total_flow}\n")
        f.write(f"Minimum Total Cost: {total_cost}\n")

    return total_flow, total_cost, flow




# Function to generate random flow problem
def generate_random_flow_problem(n):
    capacity = np.zeros((n, n), dtype=int)
    cost = np.zeros((n, n), dtype=int)
    num_edges = int(n * n / 2)
    edges = random.sample([(i, j) for i in range(n) for j in range(n) if i != j], num_edges)
    
    for i, j in edges:
        capacity[i][j] = random.randint(1, 100)
        cost[i][j] = random.randint(1, 100)
    
    return capacity, cost

# Complexity analysis
def complexity_analysis():
    n_values = [10, 20, 40, 100, 400, 1000, 4000]
    ff_times = {n: [] for n in n_values}
    pr_times = {n: [] for n in n_values}
    min_times = {n: [] for n in n_values}
    
    for n in n_values:
        for _ in range(100):
            capacity, cost = generate_random_flow_problem(n)
            
            start = time.time()
            ford_fulkerson(n, capacity, "temp_ff.txt")
            ff_times[n].append(time.time() - start)
            
            start = time.time()
            push_relabel(n, capacity, "temp_pr.txt")
            pr_times[n].append(time.time() - start)
            
            max_flow, _ = ford_fulkerson(n, capacity, "temp_ff.txt")
            target_flow = max_flow // 2
            start = time.time()
            min_cost_flow(n, capacity, cost, target_flow, "temp_min.txt")
            min_times[n].append(time.time() - start)
    
    for algo, times in [("Ford-Fulkerson", ff_times), ("Push-Relabel", pr_times), ("Min-Cost", min_times)]:
        plt.figure()
        for n in n_values:
            plt.scatter([n] * 100, times[n], alpha=0.5)
        plt.xlabel('n (Number of vertices)')
        plt.ylabel('Execution time (seconds)')
        plt.title(f'{algo} Execution Times')
        plt.xscale('log')
        plt.yscale('log')
        plt.savefig(f'{algo.lower().replace(" ", "_")}_point_cloud.png')
        plt.close()
    
    for algo, times in [("Ford-Fulkerson", ff_times), ("Push-Relabel", pr_times), ("Min-Cost", min_times)]:
        max_times = [max(times[n]) for n in n_values]
        plt.plot(n_values, max_times, label=algo)
        
        if algo == "Ford-Fulkerson":
            expected = [n * (n ** 2) / 1e6 for n in n_values]
        elif algo == "Push-Relabel":
            expected = [(n ** 3) / 1e6 for n in n_values]
        else:
            expected = [(n ** 3 * np.log(n)) / 1e6 for n in n_values]
        plt.plot(n_values, expected, '--', label=f'{algo} expected')
    
    plt.xlabel('n')
    plt.ylabel('Max execution time (seconds)')
    plt.title('Worst-Case Complexity')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.savefig('worst_case_complexity.png')
    plt.close()
    
    ratios = [max(ff_times[n]) / max(pr_times[n]) for n in n_values]
    plt.plot(n_values, ratios, label='θ_FF / θ_PR')
    plt.xlabel('n')
    plt.ylabel('Ratio')
    plt.title('Ford-Fulkerson vs Push-Relabel')
    plt.xscale('log')
    plt.savefig('ff_vs_pr_ratio.png')
    plt.close()

# Main program
def main():
    while True:
        problem_num = input("Enter problem number (1-10, or 'exit' to quit): ")
        if problem_num.lower() == 'exit':
            break
        problem_num = int(problem_num)
        filename = f"proposal_{problem_num}.txt"
        
        n, capacity = read_flow_network(filename)
        vertices = [chr(ord('s') if i == 0 else ord('t') if i == n-1 else ord('a') + i - 1) for i in range(n)]
        
        print("Running Ford-Fulkerson...")
        max_flow_ff, flow_ff = ford_fulkerson(n, capacity, f"B4-trace{problem_num}-FF.txt")
        print(f"Ford-Fulkerson Max Flow: {max_flow_ff}")
        
        print("Running Push-Relabel...")
        max_flow_pr, flow_pr = push_relabel(n, capacity, f"B4-trace{problem_num}-PR.txt")
        print(f"Push-Relabel Max Flow: {max_flow_pr}")
        
        cost = np.random.randint(1, 10, size=(n, n)) * (capacity > 0)
        target_flow = max_flow_ff // 2
        print(f"Running Min-Cost Flow with target flow {target_flow}...")
        total_flow, total_cost, flow_min = min_cost_flow(n, capacity, cost, target_flow, f"B4-trace{problem_num}-MIN.txt")
        print(f"Min-Cost Flow: Total Flow = {total_flow}, Total Cost = {total_cost}")
        
        display_matrix(capacity, "Capacity Matrix", vertices)
        display_matrix(cost, "Cost Matrix", vertices)
        display_matrix(flow_ff, "Ford-Fulkerson Flow", vertices)
        display_matrix(flow_pr, "Push-Relabel Flow", vertices)
        display_matrix(flow_min, "Min-Cost Flow", vertices)
    
    print("Running complexity analysis...")
    complexity_analysis()

if __name__ == "__main__":
    main()
