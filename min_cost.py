import numpy as np

def min_cost_flow(n, capacity, cost, source, sink, target_flow, trace_file):
    """Minimum-Cost Flow using Ford-Fulkerson with Bellman-Ford."""
    flow = np.zeros((n, n), dtype=int)
    total_flow = 0
    total_cost = 0
    
    with open(trace_file, 'w', encoding='utf-8') as f:
        f.write("Minimum-Cost Flow Execution Trace\n")
        f.write("\nCapacity table display:\n")
        for row in capacity:
            f.write(" ".join(f"{x:>2}" for x in row) + "\n")
        f.write("\nCost table display:\n")
        for row in cost:
            f.write(" ".join(f"{x:>2}" for x in row) + "\n")
        f.write("\nThe initial residual graph is the starting graph.\n")
        
        iteration = 1
        while total_flow < target_flow:
            flow_value, parent, bf_trace = bellman_ford(n, capacity, flow, cost, source, sink)
            if flow_value == 0:
                break
            flow_value = min(flow_value, target_flow - total_flow)
            f.write(f"\n* Iteration {iteration}:\n")
            f.write(bf_trace + "\n")
            f.write("\nModifications to the residual graph:\n")
            
            # Update flow
            v = sink
            path_cost = 0
            while v != source:
                u = parent[v]
                flow[u,v] += flow_value
                flow[v,u] -= flow_value
                path_cost += (cost[u,v] if capacity[u,v] > 0 else -cost[v,u]) * flow_value
                v = u
            total_flow += flow_value
            total_cost += path_cost
            
            # Display residual graph
            residual = np.zeros((n, n), dtype=int)
            for i in range(n):
                for j in range(n):
                    if capacity[i,j] > 0:
                        residual[i,j] = capacity[i,j] - flow[i,j]
                    elif flow[j,i] > 0:
                        residual[i,j] = flow[j,i]
            
            for row in residual:
                f.write(" ".join(f"{x:>2}" for x in row) + "\n")
            f.write(f"\nCost of this path: {path_cost}\n")
            iteration += 1
        
        f.write("\n* Min-cost flow display:\n")
        flow_display = np.zeros((n, n), dtype=object)
        for i in range(n):
            for j in range(n):
                flow_display[i][j] = f"{flow[i][j]}/{capacity[i][j]}" if capacity[i][j] > 0 else "0"
        for row in flow_display:
            f.write(" ".join(f"{x:>5}" for x in row) + "\n")
        f.write(f"\nTotal flow value = {total_flow}\n")
        f.write(f"Total cost = {total_cost}\n")
    
    return total_flow, total_cost, flow_display


def bellman_ford(n, capacity, flow, cost, source, sink):
    """Bellman-Ford for shortest path in weighted residual graph."""
    dist = [float('inf')] * n
    dist[source] = 0
    parent = [-1] * n
    trace = ["| k | " + " | ".join(chr(97+i) for i in range(n))]
    
    for k in range(n):
        prev_dist = dist.copy()
        for u in range(n):
            for v in range(n):
                residual = capacity[u,v] - flow[u,v] if capacity[u,v] > 0 else flow[v,u]
                if residual > 0:
                    d = cost[u,v] if capacity[u,v] > 0 else -cost[v,u]
                    if dist[v] > dist[u] + d:
                        dist[v] = dist[u] + d
                        parent[v] = u
        # Format table
        row = [f"{k:2d} |"]
        for i in range(n):
            if dist[i] == float('inf'):
                row.append(" ∞ ")
            else:
                row.append(f"{dist[i]:2d}{chr(97+parent[i]) if parent[i] >= 0 else ''}")
        trace.append(" | ".join(row))
        if dist == prev_dist:
            break
    
    if dist[sink] == float('inf'):
        return 0, parent, "\n".join(trace)
    
    # Find path and flow
    path = []
    curr = sink
    min_flow = float('inf')
    while curr != source:
        u = parent[curr]
        residual = capacity[u,curr] - flow[u,curr] if capacity[u,curr] > 0 else flow[curr,u]
        min_flow = min(min_flow, residual)
        path.append(chr(97+curr))
        curr = u
    path.append('s')
    path_str = ''.join(reversed(path))
    trace.append(f"Detection of an improving chain: {path_str} with a flow {min_flow}.")
    return min_flow, parent, "\n".join(trace)