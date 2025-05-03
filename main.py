from operation import read_flow_network, display_matrix
from ford_fulkerson import ford_fulkerson
from push_relabel import push_relabel
from min_cost import min_cost_flow

if __name__ == "__main__":
    while True:
        problem_num = input("Enter problem number (1-10, or 'exit' to quit): ")
        if problem_num.lower() == 'exit':
            break
        problem_num = int(problem_num)
        filename = f"proposal/proposal_{problem_num}.txt"

        n, capacity, cost = read_flow_network(filename)
        vertices = [chr(ord('s') if i == 0 else ord('t') if i == n - 1 else ord('a') + i - 1) for i in range(n)]

        # We display the capacity and cost matrices
        display_matrix(capacity, "Capacity Matrix", vertices)
        if cost is not None:
            display_matrix(cost, "Cost Matrix", vertices)

        # We execute the Ford-Fulkerson algorithm
        print("\nRunning Ford-Fulkerson...")
        residual_matrices_ff, flow_display_ff, max_flow_ff, flow_ff = ford_fulkerson(n, capacity, f"traces/Int1-8-trace{problem_num}-FF.txt")
        for i, residual_matrix in enumerate(residual_matrices_ff):
            display_matrix(residual_matrix, f"Residual Graph after iteration {i+1}", vertices)
        display_matrix(flow_display_ff, "Ford-Fulkerson Max Flow", vertices)
        print(f"Ford-Fulkerson Max Flow: {max_flow_ff}")

        """
        # We execute the Push-Relabel algorithm
        print("Running Push-Relabel...")
        max_flow_pr, flow_pr, flow_display_pr = push_relabel(n, capacity, f"traces/Int1-8-trace{problem_num}-PR.txt")
        display_matrix(flow_display_pr, "Push-Relabel Max Flow", vertices)
        print(f"Push-Relabel Max Flow: {max_flow_pr}")
        """
        if cost is not None:
            target_flow = max_flow_ff // 2
            print(f"\nRunning Min-Cost Flow with target flow {target_flow}")
            total_flow, total_cost, flow_min = min_cost_flow(n, capacity, cost,0, n-1, target_flow, f"traces/Int1-8-trace{problem_num}-MIN.txt")
            display_matrix(flow_min, "Min-Cost Flow", vertices)
            print(f"Min-Cost Flow: Total Flow = {total_flow}, Total Cost = {total_cost}")
        
