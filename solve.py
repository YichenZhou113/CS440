import numpy as np
import heapq
from copy import copy, deepcopy

all_comb = []

class nonogram_solver:
    def __init__ (self, constraints):

        self.__constraints = constraints
        self.__all_comb = []
        self.__allRowDomains = []
        self.__allColDomains = []
        self.__row_domains = []
        self.__col_domains = []
        self.__variable_order_queue = []
        self.__solution_matrix = []

    def set_col_domains(self, list):
        self.__col_domains = list.copy()

    def get_col_domains(self):
        return self.__col_domains

    def set_curr_col_domain(self, list, position):
        self.__col_domains[position] = list.copy()

    def get_curr_col_domain(self, position):
        return self.__col_domains[position]

    def set_row_domains(self, list):
        self.__row_domains = list.copy()

    def get_row_domains(self):
        return self.__row_domains

    def set_curr_row_domain(self, list, position):
        self.__row_domains[position] = list.copy()

    def get_curr_row_domain(self, position):
        return self.__row_domains[position]

    def append_all_comb(self, list):
        self.__all_comb.append(list)

    def get_all_comb(self):
        return self.__all_comb

    def clear_all_comb(self):
        self.__all_comb = []

    def append_allRowDomains(self, list):
        self.__allRowDomains.append(list)

    def clear_allRowDomains(self):
        self.__allRowDomains = []

    def get_curr_allRowDomains(self, position):
        return self.__allRowDomains[position]

    def get_allRowDomains(self):
        return self.__allRowDomains

    def append_allColDomains(self, list):
        self.__allColDomains.append(list)

    def clear_allColDomains(self):
        self.__allColDomains = []

    def get_curr_allColDomains(self, position):
        return self.__allColDomains[position]

    def get_allColDomains(self):
        return self.__allColDomains

    def push_orderqueue(self, length, position, type):
        heapq.heappush(self.__variable_order_queue, (length, position, type))

    def pop_orderqueue(self):
        return heapq.heappop(self.__variable_order_queue)

    def get_orderqueue(self):
        return self.__variable_order_queue

    def set_solution_matrix(self, list):
        self.__solution_matrix = list.copy()

    def get_solution_matrix(self):
        return self.__solution_matrix


def check_consistency(var, type, position, rowSolution, colSolution, row_assigned, col_assigned, row_dim, col_dim):
    consistent = True
    if(type == "row"):
        for i in range(col_dim):
            if(col_assigned[i] == 0):
                continue
            else:
                if(colSolution[i][position] != var[i]):
                    consistent = False
                    break

    else:
        for i in range(row_dim):
            if(row_assigned[i] == 0):
                continue
            else:
                if(rowSolution[i][position] != var[i]):
                    consistent = False
                    break

    return consistent


def backtrack_dfs(solver, rowSolution, colSolution, constraints, row_assigned, col_assigned):
    if(len(solver.get_orderqueue()) == 0):
        solver.set_solution_matrix(rowSolution)
        #print("success")
        return "success"
    ordered_var = solver.pop_orderqueue()
    if(ordered_var[2] == "row"):
        domain = solver.get_row_domains()[ordered_var[1]].copy()
    if(ordered_var[2] == "col"):
        domain = solver.get_col_domains()[ordered_var[1]].copy()
    for i in domain:
        #print(check_consistency(i, ordered_var[2], ordered_var[1], rowSolution, colSolution, row_assigned, col_assigned, len(constraints[0]), len(constraints[1])))
        if(check_consistency(i, ordered_var[2], ordered_var[1], rowSolution, colSolution, row_assigned, col_assigned, len(constraints[0]), len(constraints[1])) == True):
            if(ordered_var[2] == "row"):
                rowSolution[ordered_var[1]] = i.copy()
                row_assigned[ordered_var[1]] = 1
            else:
                colSolution[ordered_var[1]] = i.copy()
                col_assigned[ordered_var[1]] = 1

            result = backtrack_dfs(solver, rowSolution, colSolution, constraints, row_assigned, col_assigned)
            #print(result)
            if(result != "failure"):
                return result
        if(ordered_var[2] == "row"):
            rowSolution[ordered_var[1]] = np.zeros(len(constraints[1]), dtype = int)
            row_assigned[ordered_var[1]] = 0
            #solver.set_curr_row_domain(domain, ordered_var[1])
        else:
            colSolution[ordered_var[1]] = np.zeros(len(constraints[0]), dtype = int)
            col_assigned[ordered_var[1]] = 0
            #solver.set_curr_col_domain(domain, ordered_var[1])
    solver.push_orderqueue(ordered_var[0], ordered_var[1], ordered_var[2])

    #print("failure")
    return "failure"


def generate_comb(constraint, length, offset, old_sequence, index, last_pixel, solver):
    if(offset < length):
        if(index < len(constraint)):
            current_constraint = constraint[index]

            if(offset + current_constraint - 1 < length and last_pixel == 0):
                for i in range(current_constraint):
                    old_sequence[offset + i] = 1
                generate_comb(constraint, length, offset + current_constraint, old_sequence, index + 1, 1, solver)
                for i in range(current_constraint):
                    old_sequence[offset + i] = 0
        old_sequence[offset] = 0
        generate_comb(constraint, length, offset + 1, old_sequence, index, 0, solver)
    elif(len(constraint) <= index):
        solver.append_all_comb(old_sequence.copy())
    return


def forward_checking(solver, type, index, csp):
    # cur_row is ex: [1 0 1 1 0]
    if type == 'row':
        cur_row_set = solver.get_curr_row_domain(index)
        if !cur_row_set:
            return

        next_var = solver.pop_orderqueue()[]
        for cur_row in cur_row_set:
            for cur_var in next_var:
                if check_consistency(cur_row, type, position, rowSolution, colSolution, row_assigned, col_assigned, row_dim, col_dim):
                    break
            next_row_final_set.append(row)
        forward_checking(next_row_final_set, cur_row_index+1, csp)
        return next_row_final_set


def solve(constraints):
    """
    Implement me!!!!!!!
    This function takes in a set of constraints. The first dimension is the axis
    to which the constraints refer to. The second dimension is the list of constraints
    for some axis index pair. The third demsion is a single constraint of the form
    [i,j] which means a run of i js. For example, [4,1] would correspond to a block
    [1,1,1,1].

    The return value of this function should be a numpy array that satisfies all
    of the constraints.


	A puzzle will have the constraints of the following format:


	array([
		[list([[4, 1]]),
		 list([[1, 1], [1, 1], [1, 1]]),
         list([[3, 1], [1, 1]]),
		 list([[2, 1]]),
		 list([[1, 1], [1, 1]])],
        [list([[2, 1]]),
		 list([[1, 1], [1, 1]]),
		 list([[3, 1], [1, 1]]),
         list([[1, 1], [1, 1]]),
		 list([[5, 1]])]
		], dtype=object)

	And a corresponding solution may be:

	array([[0, 1, 1, 1, 1],
		   [1, 0, 1, 0, 1],
		   [1, 1, 1, 0, 1],
		   [0, 0, 0, 1, 1],
		   [0, 0, 1, 0, 1]])



	Consider a more complicated set of constraints for a colored nonogram.

	array([
	   [list([[1, 1], [1, 4], [1, 2], [1, 1], [1, 2], [1, 1]]),
        list([[1, 3], [1, 4], [1, 3]]),
		list([[1, 2]]),
        list([[1, 4], [1, 1]]),
		list([[2, 2], [2, 1], [1, 3]]),
        list([[1, 2], [1, 3], [1, 2]]),
		list([[2, 1]])],
       [list([[1, 3], [1, 4], [1, 2]]),
        list([[1, 1], [1, 4], [1, 2], [1, 2], [1, 1]]),
        list([[1, 4], [1, 1], [1, 2], [1, 1]]),
		list([[1, 2], [1, 1]]),
        list([[1, 1], [2, 3]]),
		list([[1, 2], [1, 3]]),
        list([[1, 1], [1, 1], [1, 2]])]],
		dtype=object)

	And a corresponding solution may be:

	array([
		   [0, 1, 4, 2, 1, 2, 1],
		   [3, 4, 0, 0, 0, 3, 0],
		   [0, 2, 0, 0, 0, 0, 0],
		   [4, 0, 0, 0, 0, 0, 1],
		   [2, 2, 1, 1, 3, 0, 0],
		   [0, 0, 2, 0, 3, 0, 2],
		   [0, 1, 1, 0, 0, 0, 0]
		 ])


    """
    solver = nonogram_solver(constraints)
    dim0 = len(constraints[0])
    dim1 = len(constraints[1])

    # Generate domains for each row of the nonogram
    for row in range(dim0):
        solver.clear_all_comb()
        tidied_constraint = []
        for i in constraints[0][row]:
            tidied_constraint.append(i[0])
        old_sequence = np.zeros(dim1, dtype = int)
        generate_comb(tidied_constraint, dim1, 0, old_sequence, 0, 0, solver)
        solver.append_allRowDomains(solver.get_all_comb().copy())
        solver.push_orderqueue(len(solver.get_all_comb()), row, "row")

    # Generate domains for each col of the nonogram
    for col in range(dim1):
        solver.clear_all_comb()
        tidied_constraint = []
        for i in constraints[1][col]:
            tidied_constraint.append(i[0])
        old_sequence = np.zeros(dim0, dtype = int)
        generate_comb(tidied_constraint, dim0, 0, old_sequence, 0, 0, solver)
        solver.append_allColDomains(solver.get_all_comb().copy())
        solver.push_orderqueue(len(solver.get_all_comb()), col, "col")

    rowSolution = np.zeros([dim0, dim1], dtype = int)
    colSolution = np.zeros([dim1, dim0], dtype = int)
    row_assigned = np.zeros(dim0, dtype = int)
    col_assigned = np.zeros(dim1, dtype = int)
    #print(solver.get_curr_allRowDomains(2))

    solver.set_row_domains(solver.get_allRowDomains())
    solver.set_col_domains(solver.get_allColDomains())

    backtrack_dfs(solver, rowSolution, colSolution, constraints, row_assigned, col_assigned)

    print(solver.get_solution_matrix())
    #print(constraints[1])


    #solution_matrix = np.zeros([dim0, dim1], dtype = int)
    #status_matrix = np.zeros(dim0, dtype = int)
    #result = backtrack_dfs({}, status_matrix, constraints)

    return np.array(solver.get_solution_matrix())
