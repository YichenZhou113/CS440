import numpy as np
import heapq
from copy import copy, deepcopy
import time
from tempfile import TemporaryFile


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
        self.__allColDomains = list.copy()

    def get_col_domains(self):
        return self.__col_domains

    def set_curr_col_domain(self, list, position):
        self.__col_domains[position] = list.copy()

    def get_curr_col_domain(self, position):
        return self.__col_domains[position]

    def set_row_domains(self, list):
        self.__allRowDomains = list.copy()

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

    def set_curr_allRowDomain(self, list, position):
        self.__allRowDomains[position] = list.copy()

    def get_allRowDomains(self):
        return self.__allRowDomains

    def append_allColDomains(self, list):
        self.__allColDomains.append(list)

    def clear_allColDomains(self):
        self.__allColDomains = []

    def get_curr_allColDomains(self, position):
        return self.__allColDomains[position]

    def set_curr_allColDomain(self, list, position):
        self.__allColDomains[position] = list.copy()

    def get_allColDomains(self):
        return self.__allColDomains

    def push_orderqueue(self, length, position, type):
        heapq.heappush(self.__variable_order_queue, (length, position, type))

    def pop_orderqueue(self):
        return heapq.heappop(self.__variable_order_queue)

    def get_orderqueue(self):
        return self.__variable_order_queue

    def clear_orderqueue(self):
        self.__variable_order_queue = []

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
    #print(len(solver.get_allColDomains()))
    if(ordered_var[2] == "row"):
        domain = solver.get_allRowDomains()[ordered_var[1]].copy()
    if(ordered_var[2] == "col"):
        domain = solver.get_allColDomains()[ordered_var[1]].copy()
    for i in domain:
        #print(check_consistency(i, ordered_var[2], ordered_var[1], rowSolution, colSolution, row_assigned, col_assigned, len(constraints[0]), len(constraints[1])))
        if(check_consistency(i, ordered_var[2], ordered_var[1], rowSolution, colSolution, row_assigned, col_assigned, len(constraints[0]), len(constraints[1])) == True):
            if(ordered_var[2] == "row"):
                old_col_domain = solver.get_allColDomains()
                rowSolution[ordered_var[1]] = i.copy()
                row_assigned[ordered_var[1]] = 1
                new_col_domain = forward_checking_row(ordered_var[1], i.copy(), solver.get_allColDomains())   # row_number, assigned value, old col domain
                if(new_col_domain != False):
                    #print("hello_row")
                    solver.set_col_domains(new_col_domain)
                    result = backtrack_dfs(solver, rowSolution, colSolution, constraints, row_assigned, col_assigned)
                    #print(result)
                    if(result != "failure"):
                        return result
            else:
                old_row_domain = solver.get_allRowDomains()
                colSolution[ordered_var[1]] = i.copy()
                col_assigned[ordered_var[1]] = 1
                new_row_domain = forward_checking_col(ordered_var[1], i.copy(), solver.get_allRowDomains())  # row_number, assigned value, old col domain
                if (new_row_domain != False):
                    solver.set_row_domains(new_row_domain)
                    result = backtrack_dfs(solver, rowSolution, colSolution, constraints, row_assigned, col_assigned)
                    #print(result)
                    if(result != "failure"):
                        return result

        if(ordered_var[2] == "row"):
            rowSolution[ordered_var[1]] = np.zeros(len(constraints[1]), dtype = int)
            row_assigned[ordered_var[1]] = 0
            #solver.set_curr_row_domain(domain, ordered_var[1])
            solver.set_col_domains(old_col_domain)
        else:
            colSolution[ordered_var[1]] = np.zeros(len(constraints[0]), dtype = int)
            col_assigned[ordered_var[1]] = 0
            #solver.set_curr_col_domain(domain, ordered_var[1])
            solver.set_row_domains(old_row_domain)
    solver.push_orderqueue(ordered_var[0], ordered_var[1], ordered_var[2])

    #print("failure")
    return "failure"


def forward_checking_row(row_number, row_assigned_value, col_domains):
    new_col_domain = []
    for i in range(len(col_domains)):       # each col
        new_col_domain.append([])
        for j in range(len(col_domains[i])):             # each potential solution of one col
            if(col_domains[i][j][row_number] == row_assigned_value[i]):
                new_col_domain[i].append(col_domains[i][j])
        if len(new_col_domain[i]) == 0:
            return False
    return new_col_domain


def forward_checking_col(col_number, col_assigned_value, row_domains):
    #print("col")
    new_row_domain = []
    for i in range(len(row_domains)):       # each col
        new_row_domain.append([])
        for j in range(len(row_domains[i])):             # each potential solution of one col
            if(row_domains[i][j][col_number] == col_assigned_value[i]):
                new_row_domain[i].append(row_domains[i][j])
        if len(new_row_domain[i]) == 0:
            return False
    return new_row_domain


def generate_comb(constraint, length, offset, old_sequence, index, last_pixel, solver, partial_solution, position, has_solution):
    if(offset < length):
        if(index < len(constraint)):
            current_constraint = constraint[index]

            if(offset + current_constraint - 1 < length and last_pixel == 0):
                for i in range(current_constraint):
                    old_sequence[offset + i] = 1


                generate_comb(constraint, length, offset + current_constraint, old_sequence, index + 1, 1, solver, partial_solution, position, has_solution)
                for i in range(current_constraint):
                    old_sequence[offset + i] = 0
        old_sequence[offset] = 0
        generate_comb(constraint, length, offset + 1, old_sequence, index, 0, solver, partial_solution, position, has_solution)
    elif(len(constraint) <= index):
        add = 1
        for each in range(len(old_sequence)):
            curr_col = partial_solution[each].copy()
            if(curr_col[position] == -2):
                continue
            elif(curr_col[position] != old_sequence[each]):
                add = 0
        if(add == 1):
            for i in range(len(has_solution)):
                has_solution[i] = has_solution[i] + old_sequence[i]
            solver.append_all_comb(old_sequence.copy())
    return


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
    start_time = time.time()
    solver = nonogram_solver(constraints)
    dim0 = len(constraints[0])
    dim1 = len(constraints[1])

    attempt_rowSolution = np.full([dim0, dim1], -2)
    attempt_colSolution = np.full([dim1, dim0], -2)



    for i in range(dim0):
        tidied_constraint = []
        for j in constraints[0][i]:
            tidied_constraint.append(j[0])
        #print(tidied_constraint)
        #if(i == 24):
            #print(tidied_constraint)
        constraint_sum = sum(tidied_constraint) + len(tidied_constraint) - 1
        comparator = dim1 - constraint_sum
        for clue in range(len(tidied_constraint)):
            if(tidied_constraint[clue] > comparator):
                num_block_to_fill = tidied_constraint[clue] - comparator
                if(clue == 0):
                    start_block = tidied_constraint[clue] - 1
                else:
                    start_block = -1
                    for k in range(clue + 1):
                        start_block += (tidied_constraint[k] + 1)
                    start_block -= 1
                #print("start_block")
                #print(clue)
                for k in range(num_block_to_fill):
                    block = start_block - k
                    #print(i, block)
                    attempt_rowSolution[i][block] = 1
                    attempt_colSolution[block][i] = 1


    for i in range(dim1):
        tidied_constraint = []
        for j in constraints[1][i]:
            tidied_constraint.append(j[0])
        #print(tidied_constraint)
        constraint_sum = sum(tidied_constraint) + len(tidied_constraint) - 1
        comparator = dim0 - constraint_sum
        for clue in range(len(tidied_constraint)):
            if(tidied_constraint[clue] > comparator):
                num_block_to_fill = tidied_constraint[clue] - comparator
                if(clue == 0):
                    start_block = tidied_constraint[clue] - 1
                else:
                    start_block = -1
                    for k in range(clue + 1):
                        start_block += (tidied_constraint[k] + 1)
                    start_block -= 1
                #print("start_block")
                #print(clue)
                for k in range(num_block_to_fill):
                    block = start_block - k
                    #print(i, block)
                    attempt_colSolution[i][block] = 1
                    attempt_rowSolution[block][i] = 1
                    #print(attempt_rowSolution[0][1])

    old_attempt_rowSolution = []
    #for i in range(3):
    counter = 0
    #while(np.array_equal(old_attempt_rowSolution, attempt_rowSolution) == False):
    for i in range(14):
        if(np.array_equal(old_attempt_rowSolution, attempt_rowSolution) == True):
            break
        counter = counter + 1
        old_row_domain = solver.get_allRowDomains().copy()
        old_col_domain = solver.get_allColDomains().copy()
        solver.clear_allRowDomains()
        solver.clear_allColDomains()
        solver.clear_orderqueue()
        old_attempt_rowSolution = attempt_rowSolution.copy()
        # Generate domains for each row of the nonogram
        for row in range(dim0):
            solver.clear_all_comb()
            tidied_constraint = []
            for i in constraints[0][row]:
                tidied_constraint.append(i[0])
            old_sequence = np.zeros(dim1, dtype = int)
            this_rowHasSolution = np.zeros(dim1, dtype = int)
            generate_comb(tidied_constraint, dim1, 0, old_sequence, 0, 0, solver, attempt_colSolution, row, this_rowHasSolution)
            for i in range(len(this_rowHasSolution)):
                if(this_rowHasSolution[i] == len(solver.get_all_comb())):
                    attempt_rowSolution[row][i] = 1
                elif(this_rowHasSolution[i] == 0):
                    attempt_rowSolution[row][i] = 0
            solver.append_allRowDomains(solver.get_all_comb().copy())
            solver.push_orderqueue(len(solver.get_all_comb()), row, "row")
        #print(preprocess_rowHasSolution)

        # Generate domains for each col of the nonogram
        for col in range(dim1):
            solver.clear_all_comb()
            tidied_constraint = []
            for i in constraints[1][col]:
                tidied_constraint.append(i[0])
            old_sequence = np.zeros(dim0, dtype = int)
            this_colHasSolution = np.zeros(dim0, dtype = int)
            generate_comb(tidied_constraint, dim0, 0, old_sequence, 0, 0, solver, attempt_rowSolution, col, this_colHasSolution)
            for i in range(len(this_colHasSolution)):
                if(this_colHasSolution[i] == len(solver.get_all_comb())):
                    attempt_colSolution[col][i] = 1
                elif(this_colHasSolution[i] == 0):
                    attempt_colSolution[col][i] = 0
            solver.append_allColDomains(solver.get_all_comb().copy())
            solver.push_orderqueue(len(solver.get_all_comb()), col, "col")

        #print(attempt_rowSolution)
        #print(attempt_colSolution)
        #print(solver.get_orderqueue())
        #print(attempt_rowSolution)
        for i in range(len(attempt_rowSolution)):
            for j in range(len(attempt_colSolution)):
                if(attempt_rowSolution[i][j] == -2 and attempt_colSolution[j][i] != -2):
                    attempt_rowSolution[i][j] = attempt_colSolution[j][i]
        #print(attempt_rowSolution)

        for i in range(len(attempt_colSolution)):
            for j in range(len(attempt_rowSolution)):
                if(attempt_colSolution[i][j] == -2 and attempt_rowSolution[j][i] != -2):
                    attempt_colSolution[i][j] = attempt_rowSolution[j][i]

    #    print("hello", counter)

    #print(preprocess_colHasSolution)
    #print("hellow")
    #print(solver.get_orderqueue())

    rowSolution = np.zeros([dim0, dim1], dtype = int)
    colSolution = np.zeros([dim1, dim0], dtype = int)
    row_assigned = np.zeros(dim0, dtype = int)
    col_assigned = np.zeros(dim1, dtype = int)

    #print(len(solver.get_allRowDomains()[0]))
    #arc_consistency(solver, dim0, dim1)
    #print(len(solver.get_allRowDomains()[0]))

    solver.set_row_domains(solver.get_allRowDomains())
    solver.set_col_domains(solver.get_allColDomains())

    #print(solver.get_curr_allColDomains(14))

    backtrack_dfs(solver, rowSolution, colSolution, constraints, row_assigned, col_assigned)

    print(solver.get_solution_matrix())
    #print(constraints[0][10])
    #print(solver.get_curr_allRowDomains(10))
    end_time = time.time()
    print('runtime is: ',str(end_time-start_time)+' sec')
    outfile = TemporaryFile()
    np.save('35_25_1_solution.npy', np.array(solver.get_solution_matrix()))
    print(np.load('35_25_1_solution.npy'))
    return np.array(solver.get_solution_matrix())
