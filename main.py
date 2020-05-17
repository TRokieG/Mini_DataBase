import numpy as np
import itertools
import time
import os
from BTrees.OOBTree import OOBTree
import sys



class Database:

    def __init__(self, operations):
        '''
        :param operations: str - input a file containing all operations line by line
        '''

        # read operations from file
        self.operations = []
        with open(operations, 'r') as file:
            line = file.readline()
            while line:
                if '/' in line:
                    self.operations.append(line[:line.index('/')])
                else:
                    self.operations.append(line)
                line = file.readline()

        '''
        tables to store all input tables
        format is for {table1:{column_name1:[...], column_name2:[...] ...}
                       table2:{column_name1:[...], column_name2:[...] ...}
                       ....... }
        '''

        self.tables = {}

        # create index variable in case to use index

        self.Btree_index = {}
        self.Hash_index = {}

        # start executing operations
        for op in self.operations:
            # new table name
            try:
                new_tn = op.split(":=")[0].strip()

                # find out the index of first char for function name
                rest = op.split(":=")[1].strip()
                op_name = rest[:rest.index('(')].strip()

                parameter_str = rest.strip()[rest.index('(') + 1: -1]
            except:
                # ONLY FOR Btree Hash and output files
                new_tn = ''
                rest = op.strip()
                op_name = rest[:rest.index('(')].strip()

                parameter_str = rest.strip()[rest.index('(') + 1: -1]



            if op_name == 'inputfromfile':
                self.inputfromfile(parameter_str, new_tn)

            elif op_name == 'project':
                self.project(parameter_str, new_tn)

            elif op_name == 'select':
                self.select(parameter_str, new_tn)
            elif op_name == 'avg':
                self.avg(parameter_str, new_tn)
            elif op_name == 'sumgroup':
                self.sumgroup(parameter_str, new_tn)
            elif op_name == 'avggroup':
                self.avggroup(parameter_str, new_tn)
            elif op_name == 'join':
                self.join(parameter_str, new_tn)
            elif op_name == 'sort':
                self.sort(parameter_str, new_tn)
            elif op_name == 'movavg':
                self.movavg(parameter_str, new_tn)
            elif op_name == 'movsum':
                self.movsum(parameter_str, new_tn)
            elif op_name == 'concat':
                self.concat(parameter_str, new_tn)
            elif op_name == 'outputtofile':
                self.outputtofile(parameter_str, new_tn)
            elif op_name == 'count':
                self.count(parameter_str, new_tn)
            elif op_name == 'Btree':
                self.Btree(parameter_str, new_tn)
            elif op_name == 'Hash':
                self.Hash(parameter_str, new_tn)
            else:
                raise ValueError('no such operation')


    def inputfromfile(self,operation, tn):
        st = time.time()
        # READ INPUT FILE AND STORE DATA INTO TABLE
        file_name = operation.strip()
        with open(f"{file_name}.txt",'r') as file:
            # read column names
            line = file.readline()
            col_n = line.split('|')
            # create table in self.table
            self.tables.update({tn: {}})
            for name in col_n:
                self.tables[tn].update({name.strip(): []})

            # read data into table
            line = file.readline()
            while line:
                data = line.split('|')
                keys = list(self.tables[tn].keys())
                for i, value in enumerate(data):
                    self.tables[tn][keys[i].strip()] += [value.strip()]
                line = file.readline()
        print('inputfle:',time.time() -st)



    def project(self, operation, tn):
        st = time.time()
        all_names = operation.split(",")
        table_name = all_names[0].strip()
        all_names.remove(table_name)
        result = {tn:{}}
        for col_n in all_names:
            col_n = col_n.strip()
            result[tn].update({col_n: self.tables[table_name][col_n]})

        self.tables.update(result)
        print('project', time.time() - st)

    def select(self, operation, tn):
        st = time.time()
        # split operation into table and conditions
        table_n = operation.split(',')[0].strip()
        conditions = operation.split(',')[1].strip()
        status = False
        if 'and' in conditions:
            op_steps = conditions.split('and')
            status = 'and'
        elif 'or' in conditions:
            op_steps = conditions.split('or')
            status = 'or'
        else:
            op_steps = [conditions]

        # start filter out the data we need
        all_idx = {}

        # check if index exists
        if table_n in list(self.Hash_index.keys()):
            if len(op_steps) == 1:
                coll_n = op_steps[0].split()[0].strip()
                if coll_n in list(self.Hash_index[table_n].keys()):
                    # Use hash index
                    self.select_with_index( 'hash',status, table_n, op_steps, tn)
                    return
            else:
                col_ns = []
                for op in op_steps:
                    col_ns += [op.split()[0].strip()]

                for col in col_ns:
                    if col in list(self.Hash_index[table_n].keys()):
                        self.select_with_index('hash', status, table_n, op_steps, tn)
                        return
        elif table_n in list(self.Btree_index.keys()):
            self.select_with_index('btree', status, table_n, op_steps, tn)
            return



        if status:
            for op in op_steps:
                # get rid of bracket (there is no bracket is there is only single solution, so if condition used)
                op = op.strip()[1:-1]
                # get column name, operator and value used to compare
                split_op = op.split()
                col_n = split_op[0]
                operator = split_op[1]
                compare_v = float(split_op[-1])
                # create dictionary to record all indexes satisfying conditions
                all_idx.update({col_n:[]})
                for i, value in enumerate(self.tables[table_n][col_n]):
                    if operator == '=':
                        if float(value) == compare_v:
                            all_idx[col_n] += [i]

                    elif operator == '>':
                        if float(value) > compare_v:
                            all_idx[col_n] += [i]

                    elif operator == '<':
                        if float(value) < compare_v:
                            all_idx[col_n] += [i]

                    elif operator == '<=':
                        if float(value) <+ compare_v:
                            all_idx[col_n] += [i]

                    elif operator == ">=":
                        if float(value) >= compare_v:
                            all_idx[col_n] += [i]

                    elif operator == "!=":
                        if float(value) != compare_v:
                            all_idx[col_n] += [i]

        else:

            for op in op_steps:
                split_op = op.split()
                col_n = split_op[0]
                operator = split_op[1]
                compare_v = float(split_op[-1])
                # create dictionary to record all indexes satisfying conditions
                all_idx.update({col_n: []})

                for i, value in enumerate(self.tables[table_n][col_n]):
                    if operator == '=':
                        if float(value) == compare_v:
                            all_idx[col_n] += [i]

                    elif operator == '>':
                        if float(value) > compare_v:
                            all_idx[col_n] += [i]

                    elif operator == '<':
                        if float(value) < compare_v:
                            all_idx[col_n] += [i]

                    elif operator == '<=':
                        if float(value) < + compare_v:
                            all_idx[col_n] += [i]

                    elif operator == ">=":
                        if float(value) >= compare_v:
                            all_idx[col_n] += [i]

                    elif operator == "!=":
                        if float(value) != compare_v:
                            all_idx[col_n] += [i]

        final_idx = set(all_idx[list(all_idx.keys())[0]])

        if status == 'or':
            for key in list(all_idx.keys()):
                final_idx = final_idx.union(set(all_idx[key]))
        elif status == 'and':
            for key in list(all_idx.keys()):
                final_idx = final_idx.intersection(set(all_idx[key]))

        final_idx = list(final_idx)


        output = {tn:{}}
        for col_n in list(self.tables[table_n].keys()):
            output[tn].update({col_n:[]})


        for idx in final_idx:
            for col_n in list(self.tables[table_n].keys()):
                output[tn][col_n] += [self.tables[table_n][col_n][idx]]

        self.tables.update(output)
        print('select:', time.time() - st)

    def select_with_index(self, index_type, status,  table_n, op_steps, tn):
        print('Using index to conduct select')
        st = time.time()
        all_idx = {}

        if status:
            for op in op_steps:
                # get rid of bracket (there is no bracket is there is only single solution, so if condition used)
                op = op.strip()[1:-1]
                # get column name, operator and value used to compare
                split_op = op.split()
                col_n = split_op[0]
                operator = split_op[1]
                compare_v = float(split_op[-1])
                # create dictionary to record all indexes satisfying conditions
                all_idx.update({col_n: []})
                for i, value in enumerate(self.tables[table_n][col_n]):
                    if operator == '=':
                        if float(value) == compare_v:
                            all_idx[col_n] += [i]

                    elif operator == '>':
                        if float(value) > compare_v:
                            all_idx[col_n] += [i]

                    elif operator == '<':
                        if float(value) < compare_v:
                            all_idx[col_n] += [i]

                    elif operator == '<=':
                        if float(value) < + compare_v:
                            all_idx[col_n] += [i]

                    elif operator == ">=":
                        if float(value) >= compare_v:
                            all_idx[col_n] += [i]

                    elif operator == "!=":
                        if float(value) != compare_v:
                            all_idx[col_n] += [i]

        else:

            for op in op_steps:
                split_op = op.split()
                col_n = split_op[0]
                operator = split_op[1]
                compare_v = float(split_op[-1])
                # create dictionary to record all indexes satisfying conditions
                all_idx.update({col_n: []})

                for i, value in enumerate(self.tables[table_n][col_n]):
                    if operator == '=':
                        if float(value) == compare_v:
                            all_idx[col_n] += [i]

                    elif operator == '>':
                        if float(value) > compare_v:
                            all_idx[col_n] += [i]

                    elif operator == '<':
                        if float(value) < compare_v:
                            all_idx[col_n] += [i]

                    elif operator == '<=':
                        if float(value) < + compare_v:
                            all_idx[col_n] += [i]

                    elif operator == ">=":
                        if float(value) >= compare_v:
                            all_idx[col_n] += [i]

                    elif operator == "!=":
                        if float(value) != compare_v:
                            all_idx[col_n] += [i]

        final_idx = set(all_idx[list(all_idx.keys())[0]])

        if status == 'or':
            for key in list(all_idx.keys()):
                final_idx = final_idx.union(set(all_idx[key]))
        elif status == 'and':
            for key in list(all_idx.keys()):
                final_idx = final_idx.intersection(set(all_idx[key]))

        final_idx = list(final_idx)

        output = {tn: {}}
        for col_n in list(self.tables[table_n].keys()):
            output[tn].update({col_n: []})

        for idx in final_idx:
            for col_n in list(self.tables[table_n].keys()):
                output[tn][col_n] += [self.tables[table_n][col_n][idx]]

        self.tables.update(output)
        print('select with index:', time.time() - st)

    def avg(self, operation, tn):
        st = time.time()
        table_n = operation.split(',')[0].strip()
        col_n = operation.split(',')[1].strip()
        total = 0
        for num in self.tables[table_n][col_n]:
            total += float(num)

        avg = total / len(self.tables[table_n][col_n])
        self.tables.update({tn:{'avg({})'.format(col_n):avg}})
        print('avg:', time.time() - st)



    def sumgroup(self, operation, tn):
        st = time.time()
        all_param = operation.split(',')
        table_n = all_param[0].strip()
        # Determine groupby column and sum column
        group_by = [all_param[i].strip() for i in range(len(all_param)) if i > 1]
        sum_group = all_param[1].strip()
        # DETERMINE DISTINCT GROUP
        distct_list = []
        for col_n in group_by:
            distct_list += [list(set(self.tables[table_n][col_n]))]
        # CALCULATE ALL COMBINATIONS
        all_comb = list(itertools.product(*distct_list))
        output = {tn:{}}
        for comb in all_comb:
            output[tn].update({comb:0})
        num_group = len(group_by)
        # ARRANGE DATA AND RETURN IT
        for idx, value in enumerate(self.tables[table_n][sum_group]):
            condition = []
            for col_n in group_by:
                condition += [self.tables[table_n][col_n][idx]]
            condition = tuple(condition)
            output[tn][condition] += float(value)
        for key in list(output[tn].keys()):
            if output[tn][key] == 0:
                output[tn].pop(key)

        self.tables.update(output)
        print('sumgroup', time.time() - st)


    def avggroup(self, operation, tn):

        st = time.time()

        all_param = operation.split(',')
        table_n = all_param[0].strip()
        # Determine groupby column and sum column
        group_by = [all_param[i].strip() for i in range(len(all_param)) if i > 1]
        sum_group = all_param[1].strip()
        # DETERMINE DISTINCT GROUP
        distct_list = []
        for col_n in group_by:
            distct_list += [list(set(self.tables[table_n][col_n]))]
        # CALCULATE ALL COMBINATIONS
        all_comb = list(itertools.product(*distct_list))
        output = {tn: {}}
        for comb in all_comb:
            output[tn].update({comb: []})
        num_group = len(group_by)
        # ARRANGE DATA AND RETURN IT
        for idx, value in enumerate(self.tables[table_n][sum_group]):
            condition = []
            for col_n in group_by:
                condition += [self.tables[table_n][col_n][idx]]
            condition = tuple(condition)
            output[tn][condition] += [float(value)]
        for key in list(output[tn].keys()):
            if output[tn][key] == 0:
                output[tn].pop(key)

        for key in list(output[tn].keys()):
            output[tn][key] = sum(output[tn][key]) / len(output[tn][key])


        self.tables.update(output)

        print('avg_group:', time.time() - st)

    def join(self, operation, tn):
        st = time.time()
        start = time.time()

        # split parameters
        all_params = operation.split(',')
        table1 = all_params[0].strip()
        table2 = all_params[1].strip()

        conditions = all_params[-1]
        status = False
        if 'and' in conditions:
            op_steps = conditions.split('and')
            status = 'and'
            for i, value in enumerate(op_steps):
                op_steps[i] = value[value.index('(') + 1: value.index(')')]
        else:
            op_steps = [conditions]

        all_idx = {table1:[], table2:[]}
        # ALWAYS MAKE '=' OPERATOR AS FIRST TO SPEED UP
        if len(op_steps) > 1:
            for i, op in enumerate(op_steps):
                if '=' in op:
                    if i == 0:
                        pass
                    else:
                        op_steps[i], op_steps[0] = op_steps[0], op_steps[i]
                    break

        # RUN N^2 ALGO TO FIND ALL MATCHING INDEXES OF TABLE 1 AND TABLE2

        for times, op in enumerate(op_steps):
            params = op.split()
            table_n = [params[0].split('.')[0], params[2].split('.')[0]]
            col_n = [params[0].split('.')[1], params[2].split('.')[1]]
            operator = params[1]
            # RECORD INDEXES TO DELETE
            to_delete = []

            if operator == '=':
                if times == 0:
                    for i, value1 in enumerate(self.tables[table_n[0]][col_n[0]]):
                        for j, value2 in enumerate(self.tables[table_n[1]][col_n[1]]):
                            if value1 == value2:
                                all_idx[table_n[0]] += [i]
                                all_idx[table_n[1]] += [j]

                else:
                    for i in range(len(all_idx[table1])):
                        if not self.tables[table_n[0]][col_n[0]][all_idx[table_n[0]][i]] == self.tables[table_n[1]][col_n[1]][all_idx[table_n[1]][i]]:
                            if i not in to_delete:
                                to_delete += [i]

            elif operator == '>':
                if times == 0:
                    for i, value1 in enumerate(self.tables[table_n[0]][col_n[0]]):
                        for j, value2 in enumerate(self.tables[table_n[1]][col_n[1]]):

                            if value1 > value2:
                                all_idx[table_n[0]] += [i]
                                all_idx[table_n[1]] += [j]

                else:
                    for i in range(len(all_idx[table1])):
                        if not self.tables[table_n[0]][col_n[0]][all_idx[table_n[0]][i]] > self.tables[table_n[1]][col_n[1]][all_idx[table_n[1]][i]]:
                            if i not in to_delete:
                                to_delete += [i]
            elif operator == '<':
                if times == 0:
                    for i, value1 in enumerate(self.tables[table_n[0]][col_n[0]]):
                        for j, value2 in enumerate(self.tables[table_n[1]][col_n[1]]):
                            if value1 < value2:
                                all_idx[table_n[0]] += [i]
                                all_idx[table_n[1]] += [j]

                else:
                    for i in range(len(all_idx[table1])):
                        if not self.tables[table_n[0]][col_n[0]][all_idx[table_n[0]][i]] < self.tables[table_n[1]][col_n[1]][all_idx[table_n[1]][i]]:
                            if i not in to_delete:
                                to_delete += [i]

            elif operator == '<=':
                if times == 0:
                    for i, value1 in enumerate(self.tables[table_n[0]][col_n[0]]):
                        for j, value2 in enumerate(self.tables[table_n[1]][col_n[1]]):
                            if value1 <= value2:
                                all_idx[table_n[0]] += [i]
                                all_idx[table_n[1]] += [j]

                else:
                    for i in range(len(all_idx[table1])):
                        if not self.tables[table_n[0]][col_n[0]][all_idx[table_n[0]][i]] <= self.tables[table_n[1]][col_n[1]][all_idx[table_n[1]][i]]:
                            if i not in to_delete:
                                to_delete += [i]

            elif operator == ">=":
                if times == 0:
                    for i, value1 in enumerate(self.tables[table_n[0]][col_n[0]]):
                        for j, value2 in enumerate(self.tables[table_n[1]][col_n[1]]):
                            if value1 >= value2:
                                all_idx[table_n[0]] += [i]
                                all_idx[table_n[1]] += [j]

                else:
                    for i in range(len(all_idx[table1])):
                        if not self.tables[table_n[0]][col_n[0]][all_idx[table_n[0]][i]] >= self.tables[table_n[1]][col_n[1]][all_idx[table_n[1]][i]]:
                            if i not in to_delete:
                                to_delete += [i]

            elif operator == "!=":
                if times == 0:
                    for i, value1 in enumerate(self.tables[table_n[0]][col_n[0]]):
                        for j, value2 in enumerate(self.tables[table_n[1]][col_n[1]]):
                            if value1 != value2:
                                all_idx[table_n[0]] += [i]
                                all_idx[table_n[1]] += [j]

                else:
                    for i in range(len(all_idx[table1])):
                        if not self.tables[table_n[0]][col_n[0]][all_idx[table_n[0]][i]] != self.tables[table_n[1]][col_n[1]][all_idx[table_n[1]][i]]:
                            if i not in to_delete:
                                to_delete += [i]


        all_idx_new = {table1:[], table2:[]}
        for i, value1 in enumerate(all_idx[table1]):
            if i not in to_delete:
                all_idx_new[table1] += [value1]

        for i, value2 in enumerate(all_idx[table2]):
            if i not in to_delete:
                all_idx_new[table2] += [value2]




        output = {tn:{}}
        for key in list(self.tables[table1].keys()):
            output[tn].update({table1+'_'+key:[]})
        for key in list(self.tables[table2].keys()):
            output[tn].update({table2+'_'+key:[]})

        # TO DO INPUT VALUES INTO OUTPUT
        for idx in all_idx_new[table1]:
            for key in list(self.tables[table1].keys()):
                output[tn][table1+'_'+key] += [self.tables[table1][key][idx]]

        for idx in all_idx_new[table2]:
            for key in list(self.tables[table2].keys()):
                output[tn][table2+'_'+key] += [self.tables[table2][key][idx]]


        self.tables.update(output)

        print('join:', time.time() - st)


    def sort(self, operation, tn):

        st = time.time()
        # this function sort the table and return the updated table
        all_params = operation.split(',')
        table_n = all_params[0].strip()
        column_names = all_params[1:]
        all_list = []
        for col in column_names:
            col = col.strip()
            all_list += [[self.tables[table_n][col]]]

        idx = np.lexsort(all_list)
        output = {tn:{}}
        for key in list(self.tables[table_n].keys()):
            new_list = np.array(self.tables[table_n][key])
            output[tn][key] = [new_list[i] for i in idx][0].tolist()
        self.tables.update(output)
        print('sort:', time.time() - st)


    def movavg(self, operation, tn):
        st = time.time()
        all_params = operation.split(',')
        table_n = all_params[0].strip()
        num_avg = int(all_params[2].strip())
        column_n = all_params[1].strip()
        movavg = []

        for idx in range(num_avg, len(self.tables[table_n][column_n])):
            sum = 0
            for idx2 in range(idx-num_avg, idx):
                sum += float(self.tables[table_n][column_n][idx2])
            avg = sum / num_avg
            movavg += [round(avg,3)]

        self.tables.update({tn: {'movavg({})'.format(column_n): movavg}})
        print('movavg:', time.time() - st)
    def movsum(self, operation, tn):
        st = time.time()
        all_params = operation.split(',')
        table_n = all_params[0].strip()
        num_avg = int(all_params[2].strip())
        column_n = all_params[1].strip()
        movsum = []
        #self.tables[table_n][column_n] = self.tables[table_n][column_n][0].tolist()
        for idx in range(num_avg, len(self.tables[table_n][column_n])):
            sum = 0
            for idx2 in range(idx - num_avg, idx):
                sum += float(self.tables[table_n][column_n][idx2])
            movsum += [sum]

        self.tables.update({tn:{'movesum({})'.format(column_n): movsum}})
        print('movsum:', time.time() - st)
        return movsum

    def concat(self, operation, tn):
        st  = time.time()
        table_ns = operation.split(',')
        t1 = table_ns[0].strip()
        t2 = table_ns[1].strip()
        if list(self.tables[t1].keys()) == list(self.tables[t2].keys()):
            output = {tn:{}}
            for key in list(self.tables[t1].keys()):
                output[tn][key] = self.tables[t1][key] + self.tables[t2][key]


            self.tables.update(output)
        else:
            print('can not concat tables with different schemas')

        print('concat:', time.time() - st)

    def outputtofile(self, operation, tn):
        st = time.time()
        params = operation.split(',')
        table_n = params[0].strip()
        output_n = params[0].strip()
        table_len = len(self.tables[table_n][list(self.tables[table_n].keys())[0]])
        with open('{}.txt'.format(output_n), 'w') as file:
            for i in range(table_len):
                line = ''
                for key in list(self.tables[table_n].keys()):
                    line += str(self.tables[table_n][key][i])+'|'
                line = line[:-1]
                file.write(line+os.linesep)
        file.close()


        print('outputfile:', time.time() - st)

    def Btree(self, operation, tn):
        st = time.time()
        params = operation.split(',')
        table_n = params[0].strip()
        col_n = params[1].strip()

        output = OOBTree()
        if table_n not in list(self.Btree_index.keys()):
            self.Btree_index.update({table_n: output})

        if col_n in list(self.Btree_index[table_n].keys()):
            print('Btree index for {} already exists'.format(col_n))

        else:
            self.Btree_index[table_n].update({col_n:OOBTree()})
            for i, value in enumerate(self.tables[table_n][col_n]):
                if value not in list(self.Btree_index[table_n][col_n].keys()):
                    self.Btree_index[table_n][col_n][value] = [i]
                else:
                    self.Btree_index[table_n][col_n][value] += [i]

        print('Btree:', time.time() - st)
    def Hash(self, operation, tn):
        st = time.time()
        params = operation.split(',')
        table_n = params[0].strip()
        col_n = params[1].strip()

        if table_n not in list(self.Hash_index.keys()):
            self.Hash_index.update({table_n:{}})

        if col_n in list(self.Hash_index[table_n].keys()):
            print('Hash index for {} already exists'.format(col_n))
        else:
            self.Hash_index[table_n][col_n] = {}
            for i, value in enumerate(self.tables[table_n][col_n]):
                if value not in list(self.Hash_index[table_n][col_n].keys()):
                    self.Hash_index[table_n][col_n][value] = [i]
                else:
                    self.Hash_index[table_n][col_n][value] += [i]
        print('Hash:', time.time() - st)

    def count(self, operation, tn):
        st = time.time()
        table_n = operation.strip()
        self.tables.update({tn:len(self.tables[table_n][list(self.tables[table_n].keys()[0])])})
        print('count:', time.time() - st)

    def countgroup(self,operation, tn):
        st = time.time()
        table_n = operation.split(',')[0].strip()
        self.tables.update({tn: len(self.tables[table_n][list(self.tables[table_n].keys()[0])])})
        print('countgroup:', time.time() - st)




if __name__ == '__main__':
    input_file = sys.argv[1]
    Database(input_file)


