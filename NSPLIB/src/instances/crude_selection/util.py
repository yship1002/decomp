def format_table(file, save_file):
    """Format the GAMS table into Pyomo data (.dat) format.

    This function only takes input files with a single table.

    Args:
        file (str): The path for the input file.
        save_file (str): The path for saving the formatted table.
    """

    # initialize
    data = {}
    new_output = ''
    idx_1 = list()
    idx_2 = list()

    # read file
    with open(file) as f:
        lines = f.readlines()

    # read first line
    title = lines[0].split()[1]
    table_name = title.split('(')[0]
    set_1_name, set_2_name = title.split('(')[1].split(')')[0].split(',')
    set_1_name, set_2_name = set_1_name.upper(), set_2_name.upper()

    # read second line
    idx_2_line = lines[1].split()
    for i in idx_2_line:
        try:
            i = int(i)
        except ValueError:
            pass
        idx_2.append(i)

    # read data
    for l in lines[2:]:
        _idx_1, l_data = l.split()[0], l.split()[1:]
        try:
            _idx_1 = int(_idx_1)
        except ValueError:
            pass
        idx_1.append(_idx_1)
        for i, d in enumerate(l_data):
            data[_idx_1, idx_2[i]] = float(d)

    # title
    new_output += f'table {table_name}({set_1_name},{set_2_name}) :\n'
    # second line
    new_output += f'{set_1_name} {set_2_name} {table_name} := \n'

    # data
    for d in data:
        new_output += f'{d[0]}\t{d[1]}\t{data[d]}\n'

    # ending semicolon
    new_output += ';\n'

    # write file
    with open(save_file, 'w') as f:
        f.write(new_output)

    return

def format_param(file, save_file):
    """Format the GAMS parameter into Pyomo data (.dat) format.

    This function only takes input files with a single param.

    Args:
        file (str): The path for the input file.
        save_file (str): The path for saving the formatted table.
    """

    # initialize
    data = {}
    new_output = ''

    # read file
    with open(file) as f:
        lines = f.readlines()

    # read first line
    title = lines[0].split()[0]
    param_name = title.split('(')[0]
    set_name = title.split('(')[1].split(')')[0].upper()


    # read data
    for l in lines[1:]:
        l_data = [_ for _ in l.split()]
        for _d in l_data.copy():
            if _d == '/':
                l_data.remove(_d)
            elif '/' in _d:
                l_data.remove(_d)
                _new_d = _d.split('/')[0]
                l_data.append(_new_d)
        _idx, _d = l_data
        try:
            _idx = int(_idx)
        except ValueError:
            pass
        data[_idx] = float(_d)

    # title
    new_output += f'param {param_name} :=\n'

    # data
    for d in data:
        new_output += f'{d}\t{data[d]}\n'

    # ending semicolon
    new_output += ';\n'

    # write file
    with open(save_file, 'w') as f:
        f.write(new_output)
    
    return
