# Assignment 2 for EE2703
# Done by Om Shri Prasath, EE17B113
# Date : 4/2/2019

# Importing libraries

import sys
import numpy as np
import pandas
import math
import cmath

# Classes for different components


class resistor:
    def __init__(self, name, node1, node2, value):
        self.name = name
        self.node1 = node1
        self.node2 = node2
        self.value = value


class capacitor:
    def __init__(self, name, node1, node2, value):
        self.name = name
        self.node1 = node1
        self.node2 = node2
        self.value = value


class inductor:
    def __init__(self, name, node1, node2, value):
        self.name = name
        self.node1 = node1
        self.node2 = node2
        self.value = value


class volt_source:
    def __init__(self, name, node1, node2, value, phase=0):
        self.name = name
        self.node1 = node1
        self.node2 = node2
        self.value = value
        self.phase = phase


class curr_source:
    def __init__(self, name, node1, node2, value, phase=0):
        self.name = name
        self.node1 = node1
        self.node2 = node2
        self.value = value
        self.phase = phase


class vcvs:
    def __init__(self, name, node1, node2, node3, node4, value):
        self.name = name
        self.node1 = node1
        self.node2 = node2
        self.node3 = node3
        self.node4 = node4
        self.value = value


class vccs:
    def __init__(self, name, node1, node2, node3, node4, value):
        self.name = name
        self.node1 = node1
        self.node2 = node2
        self.node3 = node3
        self.node4 = node4
        self.value = value


class ccvs:
    def __init__(self, name, node1, node2, volt_name, value):
        self.name = name
        self.node1 = node1
        self.node2 = node2
        self.volt_name = volt_name
        self.value = value


class cccs:
    def __init__(self, name, node1, node2, volt_name, value):
        self.name = name
        self.node1 = node1
        self.node2 = node2
        self.volt_name = volt_name
        self.value = value


# Function to parse value given in alpha numeric form to numeric form

def parse_value(val):

    if('k' in val):
        new_val = val.replace('k', "")
        return(float(new_val)*1000)
    elif('m' in val):
        new_val = val.replace('m', "")
        return(float(new_val)*0.001)
    elif(not(str.isalpha(val))):
        return(float(val))


if __name__ == '__main__':  # Main Function of the Program
    if(len(sys.argv) != 2):  # Checking whether number of arguments are correct
        # Printing error message for wrong number of arguments
        print("Wrong number of arguments!")
    else:
        filename = sys.argv[1]  # Accepting the filename of netlist file
        if(not filename.endswith(".netlist")):  # Checking whether input is netlist
            print("Wrong filetype!")  # Printing error as output
        else:
            try:
                nodes = []
                node_dict = {}
                freq = 1e-100
                formatted_lines = []
                components = {'r': [], 'c': [], 'l': [], 'v': [],
                              'i': [], 'h': [], 'g': [], 'e': [], 'f': []}
                with open(filename, "r") as f:
                    # Obtaining the lines in the file, removing the comments, as a list of words
                    [formatted_lines.append(line.split('#')[0].split())
                     for line in f.readlines()]
                    # Checking whether there is circuit in the netlist file
                if ['.circuit'] in formatted_lines:
                    check = False
                    for line in formatted_lines:
                        if line == ['.circuit']:
                            check = True  # If there is circuit, start getting values
                        elif line == ['.end']:
                            check = False  # If there is end, stop getting values
                        elif '.ac' in line:
                            freq = parse_value(line[2])
                        elif check:
                            # Add data to list
                            try:
                                # Resistor Check
                                if('R' in line[0]):
                                    components['r'].append(
                                        resistor(line[0], line[1], line[2], parse_value(line[3])))
                                    if(line[1] not in nodes):
                                        nodes.append(line[1])
                                    if(line[2] not in nodes):
                                        nodes.append(line[2])
                                # Capacitor Check
                                elif('C' in line[0]):
                                    components['c'].append(
                                        capacitor(line[0], line[1], line[2], parse_value(line[3])))
                                    if(line[1] not in nodes):
                                        nodes.append(line[1])
                                    if(line[2] not in nodes):
                                        nodes.append(line[2])
                                # Inductor Check
                                elif('L' in line[0]):
                                    components['l'].append(
                                        inductor(line[0], line[1], line[2], parse_value(line[3])))
                                    if(line[1] not in nodes):
                                        nodes.append(line[1])
                                    if(line[2] not in nodes):
                                        nodes.append(line[2])
                                # Voltage Source Check
                                elif('V' in line[0]):
                                    if(line[3] == 'ac'):
                                        components['v'].append(volt_source(
                                            line[0], line[1], line[2], parse_value(line[4])/(2*math.sqrt(2)), phase=parse_value(line[5])))
                                    else:
                                        components['v'].append(volt_source(
                                            line[0], line[1], line[2], parse_value(line[4])))
                                # Current Source Check
                                elif('I' in line[0]):
                                    if(line[3] == 'ac'):
                                        components['i'].append(curr_source(
                                            line[0], line[1], line[2], parse_value(line[4])/(2*math.sqrt(2)), phase=parse_value(line[5])))
                                    else:
                                        components['i'].append(curr_source(
                                            line[0], line[1], line[2], parse_value(line[4])))
                                # VCVS Check
                                elif('E' in line[0]):
                                    components['e'].append(
                                        vcvs(line[0], line[1], line[2], line[3], line[4], parse_value(line[5])))
                                # CCVS Check
                                elif('F' in line[0]):
                                    components['f'].append(
                                        ccvs(line[0], line[1], line[2], line[3], parse_value(line[4])))
                                # VCCS Check
                                elif('G' in line[0]):
                                    components['g'].append(
                                        vccs(line[0], line[1], line[2], line[3], line[4], parse_value(line[5])))
                                # CCCS Check
                                elif('H' in line[0]):
                                    components['h'].append(
                                        cccs(line[0], line[1], line[2], line[3], parse_value(line[4])))
                            except IndexError:
                                print("Wrong number of values given!")
                                exit()
                            except ValueError:
                                print("Wrong types of values given!")
                                exit()
                    # Setting the GND node as first(to make its node equation as Vgnd=0)
                    try:
                        nodes.remove('GND')
                        nodes = ['GND']+nodes
                    except:
                        print("No ground given!")
                        exit()
                    # Setting frequency value and inverse of frequency
                    # Creating a dictionary for nodes
                    for i in range(len(nodes)):
                        node_dict[nodes[i]] = i
                    # Matrix for storing coefficients (M)
                    M_matrix = np.zeros(
                        (len(nodes)+len(components['v']), len(nodes)+len(components['v'])), np.complex)
                    b_matrix = np.zeros(
                        (len(nodes)+len(components['v'])), np.complex)
                    # Equation for ground
                    M_matrix[0][0] = 1.0
                    # Equations for source voltages
                    for i in range(len(components['v'])):
                        source = components['v'][i]
                        l = len(nodes)
                        M_matrix[l+i][node_dict[source.node1]] = -1.0
                        M_matrix[l+i][node_dict[source.node2]] = 1.0
                        b_matrix[l+i] = cmath.rect(source.value,
                                                   source.phase*cmath.pi/180)
                    # Resistor equations
                    if(len(components['r']) != 0):
                        for res in components['r']:
                            if(node_dict[res.node1] != 0):
                                M_matrix[node_dict[res.node1]
                                         ][node_dict[res.node1]] += 1/res.value
                                M_matrix[node_dict[res.node1]
                                         ][node_dict[res.node2]] -= 1/res.value
                            if(node_dict[res.node2] != 0):
                                M_matrix[node_dict[res.node2]
                                         ][node_dict[res.node1]] -= 1/res.value
                                M_matrix[node_dict[res.node2]
                                         ][node_dict[res.node2]] += 1/res.value
                    # Capacitor equations
                    if(len(components['c']) != 0):
                        for cap in components['c']:
                            if(node_dict[cap.node1] != 0):
                                M_matrix[node_dict[cap.node1]
                                         ][node_dict[cap.node1]] += complex(0, 2*np.pi*freq*cap.value)
                                M_matrix[node_dict[cap.node1]
                                         ][node_dict[cap.node2]] -= complex(0, 2*np.pi*freq*cap.value)
                            if(node_dict[cap.node2] != 0):
                                M_matrix[node_dict[cap.node2]
                                         ][node_dict[cap.node2]] += complex(0, 2*np.pi*freq*cap.value)
                                M_matrix[node_dict[cap.node2]
                                         ][node_dict[cap.node1]] -= complex(0, 2*np.pi*freq*cap.value)
                    # Inductor equations
                    if(len(components['l']) != 0):
                        for ind in components['l']:
                            if(node_dict[ind.node1] != 0):
                                M_matrix[node_dict[ind.node1]
                                         ][node_dict[ind.node1]] -= complex(0, 1/(freq*2*np.pi*ind.value))
                                M_matrix[node_dict[ind.node1]
                                         ][node_dict[ind.node2]] += complex(0, 1/(freq*2*np.pi*ind.value))
                            if(node_dict[ind.node2] != 0):
                                M_matrix[node_dict[ind.node2]
                                         ][node_dict[ind.node2]] -= complex(0, 1/(freq*2*np.pi*ind.value))
                                M_matrix[node_dict[ind.node2]
                                         ][node_dict[ind.node1]] += complex(0, 1/(freq*2*np.pi*ind.value))
                    # Voltage source equations
                    if(len(components['v']) != 0):
                        l = len(nodes)
                        for vol in components['v']:
                            if(node_dict[vol.node1] != 0):
                                M_matrix[node_dict[vol.node1]][l] = -1.0
                            if(node_dict[vol.node2] != 0):
                                M_matrix[node_dict[vol.node2]][l] = 1.0
                    # Current source equations
                    if(len(components['i']) != 0):
                        for source in components['i']:
                            if(node_dict[source.node1] != 0):
                                b_matrix[node_dict[source.node1]] += cmath.rect(
                                    source.value, source.phase*cmath.pi/180)
                            if(node_dict[source.node2] != 0):
                                b_matrix[node_dict[source.node2]] -= cmath.rect(
                                    source.value, source.phase*cmath.pi/180)
                    # Solving equation function
                    x = np.linalg.solve(M_matrix, b_matrix)
                    # Rounding the value
                    x_rounded = np.around(x, 6)
                    x_polar = []
                    for i in x:
                        x_polar.append(cmath.polar(i*math.sqrt(2)))
                    i = []
                    # Data for printing clear data as output
                    for v in components['v']:
                        i.append("I in "+v.name)
                    # Printing data output
                    print(pandas.DataFrame(
                        x, columns=['Value'], index=nodes+i))
                else:
                    # Printing error for circuit not found
                    print('No circuit found!')
            except FileNotFoundError:
                # Printing error for file not found
                print("File does not exist!")
