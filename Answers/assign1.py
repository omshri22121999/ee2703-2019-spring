# Assignment 1 for EE2703
# Done by Om Shri Prasath, EE17B113
# Date : 25/1/2019

import sys

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
                with open(filename, "r") as f:
                    formatted_lines = []
                    # Obtaining the lines in the file, removing the comments, as a list of words
                    [formatted_lines.append(line.split('#')[0].split())
                     for line in f.readlines()]
                    # Checking whether there is circuit in the netlist file
                    if ['.circuit'] in formatted_lines:
                        formatted_lines.reverse()
                        check = False
                        for line in formatted_lines:
                            if line == ['.end']:
                                check = True  # If there is circuit, start printing
                            elif line == ['.circuit']:
                                check = False  # If there is end, stop printing
                            elif check:
                                # Print if the condition is true
                                print(" ".join(line[::-1]))
                    else:
                        # Printing error for circuit not found
                        print('No circuit found!')
            except FileNotFoundError:
                # Printing error for file not found
                print("File does not exist!")
