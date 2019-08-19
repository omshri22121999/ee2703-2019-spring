import sys


if __name__ == '__main__':
    arg = sys.argv[0]
    if(len(sys.argv) != 1):
        print("Wrong number of arguments!")
    else:
        print(arg)
