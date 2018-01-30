import sys
import pandas as pd

def main():
    data = pd.read_csv(sys.argv[1])
    print data.describe()

if __name__ == '__main__':
    main()
