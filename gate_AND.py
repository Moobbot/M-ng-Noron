# working of AND gate
def AND(a, b):

    if a == 1 and b == 1:
        return 1
    else:
        return 0


# Driver code
if __name__ == '__main__':
    print(AND(1, 1))

    print("+---------------+----------------+")
    print(" | AND Truth Table | Result |")
    print(" A = 0, B = 0 | A AND B =", AND(0, 0), " | ")
    print(" A = 0, B = 1 | A AND B =", AND(0, 1), " | ")
    print(" A = 1, B = 0 | A AND B =", AND(1, 0), " | ")
    print(" A = 1, B = 1 | A AND B =", AND(1, 1), " | ")
