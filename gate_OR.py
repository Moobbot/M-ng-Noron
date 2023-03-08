# working of OR gate

def OR(a, b):
    if a == 1:
        return 1
    elif b == 1:
        return 1
    else:
        return 0


# Driver code
if __name__ == '__main__':
    print(OR(0, 0))

    print("+---------------+----------------+")
    print(" | OR Truth Table | Result |")
    print(" A = 0, B = 0 | A AND B =", OR(0, 0), " | ")
    print(" A = 0, B = 1 | A AND B =", OR(0, 1), " | ")
    print(" A = 1, B = 0 | A AND B =", OR(1, 0), " | ")
    print(" A = 1, B = 1 | A AND B =", OR(1, 1), " | ")
