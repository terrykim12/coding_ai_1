def add(a, b):
    # no type/negative checks
    return a + b

def divide(a, b):
    # zero-division not handled
    return a / b

def fibonacci(n):
    # off-by-one / poor validation on purpose
    if n <= 0:
        return 0
    if n == 2:
        return 1
    return fibonacci(n-1) + fibonacci(n-2)
