import numpy as np

# Целевая функция
def f(x):
    return x**2 + 4 * np.exp(-0.25 * x)

# Алгоритм блочного равномерного поиска
def block_uniform_search(a, b, N):
    block_size = (b - a) / N
    min_value = float('inf')
    min_x = a
    
    steps = []  # Список для хранения шагов
    
    for i in range(N + 1):
        x = a + i * block_size
        steps.append((i + 1, x, f(x)))  # Сохраняем шаг, x и f(x)
        
        if f(x) < min_value:
            min_value = f(x)
            min_x = x
            
    # Вывод промежуточных шагов
    print("Блочный равномерный поиск:")
    print("Шаг |    x    |  f(x)")
    for step in steps:
        print(f"{step[0]:<4} | {step[1]:<8.5f} | {step[2]:<8.5f}")
    
    return min_x, min_value

# Метод Фибоначчи
def fibonacci_method(a, b, N):
    fib = [0, 1]
    for i in range(2, N + 2):
        fib.append(fib[i - 1] + fib[i - 2])
    
    k = N
    steps = []  # Список для хранения шагов
    
    while k > 1:
        x1 = a + (fib[k - 2] / fib[k]) * (b - a)
        x2 = a + (fib[k - 1] / fib[k]) * (b - a)
        
        steps.append((N - k + 1, x1, f(x1), x2, f(x2)))  # Сохраняем шаг, x1, f(x1), x2, f(x2)
        
        if f(x1) < f(x2):
            b = x2
        else:
            a = x1
            
        k -= 1
    
    min_x = (a + b) / 2
    min_value = f(min_x)
    
    # Вывод промежуточных шагов
    print("\nМетод Фибоначчи:")
    print("Шаг |    x1   |  f(x1) |    x2   |  f(x2)")
    for step in steps:
        print(f"{step[0]:<4} | {step[1]:<8.5f} | {step[2]:<8.5f} | {step[3]:<8.5f} | {step[4]:<8.5f}")

    return min_x, min_value

# Параметры
a, b = 0, 1
N = 23

# Полный вызов алгоритмов
min_x_block, min_value_block = block_uniform_search(a, b, 22)
min_x_fib, min_value_fib = fibonacci_method(a, b, 24)

# Длина отрезка неопределенности
uncertainty_length_block = b - min_x_block
uncertainty_length_fib = b - min_x_fib

# Результаты
print(f"\nБлочный равномерный поиск: x_min = {min_x_block}, \nf(x_min) = {min_value_block}, \nДлина отрезка неопределенности = {uncertainty_length_block}")
print(f"\n\nМетод Фибоначчи: x_min = {min_x_fib}, \nf(x_min) = {min_value_fib}, \nДлина отрезка неопределенности = {uncertainty_length_fib}")
