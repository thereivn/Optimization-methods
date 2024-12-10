import numpy as np

# Определяем целевую функцию
def f(x):
    x1, x2 = x
    return x1 - 1.4 * x2 + np.exp(0.01 * x1 + 0.11 * x2)

# Определяем градиент функции
def grad_f(x):
    x1, x2 = x
    df_dx1 = 1 + 0.01 * np.exp(0.01 * x1 + 0.11 * x2)
    df_dx2 = -1.4 + 0.11 * np.exp(0.01 * x1 + 0.11 * x2)
    return np.array([df_dx1, df_dx2])

# Градиентный метод с адаптивным шагом
def gradient_method(starting_point, step_size, tolerance, max_iterations=1000):
    x = np.array(starting_point)
    iterations = 0
    with open('gradient_method_iterations.txt', 'w') as f_out:
        while iterations < max_iterations:
            iterations += 1
            gradient = grad_f(x)
            if np.linalg.norm(gradient) < tolerance:  # Проверка на нулевой градиент
                break
            x_new = x - step_size * gradient
            f_out.write(f"{iterations}\t{x_new[0]}\t{x_new[1]}\t{f(x_new)}\n")
            print(f"Итерация {iterations}: точка = {x_new}, значение функции = {f(x_new)}")
            
            if np.linalg.norm(x_new - x) < tolerance:
                break
            x = x_new
    return x, f(x), iterations

# Эвристический алгоритм
def heuristic_method(starting_point, tolerance, max_iterations=1000, attempts_per_iteration=51):
    best_x = np.array(starting_point)
    best_f = f(best_x)
    iterations = 0
    tries = 0
    with open('heuristic_method_iterations.txt', 'w') as f_out:
        for _ in range(max_iterations):
            iterations += 1
            for _ in range(attempts_per_iteration):
                perturbation = np.random.uniform(-1, 1, size=2)
                x_new = best_x + perturbation
                current_f = f(x_new)
                tries += 1
                f_out.write(f"{tries} {x_new} {current_f}\n")
                print(f"Итерация {tries}: точка = {x_new}, значение функции = {current_f}")
                
                if current_f < best_f:
                    best_x, best_f = x_new, current_f
                    
            if np.abs(best_f - f(best_x)) < tolerance:
                break
                
    return best_x, best_f, iterations + tries

# Начальные параметры
starting_point = (1, 0)
tolerance = 0.0001
step_size = 0.01

# Запуск методов
grad_result = gradient_method(starting_point, step_size, tolerance)
heuristic_result = heuristic_method(starting_point, tolerance)

# Вывод результатов
print("\nРезультаты:")
print("Градиентный метод:")
print(f"Точка минимума: {grad_result[0]}, Значение функции: {grad_result[1]}, Итерации: {grad_result[2]}")

print("\nЭвристический метод:")
print(f"Точка минимума: {heuristic_result[0]}, Значение функции: {heuristic_result[1]}, Итерации: {heuristic_result[2]}")

# Сравнение методов
if grad_result[1] < heuristic_result[1]:
    print("\nГрадиентный метод более точен.")
else:
    print("\nЭвристический метод более точен.")
