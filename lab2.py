import numpy as np
import time

# Определим целевую функцию
def f(x):
    x1, x2 = x
    return x1 - 1.4 * x2 + np.exp(0.01 * x1 + 0.11 * x2)

# Определим градиент функции
def gradient(x):
    x1, x2 = x
    df_dx1 = 1 + 0.01 * np.exp(0.01 * x1 + 0.11 * x2)
    df_dx2 = -1 + 0.11 * np.exp(0.01 * x1 + 0.11 * x2)
    return np.array([df_dx1, df_dx2])

# Градиентный метод с постоянным шагом
def gradient_method(x0, epsilon, step_size=0.1, max_iterations=100):
    x = np.array(x0)
    iterations = 0
    steps = [x.copy()]

    while iterations < max_iterations:
        grad = gradient(x)
        x_new = x - step_size * grad
        iterations += 1
        steps.append(x_new.copy())
        
        if np.linalg.norm(x_new - x) < epsilon:
            break
        x = x_new

    return x, iterations, steps

# Улучшенный эвристический алгоритм
def improved_heuristic_algorithm(x0, epsilon, max_iter=1000, initial_step_range=5, decay_factor=0.95):
    x = np.array(x0)
    best_x = x
    best_f = f(x)
    iterations = 0
    steps = [x.copy()]
    step_range = initial_step_range

    for _ in range(max_iter):
        # Генерация нового решения с адаптивным диапазоном случайных шагов
        x_new = x + np.random.uniform(-step_range, step_range, size=x.shape)
        iterations += 1
        steps.append(x_new.copy())

        if f(x_new) < best_f:
            best_f = f(x_new)
            best_x = x_new
            step_range = initial_step_range  # Сброс шага, если найдено лучшее решение
        else:
            step_range *= decay_factor  # Уменьшение шага, если улучшение не найдено

        if np.linalg.norm(best_x - x) < epsilon:
            break
        
        x = best_x  # Обновляем текущее решение

    return best_x, iterations, steps

# Начальные параметры
x0 = (1, 0)
epsilon = 0.0001

# Запуск градиентного метода с таймером
start_time = time.time()
result_gradient, iterations_gradient, steps_gradient = gradient_method(x0, epsilon)
end_time = time.time()

# Запуск улучшенного эвристического алгоритма с таймером
start_time1 = time.time()
result_heuristic, iterations_heuristic, steps_heuristic = improved_heuristic_algorithm(x0, epsilon)
end_time1 = time.time()

# Вывод результатов
print("Градиентный метод:")
for step in steps_gradient:
    print(f"Шаг: {step}, f(x) = {f(step)}")
print("Эвристический алгоритм:")
for step in steps_heuristic:
    print(f"Шаг: {step}, f(x) = {f(step)}")

print(f"\nГрадиентный метод завершен за {end_time - start_time:.4f} секунд.")
print(f"Эвристический алгоритм завершен за {end_time1 - start_time1:.4f} секунд.")

print("\nРЕЗУЛЬТАТЫ ВЫПОЛНЕНИЯ МЕТОДОВ:")
print("___Градиентный метод:___")
print(f"Итог: x = {result_gradient}, итерации = {iterations_gradient}, f(x) = {f(result_gradient)}\n")
print("___Эвристический алгоритм:___")
print(f"Итог: x = {result_heuristic}, итерации = {iterations_heuristic}, f(x) = {f(result_heuristic)}\n")

# Сравнение результатов
print("Сравнение результатов:")
if iterations_gradient < iterations_heuristic:
    print("Градиентный метод быстрее.")
elif iterations_gradient > iterations_heuristic:
    print("Эвристический алгоритм быстрее.")
else:
    print("Оба метода имеют одинаковое количество итераций.")

if np.abs(f(result_gradient) - f(result_heuristic)) < epsilon:
    print("Оба метода достигают схожей точности.")
else:
    print("Методы показывают разные результаты по точности.")