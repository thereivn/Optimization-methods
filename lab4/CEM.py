import sys
import os
import gym
import numpy as np

# Создаем среду Taxi-v3
env = gym.make('Taxi-v3', render_mode='human')
env.reset()
env.render()

# Получаем количество состояний и действий
n_states = env.observation_space.n
n_actions = env.action_space.n
print("n_states=%i, n_actions=%i" % (n_states, n_actions))

def initialize_policy(n_states, n_actions):
    # Создаем массив для хранения вероятности действий
    policy = np.full((n_states, n_actions), 1.0 / n_actions)  # Равномерная политика
    return policy

# Инициализируем политику
policy = initialize_policy(n_states, n_actions)

# Проверяем, что политика корректна
assert type(policy) in (np.ndarray, np.matrix)
assert np.allclose(policy, 1. / n_actions)
assert np.allclose(np.sum(policy, axis=1), 1)

# Пример использования политики (можно добавить для проверки)
state, info = env.reset()  # Изменено на возвращение информации
done = False

while not done:
    # Убедитесь, что state является допустимым индексом
    if state < 0 or state >= n_states:
        print(f"Неверное состояние: {state}")
        break
    
    action = np.random.choice(n_actions, p=policy[state])  # Выбор действия согласно политике
    observation, reward, terminated, truncated, info = env.step(action)  # Выполняем действие

    env.render()
    state = observation  # Обновляем состояние

    # Проверяем завершение эпизода
    if terminated or truncated:  # Если эпизод завершен
        print("Episode finished.")
        break

env.close()