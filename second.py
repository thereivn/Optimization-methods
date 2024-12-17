import gym
from gym import spaces, envs

# Создаем окружение CartPole
env = gym.make('CartPole-v1', render_mode='human')
print(env.action_space)  # > Discrete(2)
print(env.observation_space)  # > Box(4,)
print(env.observation_space.high)  # > array([ 2.4 , inf, 0.20943951, inf])
print(env.observation_space.low)  # > array([-2.4 , -inf, -0.20943951, -inf])

space = spaces.Discrete(8)  # Набор из 8 элементов {0, 1, 2, ..., 7}
x = space.sample()
assert space.contains(x)
assert space.n == 8

# Получаем все зарегистрированные окружения
print(list(envs.registry.values()))

# Запускаем 20 эпизодов
for i_episode in range(20):
    observation, info = env.reset()  # Сбрасываем окружение
    for t in range(100):
        env.render()  # Отображаем окружение
        print(observation)  # Выводим текущее состояние

        action = env.action_space.sample()  # Выбираем случайное действие
        observation, reward, terminated, truncated, info = env.step(action)  # Выполняем действие

        if terminated or truncated:  # Если эпизод завершен
            print("Episode finished after {} timesteps".format(t + 1))
            break

# Закрываем окружение
env.close()