import gym

# Создаем окружение CartPole
#env = gym.make('CartPole-v1', render_mode='rgb_array')
env = gym.make('CartPole-v1', render_mode='human')

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