import gym

# Создание среды
env = gym.make('CartPole-v1', render_mode='human')

# Сброс среды
env.reset()

# Запуск цикла
for _ in range(1000):
    action = env.action_space.sample()  # Случайное действие
    env.step(action)  # Выполнение действия
    env.render()  # Отображение среды

# Закрытие среды
env.close()