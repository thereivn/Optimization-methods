import gym

# Создаем окружение Taxi-v3
env = gym.make('Taxi-v3', render_mode='human')

# Запускаем 10 случайных действий
for _ in range(10):
    observation, info = env.reset()  # Сбрасываем окружение
    done = False
    step = 0

    while not done:
        env.render()  # Отображаем текущее состояние среды
        
        action = env.action_space.sample()  # Выбираем случайное действие
        observation, reward, terminated, truncated, info = env.step(action)  # Выполняем действие
        
        # Выводим информацию о текущем шаге
        print(f"Step: {step}")
        print(f"State (observation): {observation}")
        print(f"Action: {action}")
        print(f"Reward: {reward}")
        
        step += 1
        
        if terminated or truncated:  # Если эпизод завершен
            print("Episode finished.")
            break

# Закрываем окружение
env.close()