import sys
import os
import gym
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output
from collections import defaultdict
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

# Создаем среду Taxi-v3
env = gym.make('Taxi-v3', render_mode='human')
env.reset()
env.render()

# Получаем количество состояний и действий
n_states = env.observation_space.n
n_actions = env.action_space.n
print("n_states=%i, n_actions=%i" % (n_states, n_actions))

PREFERRED_ACTION_BIAS = 0.1  # Константа для уклона предпочтительных действий

def initialize_policy(n_states, n_actions):
    # Инициализация с небольшим уклоном в сторону более предпочтительных действий
    policy = np.random.rand(n_states, n_actions)
    policy += PREFERRED_ACTION_BIAS  # Увеличиваем вероятность действия 0
    policy = np.clip(policy, 0, None)  # Убедимся, что нет отрицательных значений
    policy = policy / policy.sum(axis=1, keepdims=True)  # Нормализация

    return policy

# Инициализируем политику
policy = initialize_policy(n_states, n_actions)

# Проверяем, что политика корректна
assert type(policy) in (np.ndarray, np.matrix), "Политика должна быть массивом или матрицей"
assert np.allclose(np.sum(policy, axis=1), 1), "Сумма вероятностей по состояниям должна быть равна 1"

def generate_session(env, policy, t_max=10**4):
    states, actions = [], []
    total_reward = 0.
    s, info = env.reset()
    
    # Извлекаем информацию о позиции такси и цели
    initial_position = s

    for t in range(t_max):
        a = np.random.choice(n_actions, p=policy[s])
        result = env.step(a)
        
        if len(result) == 2:
            new_s, r = result
            done = False
        else:
            new_s, r, done = result[:3]

        # Декодируем состояние
        taxi_row, taxi_col, passenger_index, destination_index = env.decode(new_s)
        
        # Проверка, приближается ли агент к цели
        if passenger_index != 0 and destination_index == passenger_index:
            r += 1  # Добавляем награду за приближение к цели

        # Вывод промежуточных шагов
        print(f"Step: {t}, State: {s}, Action: {a}, New State: {new_s}, Reward: {r}, Done: {done}, ")

        states.append(s)
        actions.append(a)
        total_reward += r
        s = new_s
        
        if done:
            break
            
    return states, actions, total_reward

def select_elites(states_batch, actions_batch, rewards_batch, percentile):
    """
    Выберите состояния и действия из игры, которые имеют награды >= процентиль.
    :param states_batch: список списков состояний
    :param actions_batch: список списков действий
    :param rewards_batch: список наград
    :returns: elite_states, elite_actions
    """
    reward_threshold = np.percentile(rewards_batch, percentile)
    
    elite_states = []
    elite_actions = []
    
    for i in range(len(rewards_batch)):
        if rewards_batch[i] >= reward_threshold:
            elite_states.extend(states_batch[i])
            elite_actions.extend(actions_batch[i])
    
    return elite_states, elite_actions

# Пример данных для тестирования
states_batch = [
    [1, 2, 3],  # игра1
    [4, 2, 0, 2],  # игра2
    [3, 1],  # игра3
]

actions_batch = [
    [0, 2, 4],  # игра1
    [3, 2, 0, 1],  # игра2
    [3, 3],  # игра3
]

rewards_batch = [
    3,  # игра1
    4,  # игра2
    5,  # игра3
]

# Тестирование функции select_elites
test_result_0 = select_elites(states_batch, actions_batch, rewards_batch, percentile=0)
test_result_30 = select_elites(states_batch, actions_batch, rewards_batch, percentile=30)
test_result_90 = select_elites(states_batch, actions_batch, rewards_batch, percentile=90)
test_result_100 = select_elites(states_batch, actions_batch, rewards_batch, percentile=100)

assert np.all(test_result_0[0] == [1, 2, 3, 4, 2, 0, 2, 3, 1]) \
       and np.all(test_result_0[1] == [0, 2, 4, 3, 2, 0, 1, 3, 3]), \
       "Для процентиля 0 вы должны вернуть все состояния и действия в хронологическом порядке."
assert np.all(test_result_30[0] == [4, 2, 0, 2, 3, 1]) and \
       np.all(test_result_30[1] == [3, 2, 0, 1, 3, 3]), \
       "Для процентиля 30 вы должны выбрать состояния/действия только из двух первых."
assert np.all(test_result_90[0] == [3, 1]) and \
       np.all(test_result_90[1] == [3, 3]), \
       "Для процентиля 90 вы должны выбирать состояния/действия только из одной игры."
assert np.all(test_result_100[0] == [3, 1]) and \
       np.all(test_result_100[1] == [3, 3]), \
       "Убедитесь, что вы используете >=, а не >. Также дважды проверьте, как вы вычисляете процентиль."

def get_new_policy(elite_states, elite_actions):
    """
    Учитывая список лучших состояний/действий от select_elites,
    возвращает новую политику, где вероятность каждого действия
    пропорциональна количеству появлений s_i и a_i в элитарных состояниях/действиях.
    Не забудьте нормализовать политику, чтобы получить действительные вероятности.
    Для состояний, в которых вы никогда не находились, используйте равномерное распределение.
    :param elite_states: одномерный список состояний лучших сессий.
    :param elite_actions: одномерный список действий лучших сессий.
    """
    new_policy = np.zeros((n_states, n_actions))
    state_action_counts = defaultdict(lambda: defaultdict(int))
    
    # Подсчет количеств состояний и действий
    for s, a in zip(elite_states, elite_actions):
        state_action_counts[s][a] += 1
    
    # Установка вероятностей для действий
    for s in range(n_states):
        total_count = sum(state_action_counts[s].values())
        if total_count > 0:
            for a in range(n_actions):
                new_policy[s, a] = state_action_counts[s][a] / total_count
        else:
            new_policy[s] = 1.0 / n_actions  # Равномерное распределение для неизвестных состояний
    
    return new_policy

def show_progress(rewards_batch, log, percentile, reward_range=[-990, +10]):
    """
    Удобная функция, отображающая прогресс обучения
    """
    mean_reward = np.mean(rewards_batch)
    threshold = np.percentile(rewards_batch, percentile)
    log.append([mean_reward, threshold])
    
    plt.figure(figsize=[8, 4])
    
    # График средних наград и порогов
    plt.subplot(1, 2, 1)
    plt.plot(list(zip(*log))[0], label='Mean rewards')
    plt.plot(list(zip(*log))[1], label='Reward thresholds')
    plt.legend()
    plt.grid()
    
    # Гистограмма наград
    plt.subplot(1, 2, 2)
    plt.hist(rewards_batch, range=reward_range)
    plt.vlines([np.percentile(rewards_batch, percentile)],
               [0], [100], label="percentile", color='red')
    plt.legend()
    plt.grid()
    
    clear_output(True)
    print("mean reward = %.3f, threshold=%.3f" % (mean_reward, threshold))
    plt.show()

# Сбросить политику на всякий случай
policy = initialize_policy(n_states, n_actions)

# Эксперимент
n_sessions = 500  # число сессий
percentile = 70  # процент сессий с наивысшей наградой
learning_rate = 0.7  # насколько быстро обновляется политика, по шкале от 0 до 1
log = []

for i in range(100):
    # Генерирование списка n_sessions новых сессий
    sessions = [generate_session(env, policy) for _ in range(n_sessions)]
    
    # Извлечение состояний, действий и наград из сессий
    states_batch = [session[0] for session in sessions]
    actions_batch = [session[1] for session in sessions]
    rewards_batch = [session[2] for session in sessions]
    
    # Выбор элитных состояний и действий
    elite_states, elite_actions = select_elites(states_batch, actions_batch, rewards_batch, percentile)
    
    # Обновление политики
    policy = get_new_policy(elite_states, elite_actions)
    
    # Отображение прогресса
    show_progress(rewards_batch, log, percentile)


# Закрытие среды
env.close()