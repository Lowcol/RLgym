import gymnasium as gym
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# fixed policy: stick only on 20 or 21
def fixed_policy(state):
    player_sum, dealer_card, usable_ace = state
    return 0 if player_sum >= 20 else 1  # 0=stick, 1=hit

def mc_prediction_first_visit(policy, env, episodes=500000, gamma=1.0):
    returns_sum = defaultdict(float)
    returns_count = defaultdict(float)
    V = defaultdict(float)

    for i in range(episodes):
        if (i + 1) % 50000 == 0:
            print(f"Episode {i+1}/{episodes}")
        episode = []
        state, _ = env.reset()
        done = False
        # generate episode
        while not done:
            action = policy(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            episode.append((state, action, reward))
            state = next_state
            done = terminated or truncated

        # calculate returns Gt for all the episodes
        G = 0.0
        visited = set()
        for (s, a, r) in reversed(episode):
            G = gamma * G + r
            if s not in visited:
                returns_sum[s] += G
                returns_count[s] += 1.0
                V[s] = returns_sum[s] / returns_count[s]
                visited.add(s)

    return V

def make_grid_values(V):
    # Sutton & Barto show player sums 12..21 and dealer 1..10
    player_range = np.arange(12, 22)   # 12..21
    dealer_range = np.arange(1, 11)    # 1..10

    # grids for usable=True and usable=False
    grid_true = np.full((player_range.size, dealer_range.size), np.nan)
    grid_false = np.full((player_range.size, dealer_range.size), np.nan)

    for i, ps in enumerate(player_range):
        for j, dc in enumerate(dealer_range):
            s_true = (ps, dc, True)
            s_false = (ps, dc, False)
            if s_true in V:
                grid_true[i, j] = V[s_true]
            if s_false in V:
                grid_false[i, j] = V[s_false]

    return player_range, dealer_range, grid_true, grid_false

def plot_surfaces(player_range, dealer_range, grid_true, grid_false, title):
    X, Y = np.meshgrid(dealer_range, player_range)  # X=dealer, Y=player

    fig = plt.figure(figsize=(14, 6))
    fig.suptitle(title, fontsize=16)

    ax1 = fig.add_subplot(121, projection='3d')
    surf1 = ax1.plot_surface(X, Y, grid_true, rstride=1, cstride=1, edgecolor='k', linewidth=0.2, antialiased=True)
    ax1.set_xlabel('Dealer showing')
    ax1.set_ylabel('Player sum')
    ax1.set_zlabel('Value')
    ax1.set_title('Usable Ace')

    ax2 = fig.add_subplot(122, projection='3d')
    surf2 = ax2.plot_surface(X, Y, grid_false, rstride=1, cstride=1, edgecolor='k', linewidth=0.2, antialiased=True)
    ax2.set_xlabel('Dealer showing')
    ax2.set_ylabel('Player sum')
    ax2.set_zlabel('Value')
    ax2.set_title('No Usable Ace')

    plt.tight_layout(rect=[0, 0.0, 1, 0.95])
    plt.show()

if __name__ == "__main__":
    # use the Sutton & Barto variant
    env = gym.make("Blackjack-v1", sab=True)

    V_10k = mc_prediction_first_visit(fixed_policy, env, episodes=10000)
    pr, dr, g_t, g_f = make_grid_values(V_10k)
    plot_surfaces(pr, dr, g_t, g_f, "After 10,000 episodes")

    V_500k = mc_prediction_first_visit(fixed_policy, env, episodes=500000)
    pr, dr, g_t, g_f = make_grid_values(V_500k)
    plot_surfaces(pr, dr, g_t, g_f, "After 500,000 episodes")
