import numpy as np

# Markov decision process for a simple horizontal 1x5 grid

V = np.array([0.,0.,0.,0.,0.])
gamma = 0.5

#cols:
#             P: stay, left, right, left (hit), right (hit), left (leave wall), right (leave wall)
# action_stay :  [1/2, 1/4, 1/4, 0, 0, 1/2, 1/2]
# action_left :  [2/3, 1/3, 0, 1/2, 0, 1/3, 1/2]
# action_right : [2/3, 0, 1/3, 0, 1/2, 1/2, 1/3]

for i in range(10):
    V_old = V.copy()
    for v in range(len(V_old)):
        # Actions: left, stay, right
        if v == 0:
            V[0] = max([(1/2)*gamma*V_old[0] + (1/2)*gamma*V_old[1],
                        (1/2)*gamma*V_old[0] + (1/2)*gamma*V_old[1],
                        (2/3)*gamma*V_old[0] + (1/3)*gamma*V_old[1]])
        elif v == len(V_old) - 1:
            V[-1] = max([(2 / 3) * (1 + gamma * V_old[-1]) + (1 / 3) * (1 + gamma * V_old[-2]),
                        (1/2) * (1 + gamma * V_old[-1]) + (1 / 2) * (1 + gamma * V_old[-2]),
                        (1/2) * (1 + gamma * V_old[-1]) + (1/2) * (1 + gamma * V_old[-2])])
        else:
            V[v] = max([(1/3) * gamma * V_old[v-1] + (2/3) * gamma * V_old[v],
                        (1/4) * gamma * V_old[v-1] + (1/2) * gamma * V_old[v] + (1/4) * gamma * V_old[v+1],
                        (2/3) * gamma * V_old[v] + (1/3) * gamma * V_old[v+1]])
    print(V)