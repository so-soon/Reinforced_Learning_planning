import numpy as np


def policy_evaluation(env, policy, gamma=0.99, theta=1e-8):
    old_V = np.zeros(env.nS)  # V(K)(S)
    h = 1.0  # delta
    while h >= theta :
        h = 0.0
        new_V = np.zeros(env.nS)  # V(K+1)(S)
        for state in range(env.nS):
            temp_v = 0.0
            for a in range(4): # a in all action
                for next_s in env.MDP[state][a]:
                    temp_v += policy[state][a]*next_s[0]*(next_s[2] + gamma*old_V[next_s[1]]) # Bellman Expectation Eqn.
            new_V[state] = temp_v
            h = max(h,np.abs(old_V[state]-new_V[state]))
        old_V = new_V
    return old_V

def policy_improvement(env, policy, V, gamma=0.99):
    policy_stable = True
    new_policy = np.zeros([env.nS, env.nA])
    for state in range(env.nS):
        old_action = np.argmax(policy[state])
        action_values = []
        for a in range(4):
            temp_q = 0.0
            for next_s in env.MDP[state][a]:
                temp_q += next_s[0]*(next_s[2] + gamma*V[next_s[1]])
            action_values.append(temp_q)
        new_action = np.argmax(action_values) # find new action Greedily
        new_policy[state][new_action] = 1
        if old_action != new_action: # not optimal policy(policy change)
            policy_stable = False
    return new_policy,policy_stable


def policy_iteration(env, gamma=0.99, theta=1e-8):
    V = np.zeros(env.nS)
    policy = np.ones([env.nS, env.nA]) / env.nA
    flags = False # flags indicate policy stability
    while not flags:
        V = policy_evaluation(env, policy)
        policy , flags = policy_improvement(env, policy, V)

    return policy, V

def value_iteration(env, gamma=0.99, theta=1e-8):
    V = np.zeros(env.nS)
    policy = np.zeros([env.nS, env.nA])
    h = 1.0  # delta

    while h>=theta:
        h = 0.0
        new_V = np.zeros(env.nS)
        policy = np.zeros([env.nS, env.nA])
        for state in range(env.nS):
            temp_v = 0.0
            action_values = []
            for a in range(4):
                temp_q = 0.0
                for next_s in env.MDP[state][a]:
                    temp_q += next_s[0] * (next_s[2] + gamma * V[next_s[1]])
                action_values.append(temp_q)
            new_V[state] = np.max(action_values)  # Bellman Optimality Eqn.
            policy[state][np.argmax(action_values)]=1
            h = max(h, np.abs(V[state] - new_V[state]))
        V = new_V

    return policy, V
