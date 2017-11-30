from gridworld import GridWorld, GridWorld_MDP
import numpy as np
import matplotlib.pyplot as plt
import random
import math
from matplotlib.backends.backend_pdf import PdfPages


def policy_iteration(mdp, gamma=1, iters=100, plot=True):
    '''
    Performs policy iteration on an mdp and returns the value function and policy 
    :param mdp: mdp class (GridWorld_MDP) 
    :param gam: discount parameter should be in (0, 1] 
    :param iters: number of iterations to run policy iteration for
    :param plot: boolean for if a plot should be generated for the utilities of the start state
    :return: two numpy arrays of length |S| and one of length iters.
    Two arrays of length |S| are U and pi where U is the value function
    and pi is the policy. The third array contains U(start) for each iteration
    the algorithm.
    '''
    pi = np.zeros(mdp.num_states, dtype=np.int)
    U = np.zeros(mdp.num_states)
    Ustart = []

    #TODO Implement policy iteration

    # Create the reward matrix with discounting for that step
    reward_matrix = np.zeros(mdp.num_states)
    # get reward for each state and discount
    for x in range(mdp.num_states):
        reward_matrix[x] = mdp.R(x)


    i = 1
    while i <= iters:
        # Append utility of start state to Ustart
        Ustart.append(U[mdp.loc2state[mdp.start]])

        # Create empty transition matrix
        trans_matrix = np.zeros(shape=(mdp.num_states, mdp.num_states))
        # Initialize transition matrix using policy vector pi
        for x in range(mdp.num_states):
            # If state is terminal, automatically passes into the absorbing state
            if(mdp.is_terminal(x)):
                pass
            else:
                transitions = mdp.P_snexts(x, pi[x])
                # set transition probabilities
                for k, v in transitions.items():
                    trans_matrix[x, k] = v

        identity = np.eye(mdp.num_states)
        discounted = gamma * trans_matrix
        A = identity - discounted
        # calculate new utility
        pseudo_inverse = np.linalg.pinv(A)
        U = np.dot(pseudo_inverse, reward_matrix)

        old_pi = pi
        old_U = U
        for state in range(mdp.num_states):
            if(mdp.is_terminal(state)):
                pass
            else:
                utility_dict = {}
                # determine utility for each action available for state
                for x in mdp.A(state):
                    transitions = mdp.P_snexts(state, x)
                    # find the total utility from the transition
                    total = 0
                    for k, v in transitions.items():
                        total += U[k] * v
                    utility_dict[x] = total
                # select the maximum utility
                max_action = max(utility_dict, key=utility_dict.get)
                pi[state] = max_action
        i += 1

    # print(Ustart)
        # print("\nIteration " + str(i) + ": ")
        # print(U)
    # print(U)
    # END IMPLEMENTATION

    # print(Ustart)

    # for 1.2 U* and pi* tables
    # j = 0
    # for x in U:
    #     fixed = '{0:.5f}'.format(x)
    #     print(str(j) + " & " + str(fixed) + " \\\\ \\hline")
    #     j +=1
    # print()
    # for x in pi:
    #     fixed = '{0:.5f}'.format(x)
    #     print(str(fixed) + " \\\\")

    # print()
    # print(new_utility)
    #
    #
    # print(trans_matrix)
    # print_matrix_to_latex(trans_matrix)
    # print_matrix_to_latex(identity)
    # print_matrix_to_latex(A)
    # print_matrix_to_latex(pseudo_inverse)
    # for x in new_utility:
    #     fixed = '{0:.5f}'.format(x)
    #     print(str(fixed) + " \\\\")



    if plot:
        fig = plt.figure()
        plt.title("Policy Iteration with $\gamma={0}$".format(gamma))
        plt.xlabel("Iteration (k)")
        plt.ylabel("Utility of Start")
        plt.ylim(-1, 1)
        plt.plot(Ustart)

        pp = PdfPages('./plots/piplot.pdf')
        pp.savefig(fig)
        plt.close()
        pp.close()

    #U and pi should be returned with the shapes and types specified
    return U, pi, np.array(Ustart)

def print_matrix_to_latex(matrix):
    matrix_latex = "$\\begin{bmatrix} \n"
    for x in matrix:
        for y in x:
            # fixed = '{0:.3f}'.format(y)
            fixed = y
            matrix_latex += str(fixed) + " & "
        matrix_latex += "\\\\ \n"
    matrix_latex += "\end{bmatrix} $"
    print(matrix_latex)

#TODO fix the discount
def td_update(v, s1, r, s2, terminal, alpha, gamma):
    '''
    Performs the TD update on the value function v for one transition (s,a,r,s').
    Update to v should be in place.
    :param v: The value function, a numpy array of length |S|
    :param s1: the current state, an integer 
    :param r: reward for the transition
    :param s2: the next state, an integer
    :param terminal: bool for if the episode ended
    :param alpha: learning rate parameter
    :param gamma: discount factor
    :return: Nothing
    '''
    #TODO implement the TD Update
    #you should update the value function v inplace (does not need to be returned)

    # use the td learning update formula
    if not terminal:
        v[s1] = v[s1] + alpha*(r + gamma*v[s2]-v[s1])
    # TODO how to update at terminal? alpha or do we use gamma for discounting
    else:
        v[s1] += alpha*(r - v[s1])

#TODO fix the discount
def td_episode(env, pi, v, gamma, alpha, max_steps=1000):
    '''
    Agent interacts with the environment for one episode update the value function after
    each iteration. The value function update should be done with the TD learning rule.
    :param env: environment object (GridWorld)
    :param pi: numpy array of length |S| representing the policy
    :param v: numpy array of length |S| representing the value function
    :param gamma: discount factor
    :param alpha: learning rate
    :param max_steps: maximum number of steps in the episode
    :return: two floats G, v0 where G is the discounted return and v0 is the value function of the initial state (before learning)
    '''

    # are the returns the guessed reward function? G is the last return at the end of the episode.
    G = 0.
    v0 = 0.

    #TODO implement the agent interacting with the environment for one episode
    # episode ends when max_steps have been completed
    # episode ends when env is in the absorbing state
    # Learning should be done online (after every step)
    # return the discounted sum of rewards G, and the value function's estimate from the initial state v0
    # the value function estimate should be before any learn takes place in this episode


    i = 0
    # continue the episode until the max_steps have been complete, or if the episode is in absorbing state
    while(i < max_steps and not env.is_absorbing()):
        # get the initial value if this is the first step in the episode
        if(i == 0):
            # get the start state for v0 TODO is this wrong, always producing zero?
            v0 = v[env.get_state()]
        s1 = env.get_state()
        term = env.is_terminal()


        #TODO discount here?
        # take the action specified in policy (do we have to discount it here?)
        r = env.Act(pi[env.get_state()])
        s2 = env.get_state()
        # print("Go from: " + str(s1) + " to: " + str(s2) + " with reward: " + str(r))
        # add to discounted rewards
        G += math.pow(gamma, i)*r
        td_update(v, s1, r, s2, term, alpha, gamma)
        i+=1

    return G, v0

def td_learning(env, pi, gamma, alpha, episodes=200, plot=True):
    '''
    Evaluates the policy pi in the environment by estimating the value function
    with TD updates  
    :param env: environment object (GridWorld)
    :param pi: numpy array of length |S|, representing the policy 
    :param gamma: discount factor
    :param alpha: learning rate
    :param episodes: number of episodes to use in evaluating the policy
    :param plot: boolean for if a plot should be generated for returns and estimates
    :return: Two lists containing the returns for each episode and the value function estimates, also returns the value function
    '''
    returns, estimates = [], []
    v = np.zeros(env.num_states)

    # TODO Implement the td learning for every episode
    # value function should start at 0 for all states
    # return the list of returns, and list of estimates for all episodes
    # also return the value function v

    i = 0
    while i < episodes:
        ret, est = td_episode(env, pi, v, gamma, alpha)
        returns.append(ret)
        estimates.append(est)
        env.reset_to_start()
        i+=1

    if plot:
        fig = plt.figure()
        plt.title("TD Learning with $\gamma={0}$ and $\\alpha={1}$".format(gamma, alpha))
        plt.xlabel("Episode")
        plt.ylabel("Return")
        plt.ylim(-4, 1)
        plt.plot(returns)
        plt.plot(estimates)
        plt.legend(['Returns', 'Estimate'])

        pp = PdfPages('./plots/tdplot.pdf')
        pp.savefig(fig)
        plt.close()
        pp.close()

    # print("returns: " + str(returns))
    # print("\nestimates: " + str(estimates))
    # print("\nv: " + str(v))

    # j=0
    # for x in v:
    #     fixed = '{0:.5f}'.format(x)
    #     print(str(j) + " & " + str(fixed) + " \\\\ \\hline")
    #     j +=1
    # print()

    return returns, estimates, v

def egreedy(q, s, eps):
    '''
    Epsilon greedy action selection for a discrete Q function.
    :param q: numpy array of size |S|X|A| representing the state action value look up table
    :param s: the current state to get an action (an integer)
    :param eps: the epsilon parameter to randomly select an action
    :return: an integer representing the action
    '''
    # TODO implement epsilon greedy action selection

    rand = random.random()
    # take a random action
    if(rand < eps):
        return np.random.randint(0,4)
    # behave optimally
    else:
        action_utilities = {}
        for x in range(4):
            action_utilities[x] = q[s, x]
        max_action = max(action_utilities, key=action_utilities.get)
        return max_action

def q_update(q, s1, a, r, s2, terminal, alpha, gamma):
    '''
    Performs the Q learning update rule for a (s,a,r,s') transition. 
    Updates to the Q values should be done inplace
    :param q: numpy array of size |S|x|A| representing the state action value table
    :param s1: current state
    :param a: action taken
    :param r: reward observed
    :param s2: next state
    :param terminal: bool for if the episode ended
    :param alpha: learning rate
    :param gamma: discount factor
    :return: None
    '''

    # TODO implement Q learning update rule
    # update should be done inplace (not returned)
    # TODO HOW TO DISCOUNT!!!!!!!!!!???????????????????

    if not terminal:
        next_action_utilities = {}
        for x in range(4):
            next_action_utilities[x] = q[s2, x]
        max_action = max(next_action_utilities, key=next_action_utilities.get)
        q[s1, a] = q[s1, a] + alpha*(r + gamma*q[s2, max_action] - q[s1, a])
        # TODO how to update at terminal? alpha or do we use gamma for discounting
    else:
        q[s1, a] += alpha * (r - q[s1, a])


def q_episode(env, q, eps, gamma, alpha, max_steps=1000):
    '''
    Agent interacts with the environment for an episode update the state action value function
    online according to the Q learning update rule. Actions are taken with an epsilon greedy policy
    :param env: environment object (GridWorld)
    :param q: numpy array of size |S|x|A| for state action value function
    :param eps: epsilon greedy parameter
    :param gamma: discount factor
    :param alpha: learning rate
    :param max_steps: maximum number of steps to interact with the environment
    :return: two floats: G, q0 which are the discounted return and the estimate of the return from the initial state
    '''
    G = 0.
    q0 = 0.

    # TODO implement agent interaction for q learning with epsilon greedy action selection
    # Return G the discounted sum of rewards and q0 the estimate of G from the initial state


    i = 0
    # continue the episode until the max_steps have been complete, or if the episode is in absorbing state
    while (i < max_steps and not env.is_absorbing()):
        # get the initial value if this is the first step in the episode
        if (i == 0):
            # get the start state for q and the max action
            action_utilities = {}
            for x in range(4):
                action_utilities[x] = q[env.get_state(), x]
            max_action = max(action_utilities, key=action_utilities.get)
            q0 = q[env.get_state(), max_action]
        s1 = env.get_state()
        term = env.is_terminal()

        #TODO figure out
        # do we update in a or the actual result of our action? can we know what action we took, since it is non-deterministic?
        a = egreedy(q, env.get_state(), eps)

        # TODO discount here?
        # take the action returned by egreedy (optimal or rand) (do we have to discount it here?)
        r = env.Act(a)
        s2 = env.get_state()
        # print("Go from: " + str(s1) + " to: " + str(s2) + " with action: " + str(a) + " with reward: " + str(r))
        # add to discounted rewards
        G += math.pow(gamma, i) * r
        q_update(q, s1, a, r, s2, term, alpha, gamma)
        i += 1

    return G, q0

    ## END TD_EP

def q_learning(env, eps, gamma, alpha, episodes=200, plot=True):
    '''
    Learns a policy by estimating the state action values through interactions 
    with the environment.  
    :param env: environment object (GridWorld)
    :param eps: epsilon greedy action selection parameter
    :param gamma: discount factor
    :param alpha: learning rate
    :param episodes: number of episodes to learn
    :param plot: boolean for if a plot should be generated returns and estimates
    :return: Two lists containing the returns for each episode and the action value function estimates of the return, also returns the Q table
    '''
    returns, estimates = [], []
    q = np.zeros((env.num_states, env.num_actions))

    # TODO implement Q learning over episodes
    # return the returns and estimates for each episode and the Q table


    i = 0
    while i < episodes:
        # print(q)
        ret, est = q_episode(env, q, eps, gamma, alpha)
        returns.append(ret)
        estimates.append(est)
        env.reset_to_start()
        i+=1

    if plot:
        fig = plt.figure()
        plt.title("Q Learning with $\gamma={0}$, $\epsilon={1}$, and $\\alpha={2}$".format(gamma, eps, alpha))
        plt.xlabel("Episode")
        plt.ylabel("Return")
        plt.ylim(-4, 1)
        plt.plot(returns)
        plt.plot(estimates)
        plt.legend(['Returns', 'Estimate'])

        pp = PdfPages('./plots/qplot.pdf')
        pp.savefig(fig)
        plt.close()
        pp.close()

    # print(returns)
    # print(estimates)
    # print(q)


    return returns, estimates, q



if __name__ == '__main__':
    env = GridWorld()
    mdp = GridWorld_MDP()


    U, pi, Ustart = policy_iteration(mdp, plot=True)
    vret, vest, v = td_learning(env, pi, gamma=1., alpha=0.1, episodes=2000, plot=True)
    qret, qest, q = q_learning(env, eps=0.1, gamma=1., alpha=0.1, episodes=20000, plot=True)

    print(qest)
    # print(U)
    # print()
    # print(v)
    # print()
    # print(q)


    # run a few times and take the average:
    # qret, qest, q1, testing1 = q_learning(env, eps=0.1, gamma=1., alpha=0.1, episodes=20000, plot=True)
    # qret, qest, q2, testing2 = q_learning(env, eps=0.1, gamma=1., alpha=0.1, episodes=20000, plot=True)
    # qret, qest, q3, testing3 = q_learning(env, eps=0.1, gamma=1., alpha=0.1, episodes=20000, plot=True)
    # qret, qest, q4, testing4 = q_learning(env, eps=0.1, gamma=1., alpha=0.1, episodes=20000, plot=True)
    #
    # for x in range(11):
    #     sum = testing[x] + testing1[x] + testing2[x] + testing3[x] + testing4[x]
    #     print(str(x) + " & " + '{0:.5f}'.format(sum/5) + " &    \\\\ \\hline")
    # get the policy, check which one is most often
    # avg each number

