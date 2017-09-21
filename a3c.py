# from env import Env
import env
# from env_test import Env
import card
from card import action_space
import os, random
import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim
import tensorflow.contrib.rnn as rnn
import time
from env_test import get_benchmark
from collections import Counter
import struct
import copy
import random

PASS_PENALTY = 5

##################################################### UTILITIES ########################################################
def counter_subset(list1, list2):
    c1, c2 = Counter(list1), Counter(list2)

    for (k, n) in c1.items():
        if n > c2[k]:
            return False
    return True

# map char cards to 3 - 17
def to_value(cards):
    values = [card.Card.cards.index(c)+3 for c in cards]
    return values

# map 3 - 17 to char cards
def to_char(cards):
    chars = [card.Card.cards[c-3] for c in cards]
    return chars

def get_mask(cards, action_space, last_cards):
    # 1 valid; 0 invalid
    mask = np.zeros_like(action_space)
    for j in range(mask.size):
        if counter_subset(action_space[j], cards):
            mask[j] = 1
    mask = mask.astype(bool)
    if last_cards:
        for j in range(1, mask.size):
            if mask[j] == 1 and not card.CardGroup.to_cardgroup(action_space[j]).\
                    bigger_than(card.CardGroup.to_cardgroup(last_cards)):
                mask[j] = False
    else:
        mask[0] = False
    return mask

def update_params(scope_from, scope_to):
    vars_from = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope_from)
    vars_to = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope_to)

    ops = []
    for from_var, to_var in zip(vars_from, vars_to):
        ops.append(to_var.assign(from_var))
    return ops


def discounted_return(r, gamma):
    r = r.astype(float)
    r_out = np.zeros_like(r)
    val = 0
    for i in reversed(range(r.shape[0])):
        r_out[i] = r[i] + gamma * val
        val = r_out[i]
    return r_out

# Used to initialize weights for policy and value output layers
def normalized_columns_initializer(std=1.0):
    def _initializer(shape, dtype=None, partition_info=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)
    return _initializer


##################################################### UTILITIES ########################################################
class CardNetwork:
    def __init__(self, s_dim, trainer, scope, a_dim=8310):
        with tf.variable_scope(scope):
            card_cnt = 57
            self.temp = tf.placeholder(tf.float32, None, name="boltz")
            self.input = tf.placeholder(tf.float32, [None, s_dim], name="input")

            # embedding layer
            # self.state_onehot = tf.one_hot(self.input, 15, dtype=tf.float32)
            # self.state_onehot = tf.reshape(self.state_onehot, [-1, 15])
            embeddings = slim.fully_connected(
                inputs=self.input, 
                num_outputs=256, 
                activation_fn=None,
                weights_initializer=tf.contrib.layers.xavier_initializer())
            self.embeddings = tf.reshape(embeddings, [-1, 1, 64, 4])

            

            # 1D convolution
            self.conv1 = slim.conv2d(activation_fn=tf.nn.relu, inputs=self.embeddings, num_outputs=16,
                                 kernel_size=[1, 8], stride=[1, 1], padding='SAME')
            self.maxpool1 = slim.max_pool2d(inputs=self.conv1, kernel_size=[1, 4], stride=2, padding='SAME')
            self.conv2 = slim.conv2d(activation_fn=tf.nn.relu, inputs=self.maxpool1, num_outputs=32,
                                 kernel_size=[1, 4], stride=[1, 1], padding='SAME')
            self.maxpool2 = slim.max_pool2d(inputs=self.conv2, kernel_size=[1, 2], stride=2, padding='SAME')
            self.conv3 = slim.conv2d(activation_fn=tf.nn.relu, inputs=self.maxpool2, num_outputs=64,
                                 kernel_size=[1, 2], stride=[1, 1], padding='SAME')
            self.maxpool3 = slim.max_pool2d(inputs=self.conv3, kernel_size=[1, 2], stride=2, padding='SAME')

            # flatten layer
            self.fc_flattened = slim.fully_connected(inputs=slim.flatten(self.maxpool3), num_outputs=256, activation_fn=None)
            self.fc_flattened = slim.fully_connected(inputs=slim.flatten(self.fc_flattened), num_outputs=512, activation_fn=None)

            # self.fc1 = slim.fully_connected(inputs=self.fc_flattened, num_outputs=1024, activation_fn=tf.nn.sigmoid)
            self.fc2 = slim.fully_connected(inputs=self.fc_flattened, num_outputs=8310, activation_fn=tf.nn.elu)

            # value
            self.fc3 = slim.fully_connected(inputs=self.fc_flattened, num_outputs=64, activation_fn=tf.nn.elu)
            self.fc4 = slim.fully_connected(inputs=self.fc3, 
                    num_outputs=1, 
                    activation_fn=None,
                    weights_initializer=normalized_columns_initializer(1.0))

            self.policy_pred = tf.reshape(self.fc2, [1, -1])

            self.mask = tf.placeholder(tf.bool, [None, a_dim], name='mask')
            self.mask = tf.reshape(self.mask, [1, -1])
            self.valid_policy = tf.boolean_mask(self.policy_pred[0], self.mask[0])
            self.policy_norm = tf.norm(self.valid_policy)
            self.a0 = self.valid_policy[0]

            self.boltz_policy = tf.reshape(tf.nn.softmax(self.valid_policy / self.temp), [1, -1])
            self.valid_policy = tf.nn.softmax(self.valid_policy)
            self.valid_policy = tf.reshape(self.valid_policy, [1, -1])

            self.val_pred = tf.reshape(self.fc4, [-1])

            # only support batch size one since masked_a_dim is changing
            self.action = tf.placeholder(tf.int32, [None], "action_input")
            
            self.masked_a_dim = tf.placeholder(tf.int32, None)
            self.action_one_hot = tf.one_hot(self.action, self.masked_a_dim, dtype=tf.float32)

            self.val_truth = tf.placeholder(tf.float32, [None], "val_input")
            self.advantages = tf.placeholder(tf.float32, [None], "advantage_input")

            self.pi_sample = tf.reduce_sum(tf.multiply(self.action_one_hot[0], self.valid_policy[0]))
            self.pi = tf.cond(self.pi_sample > 0.99, lambda : self.pi_sample - 0.01, lambda : self.pi_sample)
            self.pred_prob = self.pi
            self.policy_loss = -tf.reduce_sum(tf.log(tf.clip_by_value(self.pi, 1e-8, 1.)) * self.advantages)

            self.val_loss = tf.reduce_sum(tf.square(self.val_pred-self.val_truth))

            self.loss = 0.2 * self.val_loss + self.policy_loss

            local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
            self.gradients = tf.gradients(self.loss, local_vars)

            self.var_norms = tf.global_norm(local_vars)
            self.gradients, self.grad_norms = tf.clip_by_global_norm(self.gradients, 1.0)
            self.apply_grads = tf.cond(self.masked_a_dim > 1, lambda : trainer.apply_gradients(zip(self.gradients, local_vars)), 
                lambda: tf.identity(tf.constant(False))) 

class CardAgent:
    def __init__(self, name, trainer):
        self.name = name
        self.episodes = tf.Variable(0, dtype=tf.int32, name='episodes_' + name, trainable=False)
        self.increment = self.episodes.assign_add(1)
        self.network = CardNetwork(54 * 6, trainer, self.name, 8310)

    def train_batch_packed(self, buffer, masks, sess, gamma, val_last):
        states = buffer[:, 0]
        actions = buffer[:, 1]
        rewards = buffer[:, 2]
        values = buffer[:, 3]
        a_dims = buffer[:, 4]

        rewards_plus = np.append(rewards, val_last)
        val_truth = discounted_return(rewards_plus, gamma)[:-1]

        val_pred_plus = np.append(values, val_last)
        td0 = rewards + gamma * val_pred_plus[1:] - val_pred_plus[:-1]
        advantages = discounted_return(td0, gamma)

        for i in range(buffer.shape[0]):
            s = states[i]
            a = actions[i]
            r = rewards[i]
            v = values[i]
            a_dim = a_dims[i]
            v_truth = val_truth[i]
            advantage = advantages[i]
            buff = np.array([[s, a, r, v, a_dim]])
            m = masks[i:i+1]
            p_norm, a0, pred_prob, loss, policy_loss, val_loss, var_norms, grad_norms = \
                self.train_batch(buff, m, sess, gamma, [v_truth], [advantage])
        
        return p_norm / buffer.shape[0], a0 / buffer.shape[0], pred_prob / buffer.shape[0], loss / buffer.shape[0], \
            policy_loss / buffer.shape[0], val_loss / buffer.shape[0], var_norms / buffer.shape[0], grad_norms / buffer.shape[0]

    def train_batch(self, buffer, masks, sess, gamma, val_truth, advantages):
        states = buffer[:, 0]
        actions = buffer[:, 1]
        rewards = buffer[:, 2]
        values = buffer[:, 3]
        a_dims = buffer[0, 4]

        # rewards_plus = np.append(rewards, val_last)
        # val_truth = discounted_return(rewards_plus, gamma)[:-1]

        # print(val_truth)
        # input('continue')

        # val_pred_plus = np.append(values, val_last)
        # td0 = rewards + gamma * val_pred_plus[1:] - val_pred_plus[:-1]
        # advantages = discounted_return(td0, gamma)
        # advantages = val_truth

        _, p_norm, a0, action_one_hot, valid_policy, pred_prob, loss, policy_loss, val_loss, var_norms, grad_norms = sess.run([self.network.apply_grads,
            self.network.policy_norm,
            self.network.a0,
            self.network.action_one_hot,
            self.network.valid_policy,
            self.network.pred_prob,
            self.network.loss, 
            self.network.policy_loss, 
            self.network.val_loss,
            self.network.var_norms,
            self.network.grad_norms], 
            feed_dict={self.network.val_truth: val_truth,
                        self.network.advantages: advantages,
                        self.network.input: np.vstack(states),
                        self.network.action: actions,
                        self.network.masked_a_dim: a_dims,
                        self.network.mask: masks})
        # print("policy_norm:", p_norm, "a0", a0)
        # print(action_one_hot)
        # print(valid_policy)
        # print(val_truth)
        # print(val_loss)
        # input('continue')
        episode = sess.run(self.episodes)
        return p_norm, a0, pred_prob, loss, policy_loss, val_loss, var_norms, grad_norms
        # if episode % 100 == 0:
        #     print("loss : %f" % loss)

class CardMaster:
    def __init__(self, env):
        self.temp = 1
        self.start_temp = 1
        self.end_temp = 0.2
        self.action_space = card.get_action_space()
        self.name = 'global'
        self.env = env
        self.a_dim = 8310
        self.gamma = 0.99
        self.sess = None

        self.train_intervals = 30

        self.trainer = tf.train.AdamOptimizer(learning_rate=0.001)
        self.episode_rewards = [[] for i in range(2)]
        self.episode_length = [[] for i in range(2)]
        self.episode_mean_values = [[] for i in range(2)]
        self.summary_writers = [tf.summary.FileWriter("train_agent%d" % i) for i in range(2)]

        self.agents = [CardAgent('agent%d' % i, self.trainer) for i in range(2)]

        self.global_episodes = tf.Variable(0, dtype=tf.int32, name='global_episodes', trainable=False)
        self.increment = self.global_episodes.assign_add(1)

    def train_batch(self, buffer, masks, sess, gamma, val_last, idx):
        buffer = np.array(buffer)
        masks = np.array(masks)
        return self.agents[idx].train_batch(buffer, masks, sess, gamma, val_last)

    def train_batch_packed(self, buffer, masks, sess, gamma, val_last, idx):
        buffer = np.array(buffer)
        masks = np.array(masks)
        return self.agents[idx].train_batch_packed(buffer, masks, sess, gamma, val_last)

    def respond(self, env):
        mask = get_mask(to_char(self.env.get_curr_cards()), self.action_space, to_char(self.env.get_last_cards()))
        s = env.get_state()
        s = np.reshape(s, [1, -1])
        policy, val = self.sess.run([
            self.agents[0].network.valid_policy,
            self.agents[0].network.val_pred],
            feed_dict={
                self.agents[0].network.input: s,
                self.agents[0].network.mask: np.reshape(mask, [1, -1])
            })
        policy = policy[0]
        valid_actions = np.take(np.arange(self.a_dim), mask.nonzero())
        valid_actions = valid_actions.reshape(-1)
        # a = np.random.choice(valid_actions, p=policy)
        a = valid_actions[np.argmax(policy)]
        # print("taking action: ", self.action_space[a])
        return env.step(self.action_space[a])

    # train two farmers simultaneously
    def run(self, sess, saver, max_episode_length, cards):
        self.sess = sess
        with sess.as_default():
            global_episodes = sess.run(self.global_episodes)
            total_episodes = 10001
            temp_decay = (self.end_temp - self.start_temp) / total_episodes
            while global_episodes < total_episodes:
                print("episode %d" % global_episodes)
                episode_buffer = [[] for i in range(2)]
                episode_mask = [[] for i in range(2)] 
                episode_values = [[] for i in range(2)]
                episode_reward = [0, 0]
                episode_steps = [0, 0]
                need_train = []

                self.env.reset()
                self.env.prepare()
                self.env.step(lord=True)

                # print("training id %d" % train_id)
                s = self.env.get_state()
                s = np.reshape(s, [1, -1])
                # rnn_state = [[c_init, h_init], [c_init.copy(), h_init.copy()]]
                # # shallow copy
                # rnn_state_backup = rnn_state.copy()
                for l in range(max_episode_length):
                    # time.sleep(1)
                    # # map 1, 3 to 0, 1
                    train_id = self.env.get_role_ID()
                    train_id = int((train_id - 1) / 2)

                    print("turn %d" % l)
                    print("training id %d" % train_id)
                    
                    mask = get_mask(to_char(self.env.get_curr_cards()), self.action_space, to_char(self.env.get_last_cards()))
                    # oppo_cnt = self.env.get_opponent_min_cnt()
                    # encourage response when opponent is about to win
                    # if random.random() > oppo_cnt / 20. and np.count_nonzero(mask) > 1:
                    #     mask[0] = False
                    policy, val = sess.run([
                        self.agents[train_id].network.boltz_policy,
                        self.agents[train_id].network.val_pred],
                        feed_dict={
                            self.agents[train_id].network.temp : self.temp,
                            self.agents[train_id].network.input: s,
                            self.agents[train_id].network.mask: np.reshape(mask, [1, -1])
                        })
                    self.temp -= temp_decay
                    policy = policy[0]
                    valid_actions = np.take(np.arange(self.a_dim), mask.nonzero())
                    valid_actions = valid_actions.reshape(-1)
                    a = np.random.choice(valid_actions, p=policy)

                    
                    a_masked = np.where(valid_actions == a)[0]


                    # print("taking action: ", self.action_space[a])
                    r, done = self.env.step(cards=to_value(self.action_space[a]))
                    s_prime = self.env.get_state()
                    s_prime = np.reshape(s_prime, [1, -1])

                    episode_buffer[train_id].append([s, a_masked, r, val[0], np.sum(mask.astype(np.float32))])
                    episode_mask[train_id].append(mask)
                    episode_values[train_id].append(val)
                    episode_reward[train_id] += r
                    episode_steps[train_id] += 1

                    if done:
                        for i in range(2):
                            if len(episode_buffer[i]) != 0:
                                p_norm, a0, pred_prob, loss, policy_loss, val_loss, var_norms, grad_norms = self.train_batch_packed(episode_buffer[i], episode_mask[i], sess, self.gamma, 0, i)
                        break
                        

                    s = s_prime

                    if len(episode_buffer[train_id]) == self.train_intervals:
                        val_last = sess.run(self.agents[train_id].network.val_pred,
                                            feed_dict={self.agents[train_id].network.input: s})
                        # print(val_last[0])
                        p_norm, a0, pred_prob, loss, policy_loss, val_loss, var_norms, grad_norms = self.train_batch_packed(episode_buffer[train_id], episode_mask[train_id], sess, self.gamma, val_last[0], train_id)
                        episode_buffer[train_id] = []
                        episode_mask[train_id] = []

                for i in range(2):
                    self.episode_mean_values[i].append(np.mean(episode_values[i]))
                    self.episode_length[i].append(episode_steps[i])
                    self.episode_rewards[i].append(episode_reward[i])

                    episodes = sess.run(self.agents[i].episodes)
                    sess.run(self.agents[i].increment)

                    update_rate = 5
                    if episodes % update_rate == 0 and episodes > 0:
                        mean_reward = np.mean(self.episode_rewards[i][-update_rate:])
                        mean_length = np.mean(self.episode_length[i][-update_rate:])
                        mean_value = np.mean(self.episode_mean_values[i][-update_rate:])

                        summary = tf.Summary()
                        summary.value.add(tag='Performance/rewards', simple_value=float(mean_reward))
                        summary.value.add(tag='Performance/length', simple_value=float(mean_length))
                        summary.value.add(tag='Performance/values', simple_value=float(mean_value))
                        summary.value.add(tag='Losses/Value Loss', simple_value=float(val_loss))
                        summary.value.add(tag='Losses/Prob pred', simple_value=float(pred_prob))
                        summary.value.add(tag='Losses/Policy Loss', simple_value=float(policy_loss))
                        summary.value.add(tag='Losses/Grad Norm', simple_value=float(grad_norms))
                        summary.value.add(tag='Losses/Var Norm', simple_value=float(var_norms))
                        summary.value.add(tag='Losses/Policy Norm', simple_value=float(p_norm))
                        summary.value.add(tag='Losses/a0', simple_value=float(a0))

                        self.summary_writers[i].add_summary(summary, episodes)
                        self.summary_writers[i].flush()

                global_episodes += 1
                sess.run(self.increment)
                # if global_episodes % 50 == 0:
                #     saver.save(sess, './model' + '/model-' + str(global_episodes) + '.cptk')
                #     print("Saved Model")

                # self.env.end()


def run_game(sess, network):
    max_episode_length = 100
    lord_win_rate = 0
    for i in range(100):
        network.env.reset()
        network.env.players[0].trainable = True
        lord_idx = 2
        network.env.players[2].is_human = True
        network.env.prepare(lord_idx)

        s = network.env.get_state(0)
        s = np.reshape(s, [1, -1])

        while True:
            policy, val = sess.run([network.agent.network.policy_pred, network.agent.network.val_pred],
                                   feed_dict={network.agent.network.input: s})
            mask = network.env.get_mask(0)
            valid_actions = np.take(np.arange(network.a_dim), mask.nonzero())
            valid_actions = valid_actions.reshape(-1)
            valid_p = np.take(policy[0], mask.nonzero())
            if np.count_nonzero(valid_p) == 0:
                valid_p = np.ones([valid_p.size]) / float(valid_p.size)
            else:
                valid_p = valid_p / np.sum(valid_p)
            valid_p = valid_p.reshape(-1)
            a = np.random.choice(valid_actions, p=valid_p)

            r, done = network.env.step(0, a)
            s_prime = network.env.get_state(0)
            s_prime = np.reshape(s_prime, [1, -1])

            if done:
                idx = network.env.check_winner()
                if idx == lord_idx:
                    lord_win_rate += 1
                print("winner is player %d" % idx)
                print("..............................")
                break
            s = s_prime
    print("lord winning rate: %f" % (lord_win_rate / 100.0))





class DemoGame:
    """docstring for ClassName"""

    def __init__(self, _handcards, _extracards):
        self.handcards = copy.deepcopy(_handcards)
        self.extracards = copy.deepcopy(_extracards)
        self.actions = []
        self.reward = 0
        self.lordID = -1

    def add_action(self, a):
        self.actions.append(a)

    def set_reward(self, r):
        self.reward = r

    def set_lordID(self, id):
        self.lordID = id


def read_seq3(filename):
    episodes = 0
    f = open(filename, 'rb')

    eof = False
    while True:
        cards = []
        for i in range(54):
            b = f.read(1)
            if not b:
                eof = True
                break
            c = str((struct.unpack('c', b)[0]).decode('ascii'))
            
            if c == '1':
                c = '10'
            cards.append(c)
        
        if eof:
            break
        lord_id = struct.unpack('H', f.read(2))[0]
        # print(cards)

        handcards = [cards[:17], cards[17:34], cards[34:51]]
        extra_cards = cards[51:]
        handcards[lord_id] += extra_cards

        demo = DemoGame(handcards, extra_cards)
        demo.set_lordID(lord_id)

        r = 0
        ind = lord_id
        while r == 0:
            a = struct.unpack('H', f.read(2))[0]
            demo.add_action(a)
            #print("a = ", a)
            put_list = action_space[a]
            # print(put_list)
            for c in put_list:
                handcards[ind].remove(c)
            ind = int(ind + 1) % 3
            if (not handcards[0]) or (not handcards[1]) or (not handcards[2]):
                r = struct.unpack('h', f.read(2))[0]
                demo.set_reward(r)

        demoGames.append(demo)

        # print(episodes)
        episodes += 1
        if episodes == 248491:
            break

def to_one_hot(v):
    result = [0 for i in range(54)]

    for color in v:
        if color > 52:
            color = 53
        result[color] += 1;

    return result

def subtract(a, b):
    c = []
    for i in range(len(a)):
        c.append(a[i] - b[i])
    return c

def to_color(ch):
    ret = card.Card.cards.index(ch) * 4
    if ret > 52:
        ret = 53
    return ret

#e.g. [['3', '3', '4', '6'], ['*', '$']] to [[0, 1, 4, 12], [52, 53]]
def to_color_handcards(ch):
    handcards = [[] for i in range(3)]
    mask = [False for i in range(54)]
    for i in range(3):
        for c in ch[i]:
            color = to_color(c)
            while mask[color]:
                color += 1
            mask[color] = True
            handcards[i].append(color)
    return handcards

def to_card(color):
    idx = 14 if color == 53 else int(color / 4)
    return card.Card.cards[idx]

def to_color_extracards(handcards, ch):
    extracards = []
    mask = [False for i in range(54)]
    for i in range(3):
        for color in handcards:
            if to_card(color) == ch[i] and (not mask[color]):
                mask[color] = True
                extracards.append(color)
                break
    return extracards

def to_color_putlist(action, handcards):
    putlist = []
    mask = [False for i in range(54)]
    for a in action:
        for color in handcards:
            if to_card(color) == a and (not mask[color]):
                mask[color] = True
                putlist.append(color)
                break
    return putlist

def printf(l):
    s = ""
    for e in l:
        if e == 0:
            s = s + "0 "
        else:
            s = s + "1 "
    s = s + '\n'
    f.write(s)


def collect_data():
    cnt = 0

    action_space = card.get_action_space()

    # print(action_space)
    # print("a : ", len(action_space))

    while cnt < N:
        gameID = random.randint(0, N - 1)
        demoGame = demoGames[gameID]

        lordID = demoGame.lordID
        gameLen = len(demoGame.actions)

        # while True:
        th = random.randint(0, gameLen - 1)
            # if not(th % 3 == lordID):
            #     break

        # if actions[th] == 0:
        #     continue
        handcards = to_color_handcards(demoGame.handcards)
        # print("demo handcards", demoGame.handcards)
        # print("hand ", handcards)
        # print("lord ", lordID)
        extracards = to_color_extracards(handcards[lordID], demoGame.extracards)
        # print("demo extracards", demoGame.extracards)
        # print("hand ", extracards)
        # print("demo actions", demoGame.actions)
        acts = [action_space[a] for a in demoGame.actions]
        # print("demo actions", acts)
        # print("th = ", th)

        outCardList = [[] for i in range(3)]

        ind = lordID
        last_cards = []
        last_ID = lordID
        for i in range(th):
            put_list = to_color_putlist(action_space[demoGame.actions[i]], handcards[ind])
            if not(put_list == []):
                last_cards = copy.deepcopy(put_list)
                last_ID = ind
            outCardList[ind] += put_list
            # print(put_list)
            for c in put_list:
                handcards[ind].remove(c)
            ind = int(ind + 1) % 3

        if (last_ID == (th + lordID) % 3):
            last_cards = []
        # print("last cards ", last_cards, [to_card(x) for x in last_cards])

        state = []
        total = [1 for i in range(54)]

        # print(handcards[th % 3])
        self_cards = to_one_hot(handcards[(th + lordID) % 3])
        remains = subtract(total, self_cards);

        history = [to_one_hot(outCardList[i]) for i in range(3)]
        # for i in range(3):
        #     print("out card ", i, outCardList[i], [to_card(x) for x in outCardList[i]])
        # print("history ", history)

        for i in range(3):
            remains = subtract(remains, history[i])

        extra_cards = to_one_hot(extracards)

        state += self_cards;
        state += remains;
        state += history[0];
        state += history[1];
        state += history[2];
        state += extra_cards;

        # numOfFeasibleActs = 0
        # print("feasible actions : ")
        action = []
        mask = get_mask([to_card(color) for color in handcards[(th + lordID) % 3]], action_space, [to_card(color) for color in last_cards])
        for a in range(len(action_space)):
            if mask[a]:
                if a == demoGame.actions[th]:
                    action.append(1)
                else:
                    action.append(0)
        #         print(action_space[a])
        #         numOfFeasibleActs += 1
        # print("num of fea : ", numOfFeasibleActs)


        # X.append(state)
        # Y.append(action)

        printf(state)
        printf(action)

        # print("state ", cnt, " ", [(i % 54, state[i]) for i in range(54 * 6)])
        # print("action ", cnt, " ", action, len(action))
        return

        cnt += 1

#size of action space : 9085

if __name__ == '__main__':
    demoGames = []
    read_seq3("seq")
    numOfDemos = len(demoGames)

    # print("hc : ", demoGames[1].handcards)

    N = 200000

    X = []
    Y = []
    f = open("data", "w")
    collect_data()
    f.close()

    # for i in range(200):
    #     print(demoGames[i].lordID)

    #collect_data()

    # print(len(demoGames))
    # print(demoGames[3333].handcards)
    # print(demoGames[3333].actions)
    # print(demoGames[3333].reward)
    # trainer = tf.train.AdamOptimizer()
    # network = CardNetwork(54 * 6, trainer, "test", 8310)
    # summary_writer = tf.summary.FileWriter("agent_test")

    # test_truth = np.zeros([1, 8310])
    # test_truth[0, 300:310] = 0.1
    # mask = np.zeros([1, 8310]).astype(np.bool)
    # mask[0, 300:310] = True
    # test_input = np.random.randint(15, size=(1, 57))
    # # c_init = np.zeros((1, network.lstm.state_size.c), np.float32)
    # # h_init = np.zeros((1, network.lstm.state_size.h), np.float32)
    # with tf.Session() as sess:
    #     sess.run(tf.global_variables_initializer())
    #     losses = []
    #     for i in range(10001):
    #         if i % 1000 == 0:
    #             print('running % d epoches' % i)
    #         _, loss = sess.run([network.apply_grads, network.loss], feed_dict={
    #             network.test_truth: test_truth,
    #             network.mask : mask,
    #             network.input: test_input
    #             })
    #         if i == 0:
    #             print("origin loss: %f" % loss)
    #         if i == 10000:
    #             print("final loss: %f" % loss)
    #         losses.append(loss)
    #         if i % 10 == 0:
    #             summary = tf.Summary()
    #             summary.value.add(tag='loss', simple_value=float(np.mean(np.array(losses))))
    #             summary_writer.add_summary(summary, i)
    #             summary_writer.flush()
    #             losses = []

    '''
    load_model = False
    model_path = './model'
    cardgame = env.Env()
    master = CardMaster(cardgame)
    saver = tf.train.Saver(max_to_keep=20)
    with tf.Session() as sess:
        if load_model:
            print('Loading Model...')
            ckpt = tf.train.get_checkpoint_state(model_path)
            saver.restore(sess, ckpt.model_checkpoint_path)
            master.run(sess, saver, 2000, cards)
            # run_game(sess, master)
        else:
            sess.run(tf.global_variables_initializer())
            master.run(sess, saver, 2000, cards)
        print(get_benchmark(cards, master))
        sess.close()
    '''
