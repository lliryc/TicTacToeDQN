from classes.field import Field
from classes.player import RandomPlayer, MinMaxPlayer, DqnPlayer
import numpy as np

size = 3
field = Field(size)

trials = 5000
trial_len = 200

# updateTargetNetwork = 1000
agent1 = DqnPlayer(True, "player1", field)
agent2 = MinMaxPlayer(False, "player2")
player = RandomPlayer(True, "r_player")
i = 0
steps = []
for trial in range(trials):
    print("new match")
    field.reset()
    done = False
    cur_state = field.get_state().reshape(1,size*size)
    for step in range(trial_len):
        try:
            raw_action = agent1.act(cur_state)
        except Exception as ex:
            pass
        action = agent1.translate(raw_action)
        if field.turns == 0:
            action = field.get_free_cell()
        reward = agent1.reward(*action)
        s1 = 0
        try:
            print(agent1.name)
            s1 = agent1.turn(field, *action)
            field.print_all()
        except Exception:
            reward = -1
            print("%s failed" % agent1.name)
            done = True
        if np.abs(s1) == size:
            print("%s wins!" % agent1.name)
            done = True
        if field.full():
            print("Draw!")
            if i == 0:
                reward = 0.5
            else:
                reward = 10 / field.turns
            done = True
        if not done:
            s2 = agent2.turn(field)
            if np.abs(s2) == size:
                print("%s wins!" % agent2.name)
                done = True
                reward = -1.0
            if field.full():
                print("Draw!")
                reward = 0.5
                done = True

        new_state = field.get_state().reshape(1,size*size)

        agent1.remember(cur_state, raw_action, reward, new_state, done)
        try:
            agent1.replay()  # internally iterates default (prediction) model
        except Exception as e1:
            raise e1
        agent1.target_train()  # iterates target model

        cur_state = new_state
        if done:
            break
        i = (i + 1) % 2
    # if step >= trial_len:
    #     print("Failed to complete in trial {}".format(trial))
    #     if step % 10 == 0:
    #         agent1.save_model("trial-{}.model".format(trial))
    # else:
    #     print("Completed in {} trials".format(trial))
agent1.save_model("success1.model")
#agent2.save_model("success2.model")
    #     exit(0)


# player1 = RandomPlayer(True, "player1")
# player2 = MinMaxPlayer(False, "player2")
#
# players = [player1, player2]
# i = 0
# print("Game starts!")
# field.print_all()
#
# while True:
#     player = players[i]
#     i = (i + 1) %  2
#     print(player.name)
#     s = player.turn(field)
#     field.print_all()
#     if np.abs(s) == size:
#         print("%s wins!" % player.name)
#         break
#     elif field.full():
#         print("Draw!")
#         break

