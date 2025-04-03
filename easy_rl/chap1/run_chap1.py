from chap1.utils import random_agent, print_all_envs, do_play_mountain_car

flag_random_agent = False
flag_print_all_envs = False
flag_play_mountain_car = True
def run_chap1_example():
    if flag_random_agent:
        random_agent()
    if flag_print_all_envs:
        print_all_envs()
    if flag_play_mountain_car:
        do_play_mountain_car()
    