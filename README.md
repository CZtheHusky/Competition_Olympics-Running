## AI3617 GTMAL Final Project

### Have a look at the skilled huskies!

https://www.bilibili.com/video/BV1TY4y1x7u1

https://www.bilibili.com/video/BV1rF411G72L


### How to train your own husky

Current multithread settings are tested with two A100 GPUs with 40GB of memory each.

> cd rl_trainer_rnn_mt
> 
> python main.py --mix_maps --reward_shaping

### Resume training from a check point

> cd rl_trainer_rnn_mt
> 
> python main.py --mix_maps --reward_shaping --check_point --actor_path '' --critic_path '' --ep ep_num --tc train_count

### Already got a husky? Time to self play!
Don't forget to prepare 17 opponents in './self_play_opponents' first. Feel free to choose them from your stored huskies.

> cd rl_trainer_rnn_mt
> 
> python self_play.py --actor_path '' --critic_path '' --reward_shaping

### Want to enjoy the races on your computer?
> cd evaluation

> python eval.py
