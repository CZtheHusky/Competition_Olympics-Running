Current multi-thread setting requires 2 Nvidia A100 GPUs.


## training with our methods

Please note that the device where inference agent lies must be manually set in './algo/ppo.py'.

>python main.py --mix_maps --reward_shaping

## load check point

>python main.py --mix_maps --reward_shaping --check_point --actor_path '' --critic_path '' --ep ep_num --tc train_count




# Self play
Prepare 17 opponents in './self_play_opponents'.
>python self_play.py --actor_path '' --critic_path '' --reward_shaping