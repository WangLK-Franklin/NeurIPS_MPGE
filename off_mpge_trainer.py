#  Copyright (c). All Rights Reserved.
#  General Optimal control Problem Solver (GOPS)
#  Intelligent Driving Lab (iDLab), Tsinghua University
#
#  Creator: iDLab
#  Lab Leader: Prof. Shengbo Eben Li
#  Email: lisb04@gmail.com
#
#  Description: Serial trainer for off-policy RL algorithms
#  Update Date: 2021-05-21, Shengbo LI: Format Revise
#  Update Date: 2022-04-14, Jiaxin Gao: decrease parameters copy times
#  Update: 2022-12-05, Wenhan Cao: add annotation

__all__ = ["OffSerialTrainer"]

from cmath import inf
import os
import time
import numpy as np
import gin
import ray
import torch
from tqdm.auto import tqdm
import copy
from torch.utils.tensorboard import SummaryWriter
from easydict import EasyDict as edict
import random
from gops.utils.common_utils import ModuleOnDevice
from gops.utils.parallel_task_manager import TaskPool
from gops.utils.tensorboard_setup import add_scalars, tb_tags
from gops.utils.log_data import LogData
from gops.trainer.buffer.replay_buffer import ReplayBuffer

from einops import rearrange
from gops.trainer.world_model.weser import world_models_diffusion
from gops.trainer.buffer.world_buffer import WorldBuffer
from gops.trainer.sampler.world_sampler import WorldSampler
from gops.trainer.world_model.weser.value_diffusion import DiffusionModel
from gops.trainer.world_model.weser.discriminator import State_discriminator
from gops.trainer.world_model.weser.discriminator import Dynamics_discriminator
import gc

def move_to_device(tensor, device):
    return tensor if tensor.device == device else tensor.to(device)
    
def seed_np_torch(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # some cudnn methods can be random even after fixing the seed unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False


class OffSerialDuelTrainer:
    
    def __init__(self, alg, sampler, buffer:ReplayBuffer, evaluator, **kwargs):
        self.alg = alg
        self.sampler = sampler
        self.buffer = buffer
        self.pretrain_buffer = copy.deepcopy(buffer)
        self.per_flag = kwargs["buffer_name"] == "prioritized_replay_buffer"
        self.device = torch.device(kwargs["device"] if torch.cuda.is_available() else "cpu")
        print(f'{self.device} is used.')
        self.evaluator = evaluator
        self.kwargs = kwargs    
        self.seed = kwargs["seed"]
        
        # create center network
        self.networks = self.alg.networks
        self.sampler.networks = self.networks
        self.default_sample = True
        seed_np_torch(self.seed)

        if kwargs["ini_network_dir"] is not None:
            self.networks.load_state_dict(torch.load(kwargs["ini_network_dir"]))
            
        self.pretrain_save_dir = kwargs.get("pretrain_dir", f"/root/DARC/data/pretrain_weights")
        os.makedirs(self.pretrain_save_dir, exist_ok=True)
        
        self.training_interval = kwargs.get("training_interval", 10)
        self.replay_batch_size = kwargs["replay_batch_size"]
        self.batch_length = kwargs["batch_length"]
        self.max_iteration = kwargs["max_iteration"]
        self.sample_interval = kwargs.get("sample_interval", 1)
        self.log_save_interval = kwargs["log_save_interval"]
        self.apprfunc_save_interval = kwargs["apprfunc_save_interval"]
        self.eval_interval = kwargs["eval_interval"]
        self.best_tar = -inf
        self.save_folder = kwargs["save_folder"]
        self.iteration = 0
        self.sample_ratio = 0.5
        self.replay_dict={}
        self.min_ratio = kwargs.get("min_replay_ratio", 0.5)
        
        self.pim_flag = kwargs.get("pim_flag", False)
        # init settngs
        if kwargs["name"] == "debug":
            self.replay_start = 2
            self.replay_interval = 1
            self.pretrain_step = 1
            self.replay_duration=2
            self.buffer_warm_size=5000
        else:
            self.replay_start = kwargs.get("replay_start", 100000)
            self.replay_interval = kwargs.get("replay_interval", 10000)
            self.pretrain_step = kwargs.get("pretrain_step", 200000)
            self.replay_duration = kwargs.get("replay_duration", 300000)
            self.buffer_warm_size=kwargs["buffer_warm_size"]
        self.writer = SummaryWriter(log_dir=self.save_folder, flush_secs=20)
       
        add_scalars(
            {tb_tags["alg_time"]: 0, tb_tags["sampler_time"]: 0}, self.writer, 0
        )
        self.writer.flush()


        self.world_model = world_models_diffusion.WorldModel(
                state_dim=kwargs.get("obsv_dim", None),
                action_dim=kwargs.get("action_dim", None),
                latent_dim=self.kwargs.get("latent_dim", 256),
                transformer_max_length=self.kwargs.get("transformer_max_length", 64),
                transformer_hidden_dim=self.kwargs.get("transformer_hidden_dim", 512),
                transformer_num_layers=self.kwargs.get("transformer_num_layers", 2),
                transformer_num_heads=self.kwargs.get("transformer_num_heads", 8),
            ).to(self.device)
        
        self.generator_model = DiffusionModel(
            state_dim=self.world_model.emb_dim,guidance_scale=kwargs.get("generator_scale",1.0)).to(self.device)
        self.state_discriminator = State_discriminator(kwargs.get("obsv_dim", None))
        self.state_discriminator.to(self.device)
        self.dynamics_discriminator = Dynamics_discriminator(kwargs.get("obsv_dim", None), kwargs.get("action_dim", None))
        self.dynamics_discriminator.to(self.device)

        self.use_gpu = kwargs["use_gpu"]
        if self.use_gpu:
            self.networks.to(self.device)
            
        #warm buffer
        # while not self.world_replayer.ready():
        #     # with ModuleOnDevice(self.networks, device="cpu"):
        #     sampler_samples, sampler_tb_dict = self.sampler.sample()
        #     samples = experience_to_dict(sampler_samples)
        #     self.world_replayer.append(samples)
        # self.sampler_tb_dict = LogData()
        while self.buffer.size < self.buffer_warm_size:
            self.sampler.mode = "train"
            samples, _ = self.sampler.sample()
            self.buffer.add_batch(samples)
        self.sampler_tb_dict = LogData()
                              
        #pretrain world model
        if not kwargs["pretrain_flag"]:
            world_model_path = os.path.join(kwargs["pretrain_dir"], kwargs["pretrain_model"])
            
            # generator_model_path = os.path.join(kwargs["pretrain_dir"], "generator_model_dmac_humanoid_pretrain.pth")
            self.world_model.load_state_dict(torch.load(world_model_path))
            # self.generator_model.load_state_dict(torch.load(generator_model_path))
            print(f"Pretrain weights loaded from {world_model_path}")
        else:
            # world_model_path = os.path.join(kwargs["pretrain_dir"], "world_model_dogrun_embent.pth")
            
            # # generator_model_path = os.path.join(kwargs["pretrain_dir"], "generator_model_dmac_humanoid_pretrain.pth")
            # self.world_model.load_state_dict(torch.load(world_model_path))
            while self.pretrain_buffer.size < kwargs["buffer_warm_size"]:
                self.sampler.mode = "pretrain"
                samples, _ = self.sampler.sample()
                self.pretrain_buffer.add_batch(samples)
            pbar = tqdm(range(self.pretrain_step), desc='Pre-training World Model')
            
            for i in pbar:
                self.sampler.mode = "pretrain"
                pretrain_samples, _ = self.sampler.sample()
                self.pretrain_buffer.add_batch(pretrain_samples)
                self.world_model.train()
                with torch.set_grad_enabled(True):
                    replayed_data = self.pretrain_buffer.replay(batch_size=self.replay_batch_size , batch_length=self.batch_length)
                    # replays = self.world_replayer.replay(self.replay_batch_size, self.batch_length)
                    world_model_tb_dict = self.world_model.update(obs=replayed_data.obs.to(self.device), 
                                            action=replayed_data.action.to(self.device), 
                                            reward=replayed_data.reward.to(self.device),
                                            next_obs = replayed_data.obs2.to(self.device),
                                            termination=replayed_data.termination.to(self.device)
                                            )
                    dynamic_loss = world_model_tb_dict["World_model/total_loss"]
                    reward_loss = world_model_tb_dict["World_model/reward_loss"]
                    state = rearrange(replayed_data.obs, 'b t d -> (b t) d')
                    
                    
                    # self.generator_model.train()
                    # gen_td = self.generator_model.train_step(state.to(self.device))
                    # gen_td_loss = gen_td["Generator/diffusion_loss"]

                    pbar.set_postfix({
                        'dynamic_loss': f'{dynamic_loss:.4f}',
                        'reward_loss': f'{reward_loss:.4f}',
                        # 'gen_td_loss': f'{gen_td_loss:.4f}'
                    })
            self.save_pretrain_weights()
        self.sampler.mode = "train"        
        self.evluate_tasks = TaskPool()
        self.last_eval_iteration = 0

        self.start_time = time.time()
        self.world_model = torch.compile(self.world_model)
        self.generator_model = torch.compile(self.generator_model)

        
    def save_pretrain_weights(self):
    # 获取名称
        model_name = self.kwargs["name"]
        
        # 保存world model权重
        world_model_path = os.path.join(self.pretrain_save_dir, f"world_model_{model_name}.pth")
        torch.save(
            self.world_model.state_dict(),
            world_model_path
        )
        
        # 保存generator model权重
        # generator_path = os.path.join(self.pretrain_save_dir, f"generator_model_{model_name}.pth")
        # torch.save(
        #     self.generator_model.state_dict(),
        #     generator_path
        # )
        # print(f"Pretrain weights saved to {self.pretrain_save_dir}")
        
    def get_imagine_samples_wodiffusion(self, imagine_batch_size,gen_state, gen_act, is_train=True):

        imagine_obs,imagine_action,imagine_reward,imagine_termination = self.world_model.imagine_data(
            agent=self.networks,
            sample_obs=gen_state,
            sample_action=gen_act,
            imagine_batch_size=imagine_batch_size,
            imagine_batch_length=self.batch_length
        )
        sample_index = int(self.sample_ratio * imagine_batch_size*self.batch_length/2)
        self.world_model.eval()
        imagine_samples = {}
        imagine_obs2 = imagine_obs[:,1]
        obs = torch.cat([imagine_obs[:, i] for i in range(self.batch_length)], dim=0)
        action = torch.cat([imagine_action[:, i] for i in range(self.batch_length)], dim=0)
        next_obs = torch.cat([imagine_obs[:, i+1] for i in range(self.batch_length)], dim=0)
        reward = torch.cat([imagine_reward[:,i] for i in range(self.batch_length)], dim=0)
        termination = torch.cat([imagine_termination[:, i] for i in range(self.batch_length)], dim=0)
        imagine_samples = {
            "obs": obs[:imagine_batch_size].to(self.device),
            "act": action[:imagine_batch_size].to(self.device),
            "rew": reward[:imagine_batch_size].to(self.device),
            "done": termination[:imagine_batch_size].to(self.device),
            "obs2": next_obs[:imagine_batch_size].to(self.device),
            # "type": "imagine",
            "weight": torch.ones(obs.shape[0])[:imagine_batch_size].to(self.device)
        }

        return imagine_samples,imagine_obs2
    
    def get_imagine_samples_pev(self, imagine_batch_size, target_zone, is_train=True):
        self.generator_model.mode = "td"
        gen_latent = self.generator_model.guided_sample_batch(
                batch_size=imagine_batch_size, 
                world_model=self.world_model, 
                policy_net=self.networks,
                # target_zone=target_zone.to(self.device)
            )
        gen_state = self.world_model.decoder(gen_latent.unsqueeze(1)).squeeze(1)
        logits = self.networks.policy(gen_state)
        action_distribution = self.networks.create_action_distributions(logits)
        gen_action, _ = action_distribution.sample()
            
        state, action, reward_hat, termination_hat = self.world_model.imagine_data(
                agent=self.networks,
                sample_obs=gen_state,
                sample_action=gen_action,
                imagine_batch_size=imagine_batch_size,
                imagine_batch_length=1 
            )
        if not is_train:
            # weight = torch.ones(imagine_batch_size)
            importance_score = self.dynamics_discriminator.forward(state[:, :1].squeeze(1).float(),
                                                                   action.squeeze(1).float(),
                                                                   state[:, 1:].squeeze(1).float())
            weight = torch.clip(importance_score[:, 0] / (importance_score[:, 1] + 1e-6), 0, 3)
        else:
            weight = torch.ones(imagine_batch_size)
        imagine_samples = {
                "obs": state[:, :1].squeeze(1),
                "act": action.squeeze(1),
                "rew": reward_hat.squeeze(1),
                "done": termination_hat.squeeze(1),
                "obs2": state[:, 1:].squeeze(1),
                "weight": weight
            }
        return imagine_samples
    
    def get_imagine_samples_pim(self, imagine_batch_size, target_zone, is_train=True):
        self.generator_model.mode = "e"
        gen_latent = self.generator_model.guided_sample_batch(
                batch_size=imagine_batch_size, 
                world_model=self.world_model, 
                policy_net=self.networks,
                target_zone=target_zone.to(self.device)
            )
        gen_state = self.world_model.decoder(gen_latent.unsqueeze(1)).squeeze(1)
        logits = self.networks.policy(gen_state)
        action_distribution = self.networks.create_action_distributions(logits)
        gen_action, _ = action_distribution.sample()
            
        state, action, reward_hat, termination_hat = self.world_model.imagine_data(
                agent=self.networks,
                sample_obs=gen_state,
                sample_action=gen_action,
                imagine_batch_size=imagine_batch_size,
                imagine_batch_length=1 
            )
        if not is_train:
            # weight = torch.ones(imagine_batch_size)
            importance_score = self.dynamics_discriminator.forward(state[:, :1].squeeze(1).float(),
                                                                   action.squeeze(1).float(),
                                                                   state[:, 1:].squeeze(1).float())
            weight = torch.clip(importance_score[:, 0] / (importance_score[:, 1] + 1e-6), 0, 3)
        else:
            weight = torch.ones(imagine_batch_size)
        imagine_samples = {
                "obs": state[:, :1].squeeze(1),
                "act": action.squeeze(1),
                "rew": reward_hat.squeeze(1),
                "done": termination_hat.squeeze(1),
                "obs2": state[:, 1:].squeeze(1),
                "weight": weight
            }
        return imagine_samples
    def mixed_sample(self):
        policy_flag = False
        total_batch_size = 2 * self.replay_batch_size
        # total_batch_size = 1024
        imagine_ratio = max((self.replay_duration + self.replay_start - self.iteration) / self.replay_duration, self.min_ratio) * int(self.iteration >= self.replay_start)
        imagine_batch_size = int(total_batch_size * imagine_ratio)
        real_batch_size = total_batch_size - imagine_batch_size
        env_samples = self.buffer.sample_batch(total_batch_size)
        # real_samples = self.buffer.sample_batch(real_batch_size)
        indices = torch.randperm(total_batch_size)[:real_batch_size]
        real_samples = {k: v[indices] for k, v in env_samples.items()}
        
        if self.iteration % self.replay_interval == 0 and self.iteration >= self.replay_start:
            policy_flag = False
            imagine_samples_pev = self.get_imagine_samples_pev(imagine_batch_size, env_samples["obs"].cpu(), is_train=False)
            if self.pim_flag == True:
                imagine_samples_pim = self.get_imagine_samples_pim(int(imagine_batch_size/2), env_samples["obs"].cpu(), is_train=False)
            else:
                imagine_samples_pim = imagine_samples_pev
            # 确保返回的 `imagine_samples_pev` 在 GPU
            for key in imagine_samples_pev.keys():
                imagine_samples_pev[key] = move_to_device(imagine_samples_pev[key].detach(), self.device)
                imagine_samples_pim[key] = move_to_device(imagine_samples_pim[key].detach(), self.device)
            # for key in imagine_samples_pim.keys():
            #     imagine_samples_pim[key] = move_to_device(imagine_samples_pim[key].detach(), self.device)

            # 清理未使用变量
            gc.collect()
            torch.cuda.empty_cache()

            with torch.no_grad():
                mixed_samples_pev = {}
                mixed_samples_pim = {}

                # for key in real_samples.keys():
                #     real_samples[key] = move_to_device(real_samples[key].detach(), self.device)

                #     if key in imagine_samples_pev:
                #         mixed_samples_pev[key] = torch.cat([real_samples[key], imagine_samples_pev[key]], dim=0)
                #     else:
                #         mixed_samples_pev[key] = torch.cat([real_samples[key], real_samples[key][:imagine_batch_size]], dim=0)
                for key in env_samples.keys():
                    env_samples[key] = move_to_device(env_samples[key].detach(), self.device)

                    if key in imagine_samples_pev:
                        mixed_samples_pev[key] = torch.cat([env_samples[key], imagine_samples_pev[key]], dim=0)
                        mixed_samples_pim[key] = torch.cat([env_samples[key], imagine_samples_pim[key]], dim=0)
                    else:
                        mixed_samples_pev[key] = torch.cat([env_samples[key], env_samples[key][:imagine_batch_size]], dim=0)
                        mixed_samples_pim[key] = torch.cat([env_samples[key], env_samples[key][:imagine_batch_size]], dim=0)
                # 释放临时变量
                del imagine_samples_pev
                gc.collect()
                torch.cuda.empty_cache()

            return mixed_samples_pev, mixed_samples_pim

        return env_samples, env_samples
    # def mixed_sample(self):
    #     policy_flag = False
    #     total_batch_size = 2*self.replay_batch_size
    #     imagine_ratio = max((self.replay_duration+self.replay_start-self.iteration)/self.replay_duration,self.min_ratio)*int(self.iteration  >= self.replay_start)

    #     # imagine_batch_size = int(total_batch_size * imagine_ratio)
    #     imagine_batch_size = int(self.replay_batch_size*imagine_ratio)
    #     real_batch_size = self.replay_batch_size

    #     # real_samples = self.world_replayer.replay_batch(real_batch_size)
    #     real_samples = self.buffer.sample_batch(real_batch_size)
    #     if (self.iteration) % self.replay_interval == 0 and  self.iteration  > self.replay_start :
    #     # if self.iteration  >= self.replay_start :
    #         policy_flag = False
    #         imagine_samples_pev = self.get_imagine_samples_pev(imagine_batch_size, real_samples["obs"],is_train=False)
    #         # imagine_samples_pim = self.get_imagine_samples_pim(imagine_batch_size, real_samples["obs"],is_train=False)
            
    #         imagine_samples_pim = imagine_samples_pev
    #         # imagine_samples,imagine_obs2 = self.get_imagine_samples_wodiffusion(imagine_batch_size, real_samples["obs2"],real_samples["act"],is_train=False)
    #         # with torch.no_grad():       
    #         #     loss = torch.nn.functional.mse_loss(imagine_obs2,real_samples['obs2'].to(self.device))
    #         # print(loss)
    #         # print("loss_obs2:"{torch.nn.functional.mse_loss(imagine_obs2,imagine_samples['obs2'])})
    #         # 3. 合并两种数据
    #         with torch.no_grad():
    #             mixed_samples_pev = {}
    #             for key in real_samples.keys():
    #                 if key in imagine_samples_pev:
    #                     mixed_samples_pev[key] = torch.cat([
    #                         real_samples[key].to(self.device),
    #                         imagine_samples_pev[key]
    #                     ], dim=0)
    #                 else:
    #                     # 对于imagine_samples中没有的键，用real_samples的值填充
    #                     mixed_samples_pev[key] = torch.cat([
    #                         real_samples[key].to(self.device),
    #                         real_samples[key][:imagine_batch_size].to(self.device)
    #                     ], dim=0)
                        
    #         with torch.no_grad():
    #             mixed_samples_pim = {}
    #             for key in real_samples.keys():
    #                 if key in imagine_samples_pim:
    #                     mixed_samples_pim[key] = torch.cat([
    #                         real_samples[key].to(self.device),
    #                         imagine_samples_pim[key]
    #                     ], dim=0)
    #                 else:
    #                     # 对于imagine_samples中没有的键，用real_samples的值填充
    #                     mixed_samples_pim[key] = torch.cat([
    #                         real_samples[key].to(self.device),
    #                         real_samples[key][:imagine_batch_size].to(self.device)
    #                     ], dim=0)
    #         return mixed_samples_pev,real_samples
        
    #     # 如果world_replayer还没准备好，返回全部真实数据
    #     return real_samples,real_samples    
    @gin.configurable    
    def step(self):
        # sampling
        torch.cuda.empty_cache()
        world_model_tb_dict = {
            "World_model/reward_loss": 0,
            "World_model/termination_loss": 0,
            "World_model/dynamics_loss": 0,
            "World_model/total_loss": 0,
        }
        
        # if self.iteration % self.sample_interval == 0:

        #     sampler_samples, sampler_tb_dict = self.sampler.sample()
        #     self.buffer.add_batch(sampler_samples)
        #     samples = experience_to_dict(sampler_samples)
        #     self.world_replayer.append(samples)
            
        if self.iteration % self.sample_interval == 0:
            # with ModuleOnDevice(self.networks, device="cpu"):
            self.sampler.mode = "train"
            sampler_samples, sampler_tb_dict = self.sampler.sample()
            self.buffer.add_batch(sampler_samples)
            self.sampler_tb_dict.add_average(sampler_tb_dict)        
            
        self.sample_ratio = (self.iteration-self.replay_start)/(self.max_iteration-self.replay_start) if self.iteration > self.replay_start else 0
        # replay
        # imaging_samples = self.get_imagine_samples_wodiffusion(self.replay_batch_size,self.sample_ratio)
        replay_samples_pev,real_samples = self.mixed_sample()
        
        
        
        # learning
        if self.use_gpu:
                for k, v in replay_samples_pev.items():
                    replay_samples_pev[k] = v.to(self.device).detach()
                    
                # for k, v in replay_samples_pim.items():
                #     replay_samples_pim[k] = v.to(self.device).detach()
                    
                for k, v in real_samples.items():
                    real_samples[k] = v.to(self.device).detach()
                    
        self.generator_model.to(self.device)
        
        # if self.iteration % self.training_interval == 0:
        self.generator_model.train()
        state=real_samples["obs"]
        latent = self.world_model.state_action_emb(state.unsqueeze(1)).squeeze(1)
        gen_info = self.generator_model.train_step(latent)
        if self.iteration % self.log_save_interval == 0:
            add_scalars(gen_info, self.writer, step=self.iteration)
        self.networks.train()
        # alg_tb_dict = self.alg.local_update(replay_samples_pev,replay_samples_pim, self.iteration)
        alg_tb_dict = self.alg.local_update(replay_samples_pev,real_samples,self.iteration)
            # print('iteration update complete!')
        self.networks.eval()
        
        #world model training
        train_interval = self.training_interval if self.iteration < self.replay_start else self.training_interval*10
        if self.iteration % (train_interval) == 0:
            self.world_model.train()
            with torch.set_grad_enabled(True):
                replayed_data = self.buffer.replay(batch_size=self.replay_batch_size , batch_length=self.batch_length)
                # replays = self.world_replayer.replay(self.replay_batch_size, self.batch_length)
                world_model_tb_dict = self.world_model.update(obs=replayed_data.obs.to(self.device), 
                                        action=replayed_data.action.to(self.device), 
                                        reward=replayed_data.reward.to(self.device),
                                        next_obs = replayed_data.obs2.to(self.device),
                                        termination=replayed_data.termination.to(self.device))
            self.world_model.eval() 
            if self.iteration % self.log_save_interval == 0:
                add_scalars(world_model_tb_dict, self.writer, step=self.iteration)

        # log
        if self.iteration % self.log_save_interval == 0:
            print("Iter = ", self.iteration)
            add_scalars(alg_tb_dict, self.writer, step=self.iteration)
            add_scalars(self.sampler_tb_dict.pop(), self.writer, step=self.iteration)


        # save
        if self.iteration % self.apprfunc_save_interval == 0:
            self.save_apprfunc()
            
        # evaluate
        if self.iteration - self.last_eval_iteration >= self.eval_interval:
            if self.evluate_tasks.count == 0:
                # There is no evaluation task, add one.
                self._add_eval_task()
            elif self.evluate_tasks.completed_num == 1:
                # Evaluation tasks is completed, log data and add another one.
                objID = next(self.evluate_tasks.completed())[1]
                total_avg_return = ray.get(objID)
                self._add_eval_task()

                if (
                    total_avg_return >= self.best_tar
                    and self.iteration >= self.max_iteration / 5
                ):
                    self.best_tar = total_avg_return
                    print("Best return = {}!".format(str(self.best_tar)))

                    for filename in os.listdir(self.save_folder + "/apprfunc/"):
                        if filename.endswith("_opt.pkl"):
                            os.remove(self.save_folder + "/apprfunc/" + filename)

                    torch.save(
                        self.networks.state_dict(),
                        self.save_folder
                        + "/apprfunc/apprfunc_{}_opt.pkl".format(self.iteration),
                    )

                self.writer.add_scalar(
                    tb_tags["Buffer RAM of RL iteration"],
                    self.buffer.__get_RAM__(),
                    self.iteration,
                )
                self.writer.add_scalar(
                    tb_tags["TAR of RL iteration"], total_avg_return, self.iteration
                )
                self.writer.add_scalar(
                    tb_tags["TAR of replay samples"],
                    total_avg_return,
                    self.iteration * self.replay_batch_size,
                )
                self.writer.add_scalar(
                    tb_tags["TAR of total time"],
                    total_avg_return,
                    int(time.time() - self.start_time),
                )
                self.writer.add_scalar(
                    tb_tags["TAR of collected samples"],
                    total_avg_return,
                    self.sampler.get_total_sample_number(),
                )

    def train(self):
        
        while self.iteration < self.max_iteration:
            self.step()
            self.iteration += 1

        self.save_apprfunc()
        self.writer.flush()

    def save_apprfunc(self):
        torch.save(
            self.networks.state_dict(),
            self.save_folder + "/apprfunc/apprfunc_{}.pkl".format(self.iteration),
        )

    def _add_eval_task(self):
        with ModuleOnDevice(self.networks, "cpu"):
            self.evaluator.load_state_dict.remote(self.networks.state_dict())
        self.evluate_tasks.add(
            self.evaluator,
            self.evaluator.run_evaluation.remote(self.iteration)
        )
        self.last_eval_iteration = self.iteration
