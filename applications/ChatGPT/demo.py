#!/usr/bin/env python
# coding: utf-8

"""
@Author     :qi.he@qdtech.ai
@Date       :2023-02-16
@Desc       :
"""
import pandas as pd
from chatgpt.nn import GPTActor, GPTCritic, RewardModel
from chatgpt.trainer import PPOTrainer
from chatgpt.trainer.strategies import ColossalAIStrategy

def train(prompts):

    strategy = ColossalAIStrategy(stage=3, placement_policy='cuda')

    with strategy.model_init_context ():
        actor = GPTActor().cuda()
        critic = GPTCritic().cuda()
        initial_model = deepcopy(actor).cuda()
        reward_model = RewardModel(deepcopy(critic.model)).cuda ()

    trainer = PPOTrainer(strategy, actor, critic, reward_model, initial_model, ...)
    trainer.fit(prompts)

if __name__ == '__main__':
    prompts = pd.read_csv('datasets/prompts.csv')['prompt']

