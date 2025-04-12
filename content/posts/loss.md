+++
title = '大模型训练中的loss计算'
date = 2025-04-12T14:04:07+08:00
draft = false
markup = 'pandoc'

+++

## 1.pretrain阶段的loss


$$
Input = <bos>, w_1, w_2...w_T 
$$

$$
Label=w_1,w_2...w_T,<eos>
$$

$$
Loss_{pretrain} = \sum_{k=1}^{T+1} \log{P(w_t|<bos>, w_1, w_2...w_{t-1})}
$$

## 2.sft阶段的loss

$$
Input = <bos>, p_1, p_2...p_{L_p},r_1,r2...r_{L_t}
$$

$$
Label=\underbrace{[-100, -100...-100]}_{L_p+1个ignoreIndex},r_1,r2...r_{L_t}
$$

$$
Loss_{sft}=\sum_{t=L_p+2}^{L_p+L_r+1}\log{P(w_t|Input_{<t})}
$$

## 3.rlfh阶段的loss

