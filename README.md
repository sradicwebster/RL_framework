# Framework for DRL algorithms

![pseudo_code](pseudo_code.png)

| Algorithm | Buffer size | Sampling steps | Minibatch size |  Value target | Policy gradient | Action selection | Additions |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Actor critic | 1 | 1 | 1 | $\delta=r+\gamma\ V(s^\prime)$ | $A(s,a) \nabla\ log(\pi)$ where $A(s,a) \approx \delta$ | Policy net softmax | - |
| PPO | $\approx 32$ | $\approx 32$ | $\approx 32$ ordered sample | reward-go-to | $A(s,a) \frac{\pi}{\pi_{t-1}}$ | Policy net softmax | Policy ratio is clipped |  
| DQN | $\approx 10000$ | 1 | $\approx 32$ | $r+\gamma\ max_a(Q(s^\prime,A))$ | - | $\epsilon$-greedy | Target Q-net (hard updates) |
| DDPG | $\approx 10000$ | 1 | $\approx 32$ | $r+\gamma\ Q(s^\prime,\mu(s^\prime)) | $Q(s,\mu(s))$ | $\mu(s) + Noise$ | Target Q-net and policy net (soft updates)

