# Chapter 04: Dynamic Programming

Dynamic programming(DP)는 MDP를 정확하게 알고 있을 때, 최적정책을 구하는 알고리즘들을 일컫는 말이다. MDP를 정확하게 알고 있다는 것은 환경에 대한 정보를 모두 알고 있다는 뜻으로 전이확률행렬(probability transition matrix) $P_{s s^{\prime}}^{a}$나 보상함수 $R$을 알고 있음을 의미한다. 물론 실제 강화학습 문제에서 이렇게 환경에 대한 정보를 완벽하게 알고 있는 것은 흔한 상황은 아니다. 전통적인 DP 알고리즘들은 이러한 이유로 사용에 제약이 있는 편이다. 그럼에도 불구하고 강화학습의 문제를 접근하는 중요한 이론적 토대를 제공하며 이후에 다루는 내용은 DP에서 접근하는 방식을 보다 적은 계산비용으로, 완벽하지 않은 환경정보인 상황에서 접근하고 있다고도 할 수 있을 것이다.

이번 문서에서는 DP가 어떻게 3장에서 다루었던 value function을 계산하는데 사용할 수 있는지를 볼 것이다. 일단 optimal value function만 찾으면 optimal value function에 대해 greedy하게 행동을 한 것이 optimal policy가 되므로 이번 장에서는 Bellman optimality equation이 자주 등장하게 된다. 특히, Bellman optimality equation에서 $p(s^{\prime}, r \mid s, a)$가 포함된 꼴을 사용하게 되는데, 이 term을 사용한다는 자체가 환경에 대한 dynamics를 알고 있음을 전제로 함에 유의하며 살펴보도록 하자. 본론에 들어가기에 앞서 다음의 Bellman optimality equation을 한 번 더 확인하도록 하자.

$$
\begin{aligned}
v_{*}(s)=& \max _{a} \mathbb{E}\left[R_{t+1}+\gamma v_{*}\left(S_{t+1}\right) \mid S_{t}=s, A_{t}=a\right] \\
=& \max _{a} \sum_{s^{\prime}, r} p\left(s^{\prime}, r \mid s, a\right)\left[r+\gamma v_{*}\left(s^{\prime}\right)\right], \text { or } \\
q_{*}(s, a) &=\mathbb{E}\left[R_{t+1}+\gamma \max _{a^{\prime}} q_{*}\left(S_{t+1}, a^{\prime}\right) \mid S_{t}=s, A_{t}=a\right] \\
&=\sum_{s^{\prime}, r} p\left(s^{\prime}, r \mid s, a\right)\left[r+\gamma \max _{a^{\prime}} q_{*}\left(s^{\prime}, a^{\prime}\right)\right]
\end{aligned}
$$

## Policy Evaluation (Prediction)

시작하기에 앞서 강화학습은 크게 정책의 평가와 정책의 학습(개선)의 단계로 나누어 볼 수 있으며 평가에 해당하는 부분을 prediction, 학습(개선)에 해당하는 부분을 control이라고 한다. 우선 임의의 정책 $\pi$를 평가하는 방법인 **policy evaluation**을 알아보자.

$$
\begin{aligned}
v_{\pi}(s) & \doteq \mathbb{E}_{\pi}\left[G_{t} \mid S_{t}=s\right] \\
&=\mathbb{E}_{\pi}\left[R_{t+1}+\gamma G_{t+1} \mid S_{t}=s\right] \\
&=\mathbb{E}_{\pi}\left[R_{t+1}+\gamma v_{\pi}\left(S_{t+1}\right) \mid S_{t}=s\right] \\
&=\sum_{a} \pi(a \mid s) \sum_{s^{\prime}, r} p\left(s^{\prime}, r \mid s, a\right)\left[r+\gamma v_{\pi}\left(s^{\prime}\right)\right]
\end{aligned}
$$

상태가치는 정의에 의해 해당 상태에서 받을 수 있는 return의 기댓값이다. 여기서 중요하게 보아야 할 것 중 하나는 expectation operator의 첨자인 $\pi$이다. 특정 정책 $\pi$에 대한 평가이므로 당연히 MDP에서 $\pi$를 따르고 있음을 전제로 해야한다. 세번째에서 네번째로 넘어가는 과정은 두 가지 의미로 해석할 수 있다. 수식만 보았을 때는 기대값을 전개하는 것으로 $\mathbb{E}[X] = \sum x p(x)$에서 random variable은 $\left[r+\gamma v_{\pi}\left(s^{\prime}\right)\right]$에 대응하고 그 때의 확률은 $\sum_{a} \pi(a \mid s) \sum_{s^{\prime}, r} p\left(s^{\prime}, r \mid s, a\right)$에 대응한다. 정책 $\pi(a \mid s)$가 정책 $\pi$일 때, 상태 $s$에서 행동 $a$를 선택할 확률이므로 직관적으로 이해할 수 있다. 계속 강조하지만 전이확률행렬인 $p$를 알고 있어야 위 식을 쓸 수 있음을 기억하자. 

위 식을 바탕으로 **iterative policy evaluation**이라는 정책평가 방법을 사용할 수 있다.

<figure align=center>
<img src="assets/images/Chapter04/Iterative_Policy_Evaluation.png" width=100% height=100%/>
<figcaption>Iterative Policy Evaluation</figcaption>
</figure>

순서대로 살펴보자. 정책을 평가하는 방법이므로 정책 $\pi$는 주어져있다고 가정한다. 그리고 정책이 수렴했는지를 확인하기 위해서 update가 일정수준 이하라면 종료하는 조건을 사용한다. 이 때, 모든 상태에서의 update되는 양의 최댓값이 threshold $\theta > 0$보다 작다면 종료한다. 상태가치 $V(s)$를 저장하기위해서 각 상태별로 상태가치의 값을 담을 수 있는 array가 준비되어야 한다. 종료상태에서의 상태가치는 $V(\text{terminal}) = 0$이다. 학습은 간단하게 이루어 진다. 우선 update양은 0으로 초기화가 된다. 그리고 각 상태별로 방문하며 현재 $V(s)$를 저장한 array의 값을 $v$로 assign한다. 여기서 update된 상태가치는 위 식에 의해 $\sum_{a} \pi(a \mid s) \sum_{s^{\prime}, r} p\left(s^{\prime}, r \mid s, a\right)\left[r+\gamma v_{\pi}\left(s^{\prime}\right)\right]$이다. 이 값은 $V(s)$로 assign된다. 기존에 있던 값 $v$와 현재 update한 값 $V(s)$의 차이와 이전 변화량 $\Delta$ 둘 중 최댓값이 threshold보다 작아질 때 까지 이 과정을 반복한다.

Loop의 순서를 보면 한 상태에서 계속 update를 하는 것이 아니라 모든 상태를 한 바퀴 다 돌고 다시 각각의 상태가 update된다는 점을 유의하자. 간단하게 모든상태에 대해서 한번씩 들리면서 $V(s)$를 update하고 한 바퀴를 다 돌았으면 다시 처음상태로 돌아가 동일한 과정을 반복하는 것이다. 이 과정을 반복하게 되면 $V(s)$ array가 어떤 값들로 수렴하게 될텐데 이 값들을 바로 실제 상태가치의 근사값이라고 보는 것이다. 여기서 의문이 생기는 점은 이런 iterative 방식이 수렴성과 존재성이 보장되느냐하는 것인데 결론부터 이야기하면 iterative policy evaluation 방법은 무한히 했을 때 실제 가치(하지만 주어지지 않는 한 영원히 알 수 없는) $v_{\pi}$로 수렴함이 증명되어 있다. 이로써 임의의 정책을 평가할 수 있는 도구를 얻게 되었다. 계속 강조하는 바지만 이 방법은 MDP를 알고 있을 때 평가할 수 있는 방법임을 상기하자. 증명에 관한 논의는 [Stack Exchange](https://ai.stackexchange.com/questions/20309/what-is-the-proof-that-policy-evaluation-converges-to-the-optimal-solution)에 언급되어 있다.

## Policy Improvement

정책의 평가에 대해서 알아보았다. 궁극적으로는 정책을 평가해서 더 나은 정책을 찾는데 강화학습의 목적이 있다. 정책 $\pi$의 가치함수 $v_{\pi}$를 갖고 있다고 해보자. 임의의 상태 $s$에서 현재 정책 $\pi$를 따를지 다른 정책을 선택할지를 판단하는 기준을 생각해볼때, 행동가치함수를 생각해볼 수 있다. 행동가치에 대한 Bellman expectation equation은 다음과 같다.

$$
\begin{aligned}
q_{\pi}(s, a) & \doteq \mathbb{E}\left[R_{t+1}+\gamma v_{\pi}\left(S_{t+1}\right) \mid S_{t}=s, A_{t}=a\right] \\
&=\sum_{s^{\prime}, r} p\left(s^{\prime}, r \mid s, a\right)\left[r+\gamma v_{\pi}\left(s^{\prime}\right)\right]
\end{aligned}
$$

만약 어떤 행동 $a$를 선택했을 때, $q(s,a)$가 $v_{\pi}(s)$보다 크다면 현재 정책은 적어도 최적이 아니다. $a$를 선택함으로써 더 높은 기대 return을 얻을 수 있기 때문이다. 같은 논리로 모든 상태 $s$에 대해서 더 높은 $q(s,a)$를 만드는 $a$를 선택하는 것만으로도 더 좋은 정책이 된다. 이러한 논리는 **policy improvement theorem**에 의해 정당화된다.

두 정책 $\pi, \pi^{\prime}$이 있다고 해보자. 모든 상태 $s \in \mathcal{S}$에 대해서 $q_{\pi}\left(s, \pi^{\prime}(s)\right) \geq v_{\pi}(s)$이면 $\pi^{\prime}$은 $\pi$보다 더 좋은 정책임이 자명하다. 이에 대한 증명은 다음과 같다.

$$
\begin{aligned}
v_{\pi}(s) & \leq q_{\pi}\left(s, \pi^{\prime}(s)\right) \\
&=\mathbb{E}\left[R_{t+1}+\gamma v_{\pi}\left(S_{t+1}\right) \mid S_{t}=s, A_{t}=\pi^{\prime}(s)\right] \\
&=\mathbb{E}_{\pi^{\prime}}\left[R_{t+1}+\gamma v_{\pi}\left(S_{t+1}\right) \mid S_{t}=s\right] \\
& \leq \mathbb{E}_{\pi^{\prime}}\left[R_{t+1}+\gamma q_{\pi}\left(S_{t+1}, \pi^{\prime}\left(S_{t+1}\right)\right) \mid S_{t}=s\right] \\
&=\mathbb{E}_{\pi^{\prime}}\left[R_{t+1}+\gamma \mathbb{E}\left[R_{t+2}+\gamma v_{\pi}\left(S_{t+2}\right) \mid S_{t+1}, A_{t+1}=\pi^{\prime}\left(S_{t+1}\right)\right] \mid S_{t}=s\right] \\
&=\mathbb{E}_{\pi^{\prime}}\left[R_{t+1}+\gamma R_{t+2}+\gamma^{2} v_{\pi}\left(S_{t+2}\right) \mid S_{t}=s\right] \\
& \leq \mathbb{E}_{\pi^{\prime}}\left[R_{t+1}+\gamma R_{t+2}+\gamma^{2} R_{t+3}+\gamma^{3} v_{\pi}\left(S_{t+3}\right) \mid S_{t}=s\right] \\
& \vdots \\
& \leq \mathbb{E}_{\pi^{\prime}}\left[R_{t+1}+\gamma R_{t+2}+\gamma^{2} R_{t+3}+\gamma^{3} R_{t+4}+\cdots \mid S_{t}=s\right] \\
&=v_{\pi^{\prime}}(s)
\end{aligned}
$$

위 증명은 특정 상태 $s$에 대해서 전개된 것인데 같은 아이디어를 확장해서 모든 상태에서 가능한 모든 행동에 대해 $q_{\pi}(s,a)$를 최대로 하는 $a$를 선택하게 할 수도 있다. 이러한 방식을 **greedy policy**라고 하며 다음과 같이 표현할 수 있다.

$$
\begin{aligned}
\pi^{\prime}(s) & \doteq \underset{a}{\arg \max } q_{\pi}(s, a) \\
&=\underset{a}{\arg \max } \mathbb{E}\left[R_{t+1}+\gamma v_{\pi}\left(S_{t+1}\right) \mid S_{t}=s, A_{t}=a\right] \\
&=\underset{a}{\arg \max } \sum_{s^{\prime}, r} p\left(s^{\prime}, r \mid s, a\right)\left[r+\gamma v_{\pi}\left(s^{\prime}\right)\right]
\end{aligned}
$$

이러한 방식은 이후상태를 고려하지 않아 근시안적인 정책이라고 할 수 있다. 지금의 선택이 이후에 어떻게 영향을 줄지는 고려하지 않고 당장의 행동가치가 큰 행동만 선택하기 때문이다. 딱 다음단계만 내다보고(one step of lookahead) 행동을 결정하는 것이다.