# Chapter 03: Finite Markov Decision Process

이 문서에서는 finite Markov decision process(finite MDP)에 대해 다룬다. 어떤 문제를 풀기 위해서는 다루려는 문제를 어떻게 바라보고 접근할지를 정해야 한다. 강화학습을 공부할 때 MDP가 반드시 등장하게 되는 이유는 강화학습이 다루는 문제를 MDP로 접근하기가 용이하기 때문이다. MDP는 강화학습이 다루는 순차적 의사결정 문제를 잘 표현하는 것은 물론 지연보상과 같은 강화학습 문제의 특징도 잘 통합해 다룰 수 있는 토대를 제공한다. Bandit 문제에서처럼 evaluative feedback을 반영하는 상황과 다른 상황에서 다른 행동을 취해야 하는 associative문제 모두 MDP를 이용해 표현이 가능하다. 

Bandit 문제에서는 상태라는 것이 처음에 보는 화면이외의 상태라고 할만한 것이 없었기에 가능한 $a$에 대해 $q_{*}(a)$를 잘 추정하는 문제로 한정되었었다. 하지만 일반적인 강화학습 문제는 다양한 상황이 존재한다. MDP는 이처럼 상황이 한 번에 종료되는 것이 아닌 연속적인 상황을 다루는데 적절하며 최적의 방법을 찾기 위해 MDP에서는 상태 $s$에서 행동 $a$에 대한 행동가치 $q_{*}(s, a)$를 추정하거나 최적의 행동이 주어졌을 때의 상태가치 $v_{*}(s)$를 추정하게 된다. 둘 다 상태에 의존함을 알 수 있는데, 이는 각각의 행동에 대한 장기적인 결과를 적절하게 추정하기 위한 필연적인 성질이다.

이번 chapter에서는 MDP가 강화학습의 문제를 어떻게 수학적으로 표현하는지를 다루고 그 과정에서 return, 가치함수, Bellman equation에 대해서 다루게 된다. MDP도 인공지능의 다른 접근방법처럼 적용할 수 있는 범위와 수학적 tracktability가 trade-off관계에 있다. 풀어 말하자면, 더 넓은 범위에 적용하려고 하면 수학적으로 tractable해지지 않고, 수학적으로 tractable한 접근을 하려면 문제의 범위를 제한해야한다는 trade-off가 있게 된다. 이번 chapter에서는 이러한 trade-off와 이로인한 어려움 점들에 대해서도 논의한다. MDP를 벗어나는 강화학습은 교재 chapter 17에서 다룬다.

## The Agent-Environment Interface

MDP는 강화학습처럼 환경과 상호작용하며 목적을 달성해야 하는 문제를 다루는데 적합한 토대를 제공한다. 학습하고 행동을 결정하는 주체를 agent라고 하며, agent가 상호작용 하는 대상을 환경(environment)라고 한다. 환경이 꼭 물리적으로 외부에 존재할 필요는 없다. Agent기준으로 외부에 있는 것은 모두 환경이라고 할 수 있다.

<figure align=center>
<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/1/1b/Reinforcement_learning_diagram.svg/1024px-Reinforcement_learning_diagram.svg.png" width=50% height=50%/>
<figcaption>Wikipedia: Reinforcement Learning</figcaption>
</figure>

Agent가 상태를 바탕으로 어떤 행동을 하게되면 환경은 해당 행동에따른 변화된 상태와 보상을 제공한다. 그리고 agent는 이 정보를 바탕으로 다음의 행동을 이어가게 된다.

순서상 닭이 먼저냐 달걀이 먼저냐 문제가 되지만 일반적으로 어떤 상태를 보고 취한 행동을 같은 time step으로 간주한다. 풀어서 설명하면 위 상호작용이 time step $t = 0, 1, 2, 3, \ldots$와 같이 발생한다고 할 때, 특정 $t$에서 agent는 환경의 상태 $S_t \in \mathcal{S}$를 보고 행동 $A_t \in \mathcal{A}(s)$를 선택한다. 그러면 그 결과로 다음 time step $t+1$에서 보상 $R_{t+1} \in \mathcal{R} \subset \mathbb{R}$과 함께 새로운 상태 $S_{t+1}$을 보게 된다. 이러한 일련의 과정을 **trajectory**라고 하며 순서대로 쓰면 다음과 같다.

$$S_{0}, A_{0}, R_{1}, S_{1}, A_{1}, R_{2}, S_{2}, A_{2}, R_{2}, \ldots$$

여기서는 **finite** MDP만을 다룬다. 즉, MDP에서 가질 수 있는 상태, 행동으 그리고 보상의 수는 모두 유한하다. Finite MDP를 가정하면 이전 상태와 행동으로부터 현재의 보상과 상태에 대한 이산확률분포를 갖게된다. 즉 다음을 정의할 수 있게 된다.

> [!NOTE]
> **Four-argument Dyanmics Function**
> $$p\left(s^{\prime}, r \mid s, a\right) \doteq \operatorname{Pr}\left\{S_{t}=s^{\prime}, R_{t}=r \mid S_{t-1}=s, A_{t-1}=a\right\}$$

눈여겨볼점은 이렇게 정의된 함수 $p$이다. $p$는 MDP가 어떻게 작동하는지를 결정하게 된다. MDP가 작동하는 방식에 대해 포괄적으로 dynamics라고 하며, 따라서 $p$가 MDP의 dynamics를 정의한다고 표현할 수 있다. 위 식에도 definition 기호인 $\doteq$를 사용하고 있다. $p$는 네개의 인자를 갖는 함수로 $p: \mathcal{S} \times \mathcal{R} \times \mathcal{S} \times \mathcal{A} \rightarrow [0, 1]$이다. 식에 있는 $\mid$ 기호는 조건부확률처럼 자연스럽게 해석할 수 있다. 강화학습의 dyanmics식에서 $\mid$는 $\mid$ 뒤의 "상태와 행동을 선택했을 때"를 의미한다.

$$
\sum_{s^{\prime} \in \mathcal{S}} \sum_{r \in \mathcal{R}} p\left(s^{\prime}, r \mid s, a\right)=1, \text { for all } s \in \mathcal{S}, a \in \mathcal{A}(s)
$$

> [!NOTE]
>
> MDP에서 환경의 dynamics는 $p$에 의해 결정된다.

이를 좀 더 자세히 알아보자. MDP는 다음과 같은 정보를 사용한다.

$$MDP \coloneqq (\mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma)$$

이후에 반복해서 언급하겠지만 이쯤에서 MDP를 **안다**라는 의미에 대해 생각해보자. 결론부터 이야기하면 MDP를 안다는 것은 위의 모든 인자들을 안다는 것이다. 상태와 행동은 agent입장에서 관측하고 결정하는 행동이므로 agent가 알 수 있다. 그렇다면 전이확률과 보상함수는 어떨까? 현재 상태에서 특정 상태로 갈 확률 그리고 현재 상태에서 특정 행동을 했을 때의 보상은 agent가 아닌 **환경이 결정하는 영역**이다. 문제의 성질에 따라 agent가 환경에 대한 정보를 알 수도 모를수도 있게 되는데 환경에 대한 정보, 즉 환경에 대한 모델을 알고 있다면 model-based가 되는 것이고 환경에 대한 모델을 모른다면 model-free가 되는 것이다. 정리하면 model-based의 경우 MDP에 대해 완전히(complete) 알고 있을 때 가능한 방식이다. 세부적인 내용은 이후에도 반복되므로 여기서는 MDP를 안다는 것은 위의 모든 인자들을 안다는 것을 의미한다 정도로 이해하는 것으로 충분하다.

그리고 MDP에서 중요한 것이 **Markov property**이다. 내용은 간단하다. 어떠한 상태가 과거의 모든 agent-environment interaction을 갖고 있다면 이는 Markov property를 만족한다고 한다. 다음상태, 즉 미래는 현재상태에만 의존하지 다음상태를 결정하는 함수는 현재상태이외의 과거상태에 의존하지 않는다는 의미이다. MDP는 기본적으로 Markov property를 만족한다.

> [!WARNING]
>
> Markov property에서 자주 발생하는 오해가 바로 상태와 시점의 구분이다. 꼭 현재 상태가 하나의 시점으로 구성될 필요는 없다. 대표적으로 Atari게임으로 유명한 DQN의 경우 이전 4개의 프레임을 하나의 상태로 구성하였다. 따라서 시점과 상태를 동일하게 생각하지 않도록 주의하자.

위의 Four-argument Dyanimcs Function에서 보상에 대해 summation하면 다음의 state-transition probabilities $p: \mathcal{S} \times \mathcal{S} \times \mathcal{A} \rightarrow [0, 1]$를 얻을 수 있다.

$$
\begin{aligned}
p\left(s^{\prime} \mid s, a\right) &\doteq \operatorname{Pr}\left\{S_{t}=s^{\prime} \mid S_{t-1}=s, A_{t-1}=a\right\} \\ &= \sum_{r \in \mathcal{R}} p\left(s^{\prime}, r \mid s, a\right)
\end{aligned}
$$

State-action pair로 표현되는 two-argument 보상함수 $r: \mathcal{S} \times \mathcal{A} \rightarrow \mathbb{R}$는 다음과 같이 정의된다.

$$
\begin{aligned}
r(s, a) &\doteq \mathbb{E}\left[R_{t} \mid S_{t-1}=s, A_{t-1}=a\right] \\ &= \sum_{r \in \mathcal{R}} r \sum_{s^{\prime} \in \mathcal{S}} p\left(s^{\prime}, r \mid s, a\right)
\end{aligned}
$$

State-action-next-state triples로 표현되는 three-argument 보상함수 $r: \mathcal{S} \times \mathcal{A} \times \mathcal{S} \rightarrow \mathbb{R}$는 다음과 같이 정의된다.

$$
\begin{aligned}
r\left(s, a, s^{\prime}\right) &\doteq \mathbb{E}\left[R_{t} \mid S_{t-1}=s, A_{t-1}=a, S_{t}=s^{\prime}\right]\\ &= \sum_{r \in \mathcal{R}} r \frac{p\left(s^{\prime}, r \mid s, a\right)}{p\left(s^{\prime} \mid s, a\right)}
\end{aligned}
$$

대부분의 경우 Four-argument Dyanimcs Function으로 표현하기도하고 나머지는 쉽게 유도가 가능하므로 Four-argument Dyanimcs Function정도는 암기하는 것이 좋다.

MDP는 순차적 의사결정 문제에 대해 유연하게 적용할 수 있다. 책의 다음 문장이 이를 깔끔하게 설명한다.

> In general, actions can be any decisions we want to learn how to make, and the states can be anything we can know that might be useful in making them.

학습하기를 바라는 결정과정은 무엇이든 행동으로 취급할 수 있으며 그러한 행동을 위해 필요한 유용한 정보는 무엇이든 상태로 취급할 수 있다. 또한 agent와 환경의 구분이 꼭 물리적인 내부와 외부로 나누어지는 것은 아니므로 혼동될 수 있는데, 일반적으로 다음의 기준을 사용하면 구분이 용이하다.

> [!NOTE]
> 
> Agent가 마음대로 변경할 수 없는 것들은 환경의 일부로 간주한다.

MDP는 상호작용이 있는 목표지향적(goal-directed) 학습문제에 대한 framework를 제공한다는 의의가 있다. Agent와 환경이 상호작용하며 목적을 달성해야 하는 상황을 상태와 행동, 전이확률과 보상함수에서의 신호교환이라는 형태로 추상화하는 도구를 제공해 이후에 다룰 다양한 도구들을 사용할 수 있는 토대를 마련해준다.

## Goals and Rewards

강화학습의 목표는 간단하다. Agent가 return을 최대로 받게 하는 것이다. 강화학습에서의 보상은 하나의 숫자이다. 각각의 시점에서의 보상은 스칼라값 $R_t \in \mathbb{R}$로 표현할 수 있다. 그런데 복잡한 작업에 대해서도 고작 하나의 스칼라 값인 보상을 사용할 수 있을까? 더 복합적인 정보를 표현할 수 있도록, 예를 들어 "보상의 집합으로 구성해야 하지 않을까?"라는 생각을 해볼 수 있다. 결론은 그렇게 해야한다는 것이다. 보상신호가 복잡해지게 되면 학습과정은 훨씬 어려워지게 된다. 따라서 강화학습은 **reward hypothesis**라는 가정을 전제로 한다.

> [!NOTE]
> **Reward Hypothesis**
> 
> 강화학습에서 달성하고자 하는 목표는 보상의 합에 대한 기댓값을 최대화하는 문제로 치환할 수 있다.

굉장히 당연한 이야기지만 동시에 중요한 가정이다. 강화학습은 MDP를 사용해 표현하고 MDP에서 학습방향은 보상에 의해 정의가 된다. 따라서 expected return을 최대화하는 방향이 우리가 풀고자하는 문제의 방향을 나타내어야 원하는 결과를 얻을 수 있다. Reward hypothesis는 이러한 기본전제를 명시적으로 표현해준다.

또한 보상설정에서 매우 중요한 부분은 보상은 달성하고자 하는 목적 그 자체가 되어야지 달성하는 방법에 관한 것이 되어서는 안된다.

> The reward signal is your way of communicating to the robot *what* you want it to achieve, not *how* you want it achieved.

실제로 강화학습 문제에서 보상설계는 그 자체로 매우 어렵고 중요한 부분이다. 관련해서, 자주 하게 되는 실수 중 하나가 보상을 사전지식(prior knowledge)을 넣기 위한 수단으로 사용하는 것이다. 교재의 예처럼 보상은 이길 때 주어지는 어떤 양수값이 되어야지, 퀸이라는 말을 잡는 것이 보상이 되어서는 안된다. 만약 이렇게 설계한다면 게임을 지더라도 퀸을 잡는 원치 않은 결과를 얻게 될 수 있다. 따라서 보상은 원하는 목적 그 자체를 표현해야지 어떤 문제의 subgoal을 달성하도록 설정되어서는 안된다.

물론, 보상을 통해서 사전지식을 넣는게 잘못된 것이지 사전지식을 주입하는 자체가 틀린 것은 아니다. 실제로, AlphaGo는 프로기사들의 대국을 사전정보로 사용하였다. 단계적으로 목적을 달성하도록 하는 curriculum learning이라는 방법도 있다. 다만, 두 방식 모두 보상으로 사전지식을 주입하지는 않는다.

## Returns and Episodes

강화학습의 목적을 이야기하면서 보상과 return에 대한 언급이 있었다. 보상은 매 time step $t$마다 환경에 의해 주어지는 스칼라 값을 갖는 신호이다. 따라서 $R_{t}, R_{t+1}, R_{t+2}, \ldots$와 같이 표기된다. 종종 강화학습의 목적이 보상을 크게한다고 표현하는 경우가 있는데, 엄밀한 의미에서 이는 틀린 문장이다. 보상은 하나의 신호에 불과하다. 강화학습을 통해서 달성하고자 하는 것은 expected return을 최대로 하는 것이다. 즉, return의 기댓값을 최대화 하겠다는 건데 그렇다면 return의 정의를 알아보자.

> [!NOTE]
> **Definition: Return**
> 
> Return은 일련의 보상에 대한 특정 함수이다.

상당히 추상적으로 정의되어있지만 이는 return이 하나의 식으로 정해져있는 것이 아니기 때문이다. 가장 간단한 형태의 return을 예로 보자.

$$G_{t} \doteq R_{t+1} + R_{t+2} + R_{t+3} + \cdots + R_{T}$$
$T$는 마지막 time step이다.

위의 return은 단순하게 이후에 받을 보상의 합이다. 여기서 $T$, final time step에 대해 생각해보자. 강화학습은 순차적 의사결정 문제를 다룬다. 여기서 문제는 끝이 있는 문제일까? 문제에 따라 그럴수도, 그렇지 않을 수도 있다. 

Agent-environment interaction이 subsequence로 쪼개질 수 있다면 이 subsequence를 **episode**라고 한다. 대표적인 것이 게임이다. 게임의 한 판이 하나의 episode인 것이다. 바둑이라는 게임은 엄청난 숫자의 경우의 수와 전략이 존재한다. 하지만 학습을 할 때는 대국단위로 쪼개서 학습을 하게 된다. 수 많은 대국을 통해 학습하겠지만 하나의 subsequence인 대국 한 판이 episode에 해당하는 것이다. 그리고 episode의 가장 마지막 상태를 **terminal state**라고 한다. Terminal state에 도달하게 되면 다시 초기상태로 돌아가게 된다. 통계적인 관점에서 보면 한 episode가 끝나고 다른 episode가 되는 것은 다른 독립사건이 발생하는 것이다. 이전 episode에서 졌던 이겼던 다음 게임과는 상관이 없다. 이렇게 episode로 나누어질 수 있는 문제를 **episodic task**라고 한다. Episodic task에서 마지막 terminal state를 구분하기 위해 nonterminal state를 $\mathcal{S}$로, terminal state를 $\mathcal{S}^{+}$로 표기한다. 마지막 time step인 $T$는 각 episode마다 달라질 수 있는 값으로 random variable이다.

그렇다면 episodic task가 아닌 문제를 생각해보자. 모든 문제가 꼭 끝이 있지는 않다. 지속적으로 작업하는 로봇을 생각해보면 작업의 끝이 명시적으로 정해져 있지 않음을 알 수 있다. 일이 들어오는대로 계속해야하는 것이다. 이러한 작업을 **continual task**라고 한다. 여기서 문제가 생긴다. $T = \infty$가 되면서 갑자기 앞에서 정의한 return이 무한급수(infinite series)가 되어버렸다. 따라서 단순합으로 정의한 앞의 return도 무한대의 값을 가질 수 있다. 이렇게 되면 expected return을 최대화 하는 것은 무한대를 최대화하는 것으로 수학적으로 맞지 않고, 무한대의 시간을 활용해서 달성하는 것은 유용하지도 않다. 따라서 자연스럽게 discounting이라는 개념을 도입하게 된다.

> [!NOTE]
> **Definition: Discounted Return**
>
> $$\begin{aligned} G_{t} &\doteq R_{t+1} + \gamma R_{t+2} + \gamma^{2} R_{t+3} + \cdots \\ &= \sum_{k=0}^{\infty} \gamma^{k} R_{t+k+1} \end{aligned}$$
> 이 때, $gamma$는 $0 \geq \gamma \geq 1$로 **discount rate**이라고 한다.

Discounted return은 은행의 복리이자와 완전히 같은 개념이다. 내년의 돈의 가치를 올해로 환산하기 위해서는 물가상승분에 대한 discount를 해주어야 한다. 내후년에는 2년이 물가상승분만큼의 discount를 해야한다. 따라서 이후에 가지고 있을 돈을 현 시점에서 환산한다면 미래의 돈은 현재로부터 떨어진 만큼 discount를 많이 해야하는 것이다. 

강화학습의 문맥에서 $\gamma = 0$으로 본다면 오직 현재의 보상만 높게하는 근시안적인 agent가 될 것이고 $\gamma = 1$에 가까워진다면 미래의 보상값을 더 강하게 고려하는 장기적인 관점의 학습을 하는 agent가 될 것이다.

Return은 다음과 같이 time step에 대해 recursive하게 나타낼 수 있다.

$$
\begin{aligned}
G_{t} & \doteq R_{t+1}+\gamma R_{t+2}+\gamma^{2} R_{t+3}+\gamma^{3} R_{t+4}+\cdots \\
&=R_{t+1}+\gamma\left(R_{t+2}+\gamma R_{t+3}+\gamma^{2} R_{t+4}+\cdots\right) \\
&=R_{t+1}+\gamma G_{t+1}
\end{aligned}
$$

또한 $\gamma$가 $\gamma < 1$인 상수이고 보상이 매 time step마다 1이라면 다음과 같이 수렴한다.

$$
G_{t}=\sum_{k=0}^{\infty} \gamma^{k}=\frac{1}{1-\gamma}
$$

## Reference

* [Wikipedia: Reinforcement Learning](https://en.wikipedia.org/wiki/Reinforcement_learning)
* [Sutton, R. S., Barto, A. G. (2018). Reinforcement learning: An introduction. Cambridge, MA: The MIT Press.](http://www.incompleteideas.net/book/the-book-2nd.html)