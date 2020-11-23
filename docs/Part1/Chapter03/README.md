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

$$
p\left(s^{\prime}, r \mid s, a\right) \doteq \operatorname{Pr}\left\{S_{t}=s^{\prime}, R_{t}=r \mid S_{t-1}=s, A_{t-1}=a\right\}
$$

눈여겨볼점은 이렇게 정의된 함수 $p$이다. $p$는 MDP가 어떻게 작동하는지를 결정하게 된다. MDP가 작동하는 방식에 대해 포괄적으로 dynamics라고 하며, 따라서 $p$가 MDP의 dynamics를 정의한다고 표현할 수 있다. 위 식에도 definition 기호인 $\doteq$를 사용하고 있다. $p$는 네개의 인자를 갖는 함수로 $p: \mathcal{S} \times \mathcal{R} \times \mathcal{S} \times \mathcal{A} \rightarrow [0, 1]$이다. 식에 있는 $\mid$ 기호는 조건부확률처럼 자연스럽게 해석할 수 있다. 강화학습의 dyanmics식에서 $\mid$는 $\mid$ 뒤의 "상태와 행동을 선택했을 때"를 의미한다.

$$
\sum_{s^{\prime} \in \mathcal{S}} \sum_{r \in \mathcal{R}} p\left(s^{\prime}, r \mid s, a\right)=1, \text { for all } s \in \mathcal{S}, a \in \mathcal{A}(s)
$$

