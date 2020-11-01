# Chapter 02: Multi-armed Bandits

강화학습 공부를 시작할 때 예제로 Multi-armed Bandit 문제가 자주 사용된다. 이 문제는 슬롯머신에서 파생한 것으로, 상대방(여기서는 슬롯머신)이 어떻게 행동하는지에 대한 정보를 모르는 상태에서 최적의 전략을 선택해야 한다는 점에서 좋은 강화학습 예제가 된다.

<figure align=center>
<img src="assets/images/Chapter02/slot_machine.jpg" width=50% height=50%/>
<figcaption>Armed bandit</figcaption>
</figure>

강화학습을 다른 기계학습방법과 구분지어주는 가장 큰 특징은 학습하는 과정에 있다. 강화학습은 기계학습처럼 정답을 알려주면서 학습하는 것이 아니라 각 행동을 스스로 평가하면서 학습한다. 즉, 학습을 하기위한 feedback의 종류가 다르며 교재에서는 선택한 행동을 기반으로 하는 feedback을 evaluative feedback, 그리고 선택한 행동과 상관없이 주어지는 feedback을 instructive feedback으로 설명한다.
 
Evaluative feedback은 행동이 **얼마나 좋았는지**에 대한 추정일 뿐, 최선의 행동이었는지를 의미하지는 않는다. 미로찾기에서 agent는 특정 상태에서 "오른쪽으로 가는 선택을 할 경우, 기대되는 보상은 1.3정도입니다"와 같은 feedback을 받는다고 생각하면 된다. 대조적으로, instructive feedback은 지도학습의 target label로 생각하면 이해가 편하다. Agent에게 "이 행동이 정답이다!"라고 명시적으로 알려주는 것이다. 이미지 분류문제에서 특정 이미지가 자동차면 자동차고 의자면 의자이지 target label을 "꽤나 자동차 사진인것 같군요!"라고 하지 않는 것으로 생각하면 된다.

Chapter 2에서 다루는 multi-armed bandit문제는 한 가지 상황에서 어떻게 행동해야 하는지만을 다루는 문제로 evaluative feedback을 이해할 수 있는 토대를 제공하며 나아가 instructive feedback과 어떻게 조합할 수 있는지를 알게 해준다. 슬롯머신과 같은 One-armed bandit상황에 비유하면 슬롯머신을 하는 사람을 레버를 당기는게 가능한 유일한 행동이며 그 결과 얻을 수 있는 상황은 슬롯머신의 결과(state)와 이에 따른 보상일 뿐이므로 한 가지 상황이라 볼 수 있다.

이번 챕터에서는 non-associative setting에서의 evaluative aspect를 다룬다. Non-associative setting은 어떤 setting을 말하는지를 먼저 간단히 알아보자.

* Non-associative:
  * 상황에 대해 독립적이다. 즉 각각의 상황은 독립사건으로 생각할 수 있다.
  * 다른 상황에서의 행동 연관지을 필요가 없다.
  * Stationary한 문제라면 최선의 행동이 존재하며 이를 선택하면 된다. Non-stationary하다면 최선의 행동을 추적하면서 선택하면 된다.
  * 예) 슬롯머신에서 지금 레버를 당기는 것과 이전에 레버를 당기는 상황은 동일하다. 슬롯머신이 고정된 확률에 의해 결과를 보여준다면 이는 Stationary & non-associative한 상황이다.

* Associative:
  * 상황에 의존적이다.
  * 해당 상황에서 행동으로의 함수(정책)를 통해 최적의 행동을 결정한다.
  * 예) 게임의 진행을 생각해보면 내가 현재 보고있는 상황은 이전에 선택한 행동의 결과이다.

그러면 본격적으로 multi-armed bandit problem을 살펴보자.

## A k-armed Bandit Problem

당신은 슬롯머신앞에 앉아있다. 그런데 일반적인 슬롯머신과 달리 이 슬롯머신은 $k$개의 레버를 가지고 있다. 당신이 레버를 당길때마다 슬롯머신은 점수를 보여줄 것이고 점수는 각각의 레버가 갖고 있는 고유의 고정된 확률분포(stationary probability distribution)에 의해 결정된다. 당신의 목표는 점수의 총합을 최대한 크게 만드는 것이며 당신은 레버를 1000번 당겨볼 수 있다. 이 때 당신은 어떤 전략을 선택할 것인가?

이 상황이 바로 $k$-armed bandit problem문제가 다루는 상황이다. 가장 높은 점수를 주는 레버가 무엇인지 알면 주구장창 그 레버만 당겨버리면 되겠지만 우리는 각 레버의 확률분포에 대한 사전지식이 없다.

그렇다면 레버마다 가지고 있는 확률분포에 대해 우리가 필요한 정보는 무엇일까? 바로 그 레버를 선택했을 때 점수에 대한 기댓값이다.

$$
q_{*}(a) \doteq \mathbb{E}[R_t \vert A_t = a]
$$

즉, 레버마다 time step $t$에서 행동 $a$를 선택했을 때의 점수에 대한 기댓값을 알 수 있다면 그 다음 선택은 그 레버만 신나게 당기면 되는 것이다. 우리는 정확한 값은 알 수 없고 추정만 할 수 있으므로 time step $t$에서 추정한 행동 $a$에 대한 가치를 $Q_t(a)$라고 하며 이 값을 실제 가치인 $q_{*}(a)$와 최대한 가깝게 추정해내는 것이 목표가 되는 것이다.

그렇다면 학습 중의 상황을 생각하자. 행동가치를 추정하는 과정에서 그 값이 얼마나 정확한지를 떠나서 time step $t$에서 가장 큰 가치를 갖는 행동이 존재하게 된다. 이러한 행동을 **greedy action**이라 한다. 해당 시점에서 가장 큰 가치를 갖는 행동을 선택하는 것을 현재 알고있는 행동가치에 대한 지식을 **exploiting**한다고 한다. Nongreedy action을 선택하는 것은 **exploring**한다고 한다. 이미 확률분포에 대한 확신이 있다면 exploiting하는 것이 효율적이지만 불확실성이 크고 앞으로 많은 시도를 해볼 수 있다면 exploring을 통해 더 좋은 결과를 얻을 가능성을 탐색하는 것이 좋을 것이다. Exploring하는 동안 단기적으로는 보상이 적을 수 있지만 더 나은 행동을 찾게 된다면 새로운 행동을 exploit하면서 장기적으로 더 큰 보상을 기대할 수 있다. Exploitation과 exploration을 동시에 할 수는 없으므로 이 둘의 균형을 조절하는 것은 그 자체로 강화학습에서 중요한 문제이다.

교재에서는 기초적인 방식의 Balancing methods를 사용해 항상 exploitation을 사용하는 것 보다 balancing method를 쓰는게 더 좋은 결과를 보여준다는 것을 보여주며 복잡한 형태의 balancing method는 따로 다루지 않는다.

## Action-value Methods

행동가치를 추정하는 다양한 방식이 있다. 이러한 방식을 통틀어 **action-value methods** 라고 부른다. 행동가치란 해당 행동을 선택하였을 때 보상의 기댓값임을 상기하자.

그렇다면 행동가치는 어떻게 계산하는지에 대해 살펴보자. 다음은 sample-average method라는 방식으로 action value를 평균으로 간단하게 구하는 방법이다. 행동을 통해 얻은 보상을 해당 행동을 시행한 횟수로 나누면 나누면 얻을 수 있다.

> [!NOTE]
> **Definition: Sample-average Method**
>
> 행동가치를 추정하는 가장 간단한 방법으로 행동을 취했을 때 얻은 결과에 단순 평균을 취하는 방법을 생각해 볼 수 있다. 이러한 방식을 **Sample-average Method** 라고 한다.
> $$\begin{aligned}Q_t(a) &\doteq \frac{\text{sum of rewards when } a \text{ taken prior to }t}{\text{number of times } a \text{ taken prior to }t} \\ &= \frac{\sum_{i=1}^{t-1}  R_i \cdot \mathbb{1}_{A_i=a}}{\sum_{i=1}^{t-1} \mathbb{1}_{A_i=a}} \end{aligned}$$
> 위 식에서 $\mathbb{1}_{\text{predicate}}$은 predicate이 True일 때 1, 아닐 때 0인 random variable이다. 만약 분모가 0이라면 사전에 정의된 Default값을 value로 반환하며 분모가 무한히 커지면, 즉 time step $t$에서의 Action이 무한히 커지면 큰 수의 법칙(The law of large numbers)에 의해 $Q_t(a)$는 $q_{*}(a)$로 수렴한다.

행동가치를 값이 있을때, greedy한 선택은 다음과 같이 표현할 수 있다.

$$
A_t = \argmax_{a}Q_{t}(a)
$$

단순하게 행동가치가 가장 큰 행동 $a$를 선택하는 것이다. 하지만 앞서 언급되었듯, greedy한 방식만 사용하게 되면 더 나은 행동을 찾을 수 있는 기회를 갖지 못한다. 따라서 간단한 대안으로 사용할 수 있는 방식은 일정 확률 $\varepsilon$에 대해 추정한 행동가치와 상관없이 선택 가능한 행동들 중에서 임의로 행동을 선택하는 것이다. 이 방식의 장점은 단순하면서도 시행이 많아지게 되면 모든 행동들이 sampling되면서 $Q_t(a)$가 $q_{*}(a)$로 수렴하게 된다는 것이다. 이러한 방식을 $\boldsymbol{\varepsilon}$-**greedy** 방법이라고 부른다.

## The 10-armed Testbed

$\varepsilon$-greedy 방법이 greedy방법에 비해 정말 효과가 있는지를 알아보기 위해 다음과 같은 실험을 살펴보자.

2,000번 랜덤시행을 한 10-armed bandit이 있다고 해보자. 각기 다른 값이 설정된 슬롯머신 10대가 있는 것이다. 그리고 행동 각각 $a = 1, \ldots, 10$이라 하고 $q_{*}(a)$는 평균 0, 분산이 1인 정규분포로 선택된 값이다. $q_{*}$는 실제 행동가치로 agent는 알 수 없는 값임에 유의한다. agent는 회색밴드로 표현된 분포에 따라 보상을 받게 되며 이 값이 agent가 관측할 수 있는 보상이다. 이러한 테스트 상황을 **10-armed testbed**라고 부른다.

<figure align=center>
<img src="assets/images/Chapter02/Fig_2.1.png"/>
<figcaption>Figure2.1: An example bandit problem from the 10-armed testbed.</figcaption>
</figure>


이 문제에 대해 sample-average method의 $\varepsilon$을 0, 0.01, 0.1로 다르게 적용해 각각 2000번 시행한 평균은 다음과 같다.

<figure align=center>
<img src="assets/images/Chapter02/Fig_2.2.png"/>
<figcaption>Figure2.2: Average performance of $\epsilon$-greedy action-value methods on the 10-armed testbed.</figcaption>
</figure>

먼저 greedy한 전략을 썼을 떄의 평균 보상을 보면 1에 도착하고 증가하지 않는다. 행동 3의 기댓값이 1.55로 행동 3을 선택하는 최고의 전략에 비할 때 greedy한 agent는 suboptimal에 최적화 되었음을 알 수 있다. 아래의 최적행동비율(여기서는 행동 3을 선택한 경우)를 보더라도 greedy agent는 33%정도에 머무르고 있다. 대조적으로 $\varepsilon$-greedy agent는 모두 점진적으로 평균 보상과 최적행동비율이 향상되는 것을 볼 수 있다. (물론 무한정 좋아지지는 않는다. 0.1로 설정된 경우 91% 확률로 최적행동을 선택한다) 충분히 exploration이 일어났다면 $\varepsilon$을 점차 줄여 더 높은 return을 기대할 수도 있다.

$\varepsilon$-greedy의 선택은 testbed의 노이즈에 따라 다르게 선택될 수 있다. 지금은 testbed가 평균 0, 분산이 1이었지만, 분산이 10이었다면 더 많은 exploration이 필요했을 것이고 $\varepsilon$-greedy와 greedy의 차이는 더 커졌을 것이다. 하지만 분산이 0이었다면 greedy는 시행 즉시 $q_{*}$를 알 수있게 될 것이므로 $\varepsilon$-greedy가 필요없는 상황이 된다. 하지만 이런 deterministic한 상황이라 하더라도 $q_{*}$가 조금씩 변하는 non-stationary한 상황이라면 nongreedy action이 기존의 greedy action보다 높은 return을 제공할 가능성이 있으므로 exploration이 필요하다.


## Incremental implementation

단순하지만 행동가치를 추정하는 방법인 sample-average method를 보았다. 이번에는 제한된 자원만 있는 상황에서 어떻게 sample-average method를 적용할 수 있는지, 그리고 실제 행동가치가 고정된채로 유지되는 것이 아니라 변화하는 환경에서는 어떻게 대응할지에 대해 알아보자.

Incremental implemenation에서 다루는 내용은 간단하게 말하면 이동평균으로 평균을 구하는 방식이다. 어떤 armed bandit의 행동가치를 다음과 같이 정의해보자.

$$ Q_{n} \doteq \frac{R_{1} + R_{2} + \cdots + R_{n-1}}{n - 1} $$

위 식은 $n-1$ time step까지의 보상 평균에 해당하는 값으로 보상을 모두 들고다닐 필요 없이 평균값 하나만 저장하면 되어 많은 메모리공간을 차지하지 않는다. 아래 유도는 이전까지의 행동가치와 현재 보상, 이 두 정보만 알면 행동가치를 update할 수 있음을 보인다.

$$
\begin{aligned}
Q_{n+1} &= \frac{1}{n} \sum_{i=1}^{n} R_{i} \\
&= \frac{1}{n} \left( R_{n} + \sum_{i=1}^{n-1} R_{i} \right) \\
&= \frac{1}{n} \left( R_{n} + (n-1) \frac{1}{n-1} \sum_{i=1}^{n-1} R_{i} \right) \\
&= \frac{1}{n} \left( R_{n} + (n-1) Q_{n} \right) \\
&= \frac{1}{n} \left( R_{n} + n Q_{n} - Q_{n} \right) \\
&= Q_{n} + \frac{1}{n} \left[ R_{n} - Q_{n} \right]
\end{aligned}
$$

행동가치를 위와 같이 update 할 수 있다는 것은 간단하게 확인이 되지만 더 중요한 것은 위와 같은 update의 형태이다. 이 책 전반에 걸쳐서 다음과 같은 update 방식은 매우 자주 등장한다.

$$ \operatorname{NewEstimate} \leftarrow \operatorname{OldEstimate} + \operatorname{StepSize} \left[ \operatorname{Target} - \operatorname{OldEstimate} \right] $$

여기서 $ \left[ \operatorname{Target} - \operatorname{OldEstimate} \right] $부분을 추정에 대한 **error**라고 한다. 또한 $\operatorname{StepSize}$는 time step이 증가할수록 점점 작아지는 변화하는 값임에 유의하자.($n$번째 행동에 대한 보상은 $\operatorname{StepSize}$가 $\frac{1}{n}$만큼 적용된다) 이후 책에서는 $ \operatorname{StepSize} $를 $\alpha$, 또는 $\alpha_{t}(a)$로 표기한다.

### Simple Bandit Algorithm

Sample average method와 $\varepsilon$-greedy을 사용해 armed-bandit의 행동가치를 추정하는 bandit 알고리즘의 pseudocode는 다음과 같다.

<figure align=center>
<img src="assets/images/Chapter02/simple_bandit_algorithm.png" width=70% height=70%/>
<figcaption>Simple Bandit Algorithm</figcaption>
</figure>

우선 모든 행동 대해 행동가치와 시행횟수는 0으로 초기화가 된다. 그리고 시행이 발생할때 마다 loop을 돌면서 값을 update 하는 구조이다. $\varepsilon$-greedy이므로 $1-\varepsilon$의 확률로 행동가치가 가장 큰 행동을 선택하고 $\varepsilon$의 확률로 무작위 행동을 선택한다. 선택한 행동으로 bandit의 레버를 당기면 보상이 주어지고 해당 게임은 끝나게 된다. (후에 다루겠지만 이러한 성질로 인해 bandit은 1-step MDP로 볼 수 있다) 시행을 했으니 counter 역할을 하는 $N(A)$를 1 올려주고 위의 식을 이용해 행동가치를 update해주면 된다.

## Reference

* [Sutton, R. S., Barto, A. G. (2018). Reinforcement learning: An introduction. Cambridge, MA: The MIT Press.](http://www.incompleteideas.net/book/the-book-2nd.html)