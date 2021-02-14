# Chapter 05: Monte Carlo Methods

Monte Carlo는 보통 구하기 어려운 통계량을 sampling을 통해 추정하는 맥락에서 등장한다. 따라서 Monte Carlo라는 이름에서 알 수 있듯, 이번 문서에서는 Monte Carlo를 강화학습에 사용하는 방법들을 살펴본다.

앞서 dynamic programming은 MDP에 대한 완전한 정보를 가지고 있을 때 적용가능한 방식이었다. 하지만 많은 경우 우리는 환경에 대한 정보(MDP)를 완전히 알지 못한다. 애초에 Monte Carlo를 사용하는 자체가 원하는 값을 몰라 통계량을 **추정**해야하는 상황이므로 Monte Carlo는 MDP를 몰라도 사용할 수 있는 방법일 것임을 짐작해 볼 수 있다. Monte Carlo는 DP의 접근방법과는 다르게 경험만을 필요로 한다. 즉, 상태, 행동, 보상의 과정인 trajectory만 있으면 된다. 가장 두드러지는 차이 중 하나는 환경의 transition probability matrix를 필요로 하는 DP와는 다르게 MDP는 경험 그 자체만을 사용하므로 더 범용적으로 사용할 수 있게 된다.

이러한 장점은 Monte Carlo를 사용해 transition probability matrix라는 확률분포를 추정한다는 관점으로도 볼 수 있을 것이다. MDP를 완전히 알고 있다면 transition probability distribution을 완벽하게 알고 있으므로 이를 바로 사용하여 풀면 된다. 하지만 많은 경우 우리는 transition probability matrix정보에 접근할 수가 없고, 다만 경험을 쌓는 것은 transtion probability matrix를 직접 찾으려는 것보다 훨씬 용이하게 수행할 수 있다. 따라서 쌓아둔 경험을 이용해 transiton probability matrix를 찾는 것은 sampling(경험)을 통해 통계량(transition probability matrix)을 추정하는 Monte Carlo의 방식에 딱 맞아 떨어진다.

MC방식은 강화학습의 문제를 sample return의 기대값을 이용해 접근한다. Return은 앞으로 받을 수 있는 보상의 총 합이다. 여기서 한 가지 제약이 생기게 되는데 앞으로 얼마만큼의 보상을 받을지를 계산해야 sample return을 만들 수 있으므로 return을 계산하기 위해서는 보상이 유한한 시점에 끝나야 한다. 즉 episodic task여야 한다는 것이다. 무한히 계속되는 continuous task라면 return을 계산하기위해 계속 기다려야 하는데 끝이 나지 않으므로 당장 sample return값을 얻는 것부터 어려워진다. 따라서 끝을 보고 sample return을 만들어 낼 수 있는 episodic task에 한정해서 MC방법을 사용하게 된다.

이러한 특성으로 인해 MC방법은 적어도 episode가 끝나야 가치함수와 정책에 대해 update를 수행할 수 있다. 후에 다루겠지만 이는 각 step마다 upate가 가능한 temporal difference방식과 대비되는 차이점이다. 강화학습에서의 Monte Carlo는 이처럼 **완전한(complete)** return의 평균에 기반한 학습방식을 의미한다.

MDP가 주어진 DP에서는 가치함수를 MDP를 이용해 계산할 수 있었지만 MDP에 대한 정보를 모를 때는 가치함수를 sample return을 통해 학습을 해야한다는 큰 차이가 있다. 이렇게 학습한 가치함수와 정책에 GPI의 아이디어를 사용해 최적가치/정책을 추정할 수 있게 된다. 이번 단원에서도 DP에서와 마찬가지로 주어진 정책을 평가하는 정책평가단계와 정책개선단계로 나누어 문제를 접근한다. 하지만 이번 단원에서는 Monte Carlo 방법을 사용해 과정을 수행하게 된다.

## 5.1 Monte Carlo Prediction

Prediction은 주어진 정책을 평가하는 과정이며 이 평가는 가치함수를 추정하는 과정이 된다. 상태가치의 정의를 상기해보면 해당 상태에서 앞으로 받을 return의 기대값으로 정의가 되었다. 따라서 가장 직관적인 추정은 경험을 통해 해당 상태이후에 받은 보상들을 모두 더한 값들의 평균을 구하는 것이다. 이러한 경험이 많아질수록 더 정확한 값으로 수렴을 할 것이라는 것이 Monte Carlo 방법의 기본적인 아이디어이다.

상태가치 $v_{\pi}(s)$가 의미하는 바는 상태 $s$에서 정책 $\pi$를 따라 episode를 진행했을 때 받을 return의 기대값이다. 이 때, episode가 진행되는 동안 방문한 상태들을 $s$의 **visit**이라고 한다. 물론 각각의 상태는 episode가 진행되면서 여러차례 방문할 수 있으며 최초의 방문에 한정해서 $s$의 first visit이라고 해보자. First visit을 따로 정의하는 이유는 다음과 같은 두 가지 방법을 생각해 볼 수 있기 때문이다.

* first-visit MC method
* every-visit MC method

First-visit MC method에서는 $s$의 최초 방문을 기준으로 return의 평균을 구해 $v_{\pi}(s)$를 추정한다. 반면, every-visit MC method에서는 이후에 방문하는 $s$들 각각에 대한 return의 평균을 통해 가치를 추정한다. 이번 문서에서는 first-visit MC method를 주로 다루게 되며 every-visit MC method에 대해서는 function approximation과 eligibility trace라는 개념과 함께 이후에 다루게 된다. 아래 pseudocode에서 every-visit은 fist-visit과 대부분 같으나 마지막에서 세번째 줄인 $S_{t}$가 이미 평가되었는지를 확인하는 조건이 없다는 차이가 있다.

<figure align=center>
<img src="assets/images/Chapter05/fv-mc-pred.png"/>
<figcaption>First-visit MC prediction</figcaption>
</figure>

두 방식 모두 무한히 많은 경우를 반복하게 되면 큰 수의 법칙(the law of large numbers)에 의해 실제 값인 $v_{\pi}(s)$로 수렴하게 된다. 가치함수의 정의가 $v_{\pi}(s_{t}) = \mathbb{E}_{\pi} [G_{t} \mid s = s_{t}]$이므로 각각의 상태에서 return을 구하는 것은 unbiased estimate이 되고, 무수히 많은 episode에 대해 이를 반복하게 되면 실제 return으로 수렴시킬 수 있다.

Monte Carlo 방법은 가장 쉽게 추정치를 얻을 수 있는 방법이다. 그런데 만약 MDP에 대한 완전한 정보를 알고 있으면 DP를 사용하는 것이 무조건 유리할까? 꼭 그렇지는 않다. MDP, 특히 transition probability matrix를 알고 있다고 하더라도 episode가 진행되는 동안 각각의 확률을 반영해 미리 계산하는 것은 쉽지 않다. 후에 다루겠지만 Monte Carlo는 bootstrap관점에서 backup diagram을 수직으로 끝까지 깊이 탐색하는 방법이라면 DP는 backup diagram 기준으로 width 방향으로 모두 계산해보아야 하는 과정이다. 따라서 MDP를 안다고 하더라도 DP에 필요한 확률을 모두 반영하는 것은 여전히 간단한 문제가 아니며 그 과정에서 확률이 연쇄적으로 만들어내는 오차와 복잡도도 커지게 된다. Monte Carlo는 단지 쉽게 추정치를 얻는 방법이 아니라 그 자체로도 강력한 추정 기법이 된다.

그렇다면 Monte Carlo 방법의 backup diagram은 어떻게 그려질까? 말 그대로 해보고 확인하는 방법인 만큼 시작 state에서 시작해서 terminal state까지의 trajectory를 한 줄로 그려낸 형태가 된다.

<figure align=center>
<img src="assets/images/Chapter05/mc_backup_diagram.png"/>
<figcaption>Backup diagram of Monte Carlo Methods</figcaption>
</figure>

이는 앞서 본 DP의 backup diagram과 대비된다. DP에서는 가능한 모든 transition을 고려해야하므로 가능한 상태와 행동이 모두 표시된다. 반면 Monte Carlo는 경험한 하나의 trajectory만 표시된다. 또한 DP는 one-step transition, 즉 상태에서 가능한 행동과 state-action pair에 대한 transition probability matrix에 의해 도착하는 다음 상태까지 표현되는 반면, Monte Carlo diagram에서는 terminal state까지 하나의 완전한 episode가 표시된다. Backup diagram에서 볼 수 있는 알고리즘의 차이를 알아두는 것은 중요하다.

또한 Monte Carlo의 중요한 성질 중 하나는 각각의 상태에 대한 추정값은 독립적이라는 것이다. Monte Carlo methods에서 각각의 trajectory는 독립적으로 수행된 일련의 과정들로 다른 추정치와는 독립적이다. 앞의 단원에서 다루었듯, DP에서는 다음 상태의 추정치를 사용해 현재 상태를 update하므로 DP는 다른 상태의 추정치에 대해 독립적이라고 할 수 없다.

이는 한 상태의 가치를 추정한다고 할 때 terminal state까지 몇 개의 상태가 있든지 상관없다는 것을 의미한다. 최종 return을 알았으면 시작한 상태의 가치는 바로 추정하면 된다. 그리고 동일한 상태에서 여러 episode를 끝까지 진행한 뒤에 평균을 내면 더 정확한 추정을 할 수 있고 이게 Monte Carlo의 핵심 아이디어이다.