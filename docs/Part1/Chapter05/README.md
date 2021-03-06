# Chapter 05: Monte Carlo Methods

Monte Carlo는 보통 구하기 어려운 통계량을 sampling을 통해 추정하는 맥락에서 등장한다. 따라서 Monte Carlo라는 이름에서 알 수 있듯, 이번 문서에서는 Monte Carlo를 강화학습에 사용하는 방법들을 살펴본다.

앞서 dynamic programming은 MDP에 대한 완전한 정보를 가지고 있을 때 적용가능한 방식이었다. 하지만 많은 경우 우리는 환경에 대한 정보(MDP)를 완전히 알지 못한다. 애초에 Monte Carlo를 사용하는 자체가 원하는 값을 몰라 통계량을 **추정**해야하는 상황이므로 Monte Carlo는 MDP를 몰라도 사용할 수 있는 방법일 것임을 짐작해 볼 수 있다. Monte Carlo는 DP의 접근방법과는 다르게 경험만을 필요로 한다. 즉, 상태, 행동, 보상의 과정인 trajectory만 있으면 된다. 가장 두드러지는 차이 중 하나는 환경의 transition probability matrix를 필요로 하는 DP와는 다르게 MDP는 경험 그 자체만을 사용하므로 더 범용적으로 사용할 수 있게 된다.

이러한 장점은 Monte Carlo를 사용해 transition probability matrix라는 확률분포를 추정한다는 관점으로도 볼 수 있을 것이다. MDP를 완전히 알고 있다면 transition probability distribution을 완벽하게 알고 있으므로 이를 바로 사용하여 풀면 된다. 하지만 많은 경우 우리는 transition probability matrix정보에 접근할 수가 없고, 다만 경험을 쌓는 것은 transtion probability matrix를 직접 찾으려는 것보다 훨씬 용이하게 수행할 수 있다. 따라서 쌓아둔 경험을 이용해 transiton probability matrix를 찾는 것은 sampling(경험)을 통해 통계량(transition probability matrix)을 추정하는 Monte Carlo의 방식에 딱 맞아 떨어진다.

MC방식은 강화학습의 문제를 sample return의 기대값을 이용해 접근한다. Return은 앞으로 받을 수 있는 보상의 총 합이다. 여기서 한 가지 제약이 생기게 되는데 앞으로 얼마만큼의 보상을 받을지를 계산해야 sample return을 만들 수 있으므로 return을 계산하기 위해서는 보상이 유한한 시점에 끝나야 한다. 즉 episodic task여야 한다는 것이다. 무한히 계속되는 continuous task라면 return을 계산하기위해 계속 기다려야 하는데 끝이 나지 않으므로 당장 sample return값을 얻는 것부터 어려워진다. 따라서 끝을 보고 sample return을 만들어 낼 수 있는 episodic task에 한정해서 MC방법을 사용하게 된다.

이러한 특성으로 인해 MC방법은 적어도 episode가 끝나야 가치함수와 정책에 대해 update를 수행할 수 있다. 후에 다루겠지만 이는 각 step마다 upate가 가능한 temporal difference방식과 대비되는 차이점이다. 강화학습에서의 Monte Carlo는 이처럼 **완전한(complete)** return의 평균에 기반한 학습방식을 의미한다.

MDP가 주어진 DP에서는 가치함수를 MDP를 이용해 계산할 수 있었지만 MDP에 대한 정보를 모를 때는 가치함수를 sample return을 통해 학습을 해야한다는 큰 차이가 있다. 이렇게 학습한 가치함수와 정책에 GPI의 아이디어를 사용해 최적가치/정책을 추정할 수 있게 된다. 이번 단원에서도 DP에서와 마찬가지로 주어진 정책을 평가하는 정책평가단계와 정책개선단계로 나누어 문제를 접근한다. 하지만 이번 단원에서는 Monte Carlo 방법을 사용해 과정을 수행하게 된다.

## Monte Carlo Prediction

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

## Monte Carlo Estimation of Action Values

Model을 모르는 상황에서는 state action pair를 통해서 얻는 action value가 유용하다. Backup diagram을 상태가치에서 생각해보면, 현재상태에서 정책에 의한 행동, 그리고 transition probability matrix에 의해 다음 상태로 연결이 된다. 하지만 model을 모른다면 transition probability matrix를 모르므로 행동을 선택할 때, 다음상태가 어디로 될지 알 수 없다. 하지만 action value $Q(s, a)$는 state action pair에 대해서 결과를 얻을 수 있으므로 transition probability matrix를 몰라도 사용할 수 있다. **따라서 Monte Carlo 방법의 목표는 최적 행동가치함수 $q_{*}$을 찾는 것이다.** 이를 위해서는 우선 정책평가를 어떻게 할 지를 정해야 한다.

정책평가에서는 정책이 주어지고 이 정책을 따랐을 때의 가치함수를 구하게 된다. 정책이 주어졌으므로 이 정책을 따랐을 때의 $q_{\pi}(s,a)$를 구해야 한다. 한 episode를 끝까지 진행하고 나면 episode가 진행되는 동안의 trajectory를 볼 수 있으므로 각 상태에서 선택한 행동들을 모을 수 있게 된다. 여기서 first-visit MC와 every-visit MC가 살짝 달라지는 부분이 생긴다. First-visit MC는 최초로 방문한 상태에서의 행동을 기준으로 return에 대해 평균을 내고 every-visit MC에서는 상태를 방문했던 모든 경우에 대해서 평균을 계산한다. 두 가지 방식 모두 무수히 많은 episode를 통해 계산해 나아가면 unbiased estimate이므로 실제 값으로 수렴하게 된다.

여기서 exploration관점에서 생각해볼 문제가 있다. 만약 최초의 정책평가가 MC로 이루어진 뒤 정책개선이 발생한다고 해보자. 이 때 정책이 deterministic policy라면, 즉 정책이 가장 높은 Q-value를 주는 action만 선택한다면 최초에 초기화된 상태에서 최적정책이 아님에도 불구하고 더 높은 가치를 갖는 행동만을 선택해 다른 state-action pair에 대해서는 시도조차하지 않는 문제가 발생한다. 특히, 초반부에는 exploration을 적극적으로 해야하는데 이는 심각한 문제가 된다. 강화학습에서 deterministic한 정책이 탐색을 하지 못하는 문제를 **problem of maintaining exploration**이라고 한다.

Exploration을 충분히 일어날 수 있게 하는 것은 강화학습에서 중요하게 고려해야 하는 부분으로 가장 흔하게 사용되는 방법으로는 각 상태에서 모든 행동들이 선택될 가능성이 열려있도록 stochastic하게 정책을 만드는 것이다.

## Monte Carlo Control

이제 MC estimation을 control에 적용해보자. Coontrol인 만큼 optimal policy를 찾는 것이 목표이다. 접근 자체는 DP에서 다루었던 GPI와 같다.

<figure align=center>
<img src="assets/images/Chapter05/mc_gpi.png" width=30% height=30%/>
<figcaption></figcaption>
</figure>

GPI는 어떤 정책과 가치함수가 있을 때, 정책에 대한 정책평가를 통해 가치함수를 학습하고 학습된 가치함수를 통해 정책을 개선하는 과정을 반복하면 최적정책과 최적가치함수로 나아갈 수 있음을 말한다.

이번 문서는 MC에 대해 다루므로 이제 GPI에 MC가 어떻게 적용되는지에 초점을 두고 알아보자. 초기정책 $\pi_{0}$가 있다고 할때, GPI의 과정은 다음과 같은 반복이 이루어진다.

$$
\pi_{0} \stackrel{\mathrm{E}}{\longrightarrow} q_{\pi_{0}} \stackrel{\mathrm{I}}{\longrightarrow} \pi_{1} \stackrel{\mathrm{E}}{\longrightarrow} q_{\pi_{1}} \stackrel{\mathrm{I}}{\longrightarrow} \pi_{2} \stackrel{\mathrm{E}}{\longrightarrow} \cdots \stackrel{\mathrm{I}}{\longrightarrow} \pi_{*} \stackrel{\mathrm{E}}{\longrightarrow} q_{*}
$$

정책평가단계에서는 MC Prediction을 사용하게 된다. 완료된 수 많은 episode를 사용해서 각 상태의 가치를 평가할 수 있다. 이러한 반복이 무한히 많아지게 되면 점근적으로 실제 가치함수에 가까워지게 된다. 이론적으로만 가능하지만 주어진 정책에 대해 무한히 많은 episode를 사용해 평가했다고 한다면 정책 $\pi_{k}$에 대한 실제 가치함수 $q_{\pi_k}$를 정확하게 계산할 수 있다.

정책개선은 앞서 평가한 가치함수를 사용해 간단한 greedy policy를 만드는 방법이 있다. 이 때, transition probability matrix를 모르므로 행동가치함수인 $q_{\pi}$를 사용한다. 즉, 어떤 상태 $s$에서 선택하는 행동은 다음과 같이 결정된다.
$$\pi_(s) \doteq \argmax_{a} q(s,a)$$
개선된 정책 $\pi_{k+1}$은 행동가치함수 $q_{\pi_k}$에 대한 greedy policy로 선택한다. Policy improvement theorem에 의해 다음이 성립한다.
$$
\begin{aligned}
q_{\pi_{k}}\left(s, \pi_{k+1}(s)\right) &=q_{\pi_{k}}\left(s, \underset{a}{\arg \max } q_{\pi_{k}}(s, a)\right) \\
&=\max _{a} q_{\pi_{k}}(s, a) \\
& \geq q_{\pi_{k}}\left(s, \pi_{k}(s)\right) \\
& \geq v_{\pi_{k}}(s)
\end{aligned}
$$
따라서 $\pi_{k+1}$은 $\pi_{k}$보다 최소한 같거나 더 좋다는 것이 보장된다. 이러한 성질로 인해 GPI에 따라 시행하면 최적정책과 최적가치로 수렴할 수 있다. 또한 MC methods가 환경에 대한 dynamics를 전혀 모르더라도 sample episode를 활용해 최적정책을 찾을 수 있는 이론적 뒷받침이 된다.

알고리즘을 실제로 적용하기위해서는 정책평가단계에서 무한히 많은 정책평가를 반복하는 걸 현실적인 방법으로 바꾸어야 한다. 가장 간단한 방법 중 하나는 정책평가의 수렴을 간접적으로 확인하는 것으로 추정하는 가치함수 $q_{\pi_k}$의 변화폭이 미리 지정한 매우 작은 값 이하로 될 때까지 반복하는 것이다. 실제로 이렇게 하면 거의 수렴한 상태를 유지할 수 있다는 장점이 있지만 작은 문제에 대해서도 정책평가과정에 들어가는 연산이 많아 최적정책까지 가는데 오래걸리고 실제로 사용할 수준이 되기위해서는 매우 많은 episode에 대해 적용해야한다는 문제가 있다.

다른 방법은 DP에서 다룬 것과 동일한 GPI 방식을 사용하는 것이다. 정책평가를 정해진 반복횟수 만큼만 돌리고 정책개선을 함으로써 추정된 가치함수는 부정확하더라도 더 효율적으로 최적정책을 향해 나아갈 수 있으며 극단적으로 정책평가를 1회만 하고 정책개선을 하는 value iteration도 이러한 방법 중 하나이다. In-place 방식도 생각해 볼 수 있는데, value iteration이 1회를 평가하더라도 모든 상태들에 대해 1회 평가했던 것에 반해 in-place 방식에서는 정책평가와 개선을 모든 상태가 아니라 하나의 상태에 대해 진행한다는 차이가 있다.

MC policy iteration은 기본구조는 이름에서 나타나듯 policy iteration과 같다. 다만 MC 방법을 이용하므로 한 episode가 끝나야 trajectory에 대한 return을 사용할 수 있으므로 epsode단위로 정책평가와 개선이 이루어진다. Monte Carlo with Exploring Starts라고 부르는 이 방식의 pseudocode는 다음과 같다.

<figure align=center>
<img src="assets/images/Chapter05/mc-es.png" width=60% height=60%/>
<figcaption></figcaption>
</figure>

Pseudocode를 자세히 살펴보자. 우선 정책 $\pi$와 행동가치함수 $Q$는 임의의 값으로 초기화 하고 $Q(s,a)$를 저장할 list를 준비한다. Episode마다 반복문을 실행한다. 초기 상태와 행동은은 임의로 sampling한다. 이렇게 얻은 초기 상태와 행동 $S_0$, $A_0$를 정책 $\pi$를 따라가면 trajectory를 생성한다. $T$ step 만큼 진행되고 episode가 끝났다면 $\pi: S_0, A_0, R_1, \ldots, S_{T-1}, A_{T-1}, R_{T}$와 같은 trajectory를 얻게 된다. 이제 이 trajectory를 사용해 정책평가와 개선을 진행한다. Trajectory를 역방향으로 roll out하면서 각 timestep별로 return을 계산한다. Trajectory에서 $S_t, A_t$가 나오지 않는 한 $G$에 $S_t, A_t$ pair의 return을 append하고 해당 state-action pair의 return에 있는 값들의 평균을 사용해 $Q(S_t, A_t)$를 update한다. 그리고 상태가치가 update되었으므로 정책은 update된 상태가치에서 greedy하게 작동하도록 $\argmax_{a}Q(S_t, a)$로 바꾸어 준다. 알고리즘에 exploring starts가 붙은 이유는 0이상의 확률을 갖는 모든 state-action pair가 시작점이 될 수 있기 때문이다.

## Monte Carlo Control without Exploring Starts

앞에서 exploring start를 가정했지만 이는 현실적으로 사용하기 어려운 방식이다. 학습을 위해서는 모든 행동들이 충분히 선택되어 평가될 수 있어야 하는데 이를 위한 방법은 크게 **on-policy**와 **off-policy** 방법 두 가지가 있다. On-policy는 평가 또는 개선하는 정책이 행동을 결정하는 정책과 같은 경우를 말한다. 다시 말해, behavior policy와 target policy가 같은 경우이다. Off policy는 반대로 behvaior policy와 target policy가 다른 경우를 말한다.

앞에서 다룬 Monte Carlo ES 방법은 on-policy 방법에 해당한다. 여기서는 우선 Monte Carlo ES에서 비현실적인 ES를 떼어내는 과정을 살펴본다.

On-policy control에서 정책은 일반적으로 soft하다. Soft 정책의 의미는 모든 정책함수의 결과가 모든 상태, 행동에 대해서 양수를 갖는 경우이다. $(\pi (a \mid s))$ 모든 행동이 선택될 확률이 열려있는 것이다. 그리고 학습이 진행되면서 최적정책쪽으로 정책확률분포가 이동하게 될 것이다. 여기서 제시하는 on-policy 방법은 $\epsilon$-greedy 정책을 사용한다. $\epsilon$의 확률로 random action을 선택하고 $(1-\epsilon)$의 확률로 추저한 행동가치에 대해 greedy한 선택을 한다. 이러한 방식은 매우 간단한 방법이지만 다양한 환경에서 꽤나 유용한 정책임이 확인되었다. 즉, nongreedy로 행동을 선택할 때는 모든 행동공간에서 선택될 가능성이 열려있고 최소한 $\frac{\epsilon}{\lvert \mathcal{A}(\boldsymbol{s}) \rvert}$의 확률은 선택될 가능성이 보장된다. 그리고 greedy한 선택을 하면 nongreedy의 경우까지 포함해 $1-\epsilon + \frac{\epsilon}{\lvert \mathcal{A} (\boldsymbol{s})\rvert}$의 확률을 갖게 된다. $\epsilon$-greedy는 $\epsilon$-soft에 속하는 방법으로 $\pi(a \mid s) \geq \frac{\varepsilon}{|\mathcal{A}(s)|}$를 모든 상태와 공간에 대해 보장해준다.

On-policy Monte Carlo control도 기본적으로는 GPI의 아이디어를 따른다. 여기서는 first-visit MC부터 소개한다. 앞의 MC with ES는 exploring start가 exploration을 보장해주었지만 지금은 ES부분을 떼어내는 것이 목적이므로 ES가 해주던 exploration역할을 해줄 수 있는 대체재를 사용해야한다. 이 대체재로서 위의 $\epsilon$-greedy를 사용하면 exploration을 보장할 수 있게 된다. Pseudocode는 다음과 같다.

<figure align=center>
<img src="assets/images/Chapter05/on-policy_mc_control.png" width=100% height=100%/>
<figcaption></figcaption>
</figure>

Hyperparameter로 exploration할 확률 $\epsilon$을 정의하고 정책은 $\epsilon$-soft로, $Q(s,a)$는 임의의 값으로 초기화한다. Return을 저장할 리스트도 준비한다. MC 방법인 만큼 episode전체를 진행한 뒤 업디이트가 이루어진다. 앞서 정의한 $\epsilon$-soft에 의해 episode를 진행해 trajectory를 얻는다. 앞의 MC방법과 마찬가지로 마지막 상태에서부터 역순으로 return을 계산하기 시작한다. 따라서 역순으로 올 때 timestep $t$에서의 return은 $G \leftarrow \gamma G+R_{t+1}$이다. First-visit MC이므로 $S_{t}, A_{t}$가 trajectory에서 등장할지 않을때까지 이어지는 과정을 반복한다. 계산한 return $G$를 시작할 때 정의한 return table에서 $S_{t}, A_{t}$칸에 기록한다. 그리고 MC이므로 $Q(S_{t}, A_{t})$의 값은 return table의 $(S_t, A_t)$에 해당하는 값들의 평균으로 추정한다. 그리고 timestep $t$의 상태에서 방금 update한 $Q$를 이용해 가장 높은 행동가치를 제공하는 행동을 $A^{*}$로 assign한다. 이제 정책을 update하게 되는데 timestep $t$의 상태에 대한 모든 action을 다음의 규칙에 따라 확률을 정의한다.
$$\pi\left(a \mid S_{t}\right) \leftarrow\left\{\begin{array}{ll}
1-\varepsilon+\varepsilon /\left|\mathcal{A}\left(S_{t}\right)\right| & \text { if } a=A^{*} \\
\varepsilon /\left|\mathcal{A}\left(S_{t}\right)\right| & \text { if } a \neq A^{*}
\end{array}\right.$$
가장 높은 행동가치를 제공한 행동이외에는 $\epsilon/\lvert \mathcal{A}(S_t) \rvert$의 확률을 나누어 갖게 된다.

Policy improvement theorem에 의해 $\pi$에 대해 $\epsilon$-soft 방식이 $q_{\pi}$를 개선하는 것을 보장할 수 있다. $\pi^{\prime}$이 $\epsilon$-greedy라고 해보자. $q_{\pi}(s, \pi^{\prime}(s))$에서 정책 $\pi{\prime}$은 확률적으로 정의된다. 모든 가능한 행동에 대해서 행동가치는 다음과 같이 기댓값의 형태로 표현이 가능하다.

$$q_{\pi}\left(s, \pi^{\prime}(s)\right) =\sum_{a} \pi^{\prime}(a \mid s) q_{\pi}(s, a)$$

앞서 다룬 바에 따라 다음도 성립한다.

$$
\begin{aligned}
q_{\pi}\left(s, \pi^{\prime}(s)\right) &=\sum_{a} \pi^{\prime}(a \mid s) q_{\pi}(s, a) \\
&=\frac{\varepsilon}{|\mathcal{A}(s)|} \sum_{a} q_{\pi}(s, a)+(1-\varepsilon) \max _{a} q_{\pi}(s, a) \tag{5.2} \\
& \geq \frac{\varepsilon}{|\mathcal{A}(s)|} \sum_{a} q_{\pi}(s, a)+(1-\varepsilon) \sum_{a} \frac{\pi(a \mid s)-\frac{\varepsilon}{|\mathcal{A}(s)|}}{1-\varepsilon} q_{\pi}(s, a) \\
&=\frac{\varepsilon}{|\mathcal{A}(s)|} \sum_{a} q_{\pi}(s, a)-\frac{\varepsilon}{|\mathcal{A}(s)|} \sum_{a} q_{\pi}(s, a)+\sum_{a} \pi(a \mid s) q_{\pi}(s, a) \\
&=v_{\pi}(s)
\end{aligned}
$$

따라서 policy imporvement theorem에 의해 $\pi^{\prime} \geq \pi$로 개선을 보장하게 된다. 등호는 최적정책에 도달했을 때이다. 이 등호조건을 증명해보자. 등호조건은 $\pi^{\prime}$과 $\pi$가 $\epsilon$-soft 정책중에서 최적정책일 떄에 한해서 성립한다.

기존환경과 동일한 새로운 환경이 있다고 해보자. 새로운 환경은 기존환경과 동일한 상태와 행동을 갖고 있다고 하자. 이 떄 새로운 환경에서 할 수 있는 최선은 기존환경의 정책을 $\epsilon$-soft로 사용하는 것이다. 새로운 환경에서의 최적가치함수를 각각 $\tilde{v_{*}}$,, $\tilde{q_{*}}$라고 하면 정책 $\pi$는 모든 $\epsilon$-soft 정책들 중에서 $v_{\pi} = \tilde{v_{*}}$일 때 최적정책이 된다. 상태가치 정의에 의해 $\tilde{v_{*}}$는 다음과 같다.
$$\begin{aligned}
\widetilde{v}_{*}(s)=&(1-\varepsilon) \max _{a} \widetilde{q}_{*}(s, a)+\frac{\varepsilon}{|\mathcal{A}(s)|} \sum_{a} \widetilde{q}_{*}(s, a) \\
=&(1-\varepsilon) \max _{a} \sum_{s^{\prime}, r} p\left(s^{\prime}, r \mid s, a\right)\left[r+\gamma \widetilde{v}_{*}\left(s^{\prime}\right)\right] \\
+& \frac{\varepsilon}{|\mathcal{A}(s)|} \sum_{a} \sum_{s^{\prime}, r} p\left(s^{\prime}, r \mid s, a\right)\left[r+\gamma \widetilde{v}_{*}\left(s^{\prime}\right)\right]
\end{aligned}$$
등호는 $\epsilon$-soft 정책 $\pi$가 더 이상 개선되지 않을 때이다. 따라서 가치함수가 최적함수이므로 다음이 성립하며 이는 (5.2)에 의해 다음과 같다.
$$
\begin{aligned}
v_{\pi}(s)=&(1-\varepsilon) \max _{a} q_{\pi}(s, a)+\frac{\varepsilon}{|\mathcal{A}(s)|} \sum_{a} q_{\pi}(s, a) \\
=&(1-\varepsilon) \max _{a} \sum_{s^{\prime}, r} p\left(s^{\prime}, r \mid s, a\right)\left[r+\gamma v_{\pi}\left(s^{\prime}\right)\right] \\
+& \frac{\varepsilon}{|\mathcal{A}(s)|} \sum_{a} \sum_{s^{\prime}, r} p\left(s^{\prime}, r \mid s, a\right)\left[r+\gamma v_{\pi}\left(s^{\prime}\right)\right]
\end{aligned}
$$
최적정책은 유일하므로 $v_{\pi} = \tilde{v_{*}}$이다.

정리하면, $\epsilon$-soft를 정책으로 사용할 때 policy iteration으로 사용할 수 있고 정책의 개선도 보장된다는 것이 핵심이다. $\epsilon$-soft를 사용함으로써 exploring starts를 하지 않아도 된다는 점이 장점이다. 

## Off-policy Prediction via Importance Sampling

모든 control 방법은 다음 두 가지를 동시에 달성하여야 한다.

* 알고있는 최적행동 정보를 통해 행동가치를 학습하여야 한다.
* 모든 행동에 대해서 탐색해(explore) 더 좋은 행동을 찾아야 하므로 최적행동이 아닌 행동에 대한 탐색을 해야한다.

최적행동을 선택하면서 최적행동이외의 다른행동에 대해 탐색할 수 없기 때문에 이 두 가지는 동시에 진행할 수 없는 과정이다. 하지만 두 가지 모두 최적정책을 찾기 위해서는 필수적인 과정이기도 하다. 앞서 다룬 on-policy 방식은 이에 대해 최적 정책이 아닌 현재 정책에 대해 가치함수를 학습하는 방식으로 타협한 방법으로 볼 수 있다. 애초에 최적정책을 찾아야 하는데 최적정책을 사용할 수는 없으므로 불가피한 타협이기는 하다. Target policy와 behavior policy를 따로 두는 방법도 생각해 볼 수 있다. 이 두가지는 Deep RL에서도 자주 등장하는 개념으로 학습을 통해 개선시키는, 즉 최적정책으로 나아가는 정책을 target policy라고 하며, 행동을 만들어내는 정책을 따로 두어 behavior policy라고 한다. Behavior policy에 탐색을 하는 성질을 부여하면 위의 두 가지 목적을 달성할 수 있다. 여기서 on-policy와 구분되는 차이가 발생한다. On-policy는 target policy와 behavior policy가 같았다면, off-policy에서는 target policy와 behavior policy가 달라진다. 그리고 이러한 방식을 off-policy learning이라고 한다.

교재 전반에서 on-policy와 off-policy방법 모두를 다루게 되며 보통 on-policy가 더 간단하므로 먼저 소개가 되고 off-policy방법은 추가적인 개념과 notation이 필요하므로 나중에 소개가 된다. Off-policy는 앞서 언급한대로 behavior policy와 target policy를 다르게 사용하므로 off-policy는 모델의 분산이 더 크고 수렴속도가 느리다는 성질이 있으나 on-policy보다 더 범용적으로 사용할 수 있고 더 좋은 성능을 보여준다. 따지고보면 on-policy는 off-policy의 특수한 경우로도 볼 수 있을 것이다. Off-policy가 가지는 대표적인 장점은 훈련하는 정책과 행동정책이 달라도 되므로 이전 정책에 의해 생성된 데이터를 사용할 수 있고(on-policy에서는 안된다!) 이러한 성질을 이용해 초기에 expert데이터를 사용해 좋은 초기값을 사용하는 것도 가능하다. 또한 여러 agent를 사용해서 학습시키도 용이하다.

여기서는 prediction에서의 off-policy 방법을 사용하는 것으로 시작한다. Prediction이므로 target policy이든 behavior policy이든 고정되었다고 가정한다. 정책을 활용해서 가치함수인 $v_{\pi}$와 $q_{\pi}$를 추정하며 앞의 prediction과 구분되는 차이점은 여기서 사용하는 행동정책인 behavior policy $b$는 학습시키려는 target policy $\pi$와 다른 정책이라는 점이다. 여기서 자연스럽게 생기는 의문은 target policy에 의한 가치함수를 추정하려고 하는데 가지고 있는 trajectory는 target policy가 만들어낸 trajectory가 아니라 다른 정책, behavior policy가 만들어낸 trajectory라는 점이다. 이 부분에 대해 문제의식을 가지고 이후 내용을 보면 좋다.

현재 우리가 가지고 있는 것은 behavior policy에 의해서 만들어진 episode이다. 하지만 추정해야하는 것은 target policy의 가치함수이다. 따라서 비록 $b$의 정책을 따라 갔지만 $\pi$를 따랐다면 어떤 행동을 했을지에 대한 정보도 필요하다. 적어도 어떤 상태 $s$에서 행동 $a$를 할 가능성이 두 정책에게 모두 열려있어야 한다. ($\pi(a \mid s) > 0$ implies $b(a \mid s) > 0$) 이러한 가정을 **assumption of coverage**라고 한다. $b$의 coverage는 특정상태에서 $\pi$와 다르다면 stochastic해야하고 target policy인 $\pi$는 deterministic policy로 control문제에서 사용하게 된다. Control 문제에서 target policy는 주로 현재 추정한 행동가치에 대해 greedy한 정책을 사용한다. Behavior policy는 탐색을 위해 일정수준의 stochastic한 행동결정을 하게 되고 target policy는 현시점에서 deterministic한 최적정책이라고 생각하면 된다.

거의 모든 off-policy 방법은 **importance sampling**을 사용한다. Importance sampling은 어떤 분포의 기대값을 다른 분포의 sample을 이용해 추정하는 기법이다.