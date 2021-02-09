# Chapter 05: Monte Carlo Methods

Monte Carlo는 보통 구하기 어려운 통계량을 sampling을 통해 추정하는 맥락에서 등장한다. 따라서 Monte Carlo라는 이름에서 알 수 있듯, 이번 문서에서는 Monte Carlo를 강화학습에 사용하는 방법들을 살펴본다.

앞서 dynamic programming은 MDP에 대한 완전한 정보를 가지고 있을 때 적용가능한 방식이었다. 하지만 많은 경우 우리는 환경에 대한 정보(MDP)를 완전히 알지 못한다. 애초에 Monte Carlo를 사용하는 자체가 원하는 값을 몰라 통계량을 **추정**해야하는 상황이므로 Monte Carlo는 MDP를 몰라도 사용할 수 있는 방법일 것임을 짐작해 볼 수 있다. Monte Carlo는 DP의 접근방법과는 다르게 경험만을 필요로 한다. 즉, 상태, 행동, 보상의 과정인 trajectory만 있으면 된다. 가장 두드러지는 차이 중 하나는 환경의 transition probability matrix를 필요로 하는 DP와는 다르게 MDP는 경험 그 자체만을 사용하므로 더 범용적으로 사용할 수 있게 된다.

이러한 장점은 Monte Carlo를 사용해 transition probability matrix라는 확률분포를 추정한다는 관점으로도 볼 수 있을 것이다. MDP를 완전히 알고 있다면 transition probability distribution을 완벽하게 알고 있으므로 이를 바로 사용하여 풀면 된다. 하지만 많은 경우 우리는 transition probability matrix정보에 접근할 수가 없고, 다만 경험을 쌓는 것은 transtion probability matrix를 직접 찾으려는 것보다 훨씬 용이하게 수행할 수 있다. 따라서 쌓아둔 경험을 이용해 transiton probability matrix를 찾는 것은 sampling(경험)을 통해 통계량(transition probability matrix)을 추정하는 Monte Carlo의 방식에 딱 맞아 떨어진다.

MC방식은 강화학습의 문제를 sample return의 기대값을 이용해 접근한다. Return은 앞으로 받을 수 있는 보상의 총 합이다. 여기서 한 가지 제약이 생기게 되는데 앞으로 얼마만큼의 보상을 받을지를 계산해야 sample return을 만들 수 있으므로 return을 계산하기 위해서는 보상이 유한한 시점에 끝나야 한다. 즉 episodic task여야 한다는 것이다. 무한히 계속되는 continuous task라면 return을 계산하기위해 계속 기다려야 하는데 끝이 나지 않으므로 당장 sample return값을 얻는 것부터 어려워진다. 따라서 끝을 보고 sample return을 만들어 낼 수 있는 episodic task에 한정해서 MC방법을 사용하게 된다.

이러한 특성으로 인해 MC방법은 적어도 episode가 끝나야 가치함수와 정책에 대해 update를 수행할 수 있다. 후에 다루겠지만 이는 각 step마다 upate가 가능한 temporal difference방식과 대비되는 차이점이다. 강화학습에서의 Monte Carlo는 이처럼 **완전한(complete)** return의 평균에 기반한 학습방식을 의미한다.

MDP가 주어진 DP에서는 가치함수를 MDP를 이용해 계산할 수 있었지만 MDP에 대한 정보를 모를 때는 가치함수를 sample return을 통해 학습을 해야한다는 큰 차이가 있다. 이렇게 학습한 가치함수와 정책에 GPI의 아이디어를 사용해 최적가치/정책을 추정할 수 있게 된다. 이번 단원에서도 DP에서와 마찬가지로 주어진 정책을 평가하는 정책평가단계와 정책개선단계로 나누어 문제를 접근한다. 하지만 이번 단원에서는 Monte Carlo 방법을 사용해 과정을 수행하게 된다.