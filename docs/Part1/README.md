# Part 1: Tabular Solution Methods

교재는 크게 세 개의 part로 나누어진다. 첫 번째 part의 제목은 Tabular Solution Methods이다. 강화학습에서 가장 간단한 형태로 상태와 행동의 가치함수가 array 형태로 표현되는 형태이다.

가령 미로찾기를 풀어내는 agent가 있다고 해보자. Part 1에서는 미로에서의 상태와 행동을 2차원 array꼴로 나타내 푸는 방법을 다룬다. 이 때, 각 행이 미로에서의 상태, 각 열이 해당 상태에서의 행동을 가리키고 array의 값은 가치함수값이 된다. 미로에서의 상태와 행동, 그 때의 가치를 2차원 array안에 담아내는 것이다. Agent는 이 array를 사용해 학습하고 update하면서 최적해를 찾아갈 수 있다. 이와 같이 tabular solution으로 접근할 수 있는 문제들은 대게 정확한 최적의 가치함수와 정책을 찾아낼 수 있다. 

Part 2의 제목은 Approximated Solution Methods인데, Part 2에서 다루는 방법은 이름에서 언급되듯 해법에 대한 근사값을 추정할 뿐, 정확한 해를 보장하지는 않는다. 하지만 그 대가로 더 많은 문제들에 적용할 수 있다는 장점이 있다.

Part 1에서는 단일상태(single state)를 가지는 Bandit problem을 다루고 강화학습의 문제를 정의하는 Finite Markov Decision Process(MDP)와 핵심 개념인 Bellman equation, 가치함수를 자세하게 다룬다. 이어서 Finite MDP로 문제를 formulation한 뒤 이를 푸는 dynamic programming, Monte Carlo methods, temporal-difference learning 세 가지 방법들을 다룬다. 각 방법의 특징은 요약하면 다음과 같다.

* Dynamic Programming: 수학적으로 잘 정의되어 있으나 환경에 대해 완전(complete)하고 정확한(accurate)한 model을 필요로 한다.
* Monte Carlo methods: Model을 필요로하지 않고 개념적으로 단순하나 단계적 증분 계산(step-by-step incremental computation)에는 적합하지 않다.
* Temporal-difference methods: Model을 필요로하지 않고 완전하게 증분적으로 계산(fully incremental)가능하나 분석하기가 용이하지 않다.

Part 1 후반부에서는 위의 내용을 바탕으로 Monte Carlo methods와 temporal-difference methods를 Multi-step bootstrapping method를 사용해 조합하는 방식을 다루며 마지막으로는 tabular 문제의 완전한 단일 해(complete and unified solution)를 찾는 방법으로써 어떻게 temporal-difference learning methods와 model learning, planning methods을 조합할 수 있는지를 다룬다.