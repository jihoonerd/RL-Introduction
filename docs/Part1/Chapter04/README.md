# Chapter 04: Dynamic Programming

 Dynamic programming(DP)는 MDP를 정확하게 알고 있을 때, 최적정책을 구하는 알고리즘들을 일컫는 말이다. MDP를 정확하게 알고 있다는 것은 환경에 대한 정보를 모두 알고 있다는 뜻으로 전이확률행렬(probability transition matrix) $P_{s s^{\prime}}^{a}$나 보상함수 $R$을 알고 있음을 의미한다. 물론 실제 강화학습 문제에서 이렇게 환경에 대한 정보를 완벽하게 알고 있는 것은 흔한 상황은 아니다. 전통적인 DP 알고리즘들은 이러한 이유로 사용에 제약이 있는 편이다. 그럼에도 불구하고 강화학습의 문제를 접근하는 중요한 이론적 토대를 제공하며 이후에 다루는 내용은 DP에서 접근하는 방식을 보다 적은 계산비용으로, 완벽하지 않은 환경정보인 상황에서 접근하고 있다고도 할 수 있을 것이다.

 이번 문서에서는 DP가 어떻게 3장에서 다루었던 value function을 계산하는데 사용할 수 있는지를 볼 것이다. 일단 optimal value function만 찾으면 optimal value function에 대해 greedy하게 행동을 한 것이 optimal policy가 되므로 이번 장에서는 Bellman optimality equation이 자주 등장하게 된다. 특히, Bellman optimality equation에서 $p(s^{\prime}, r \mid s, a)$가 포함된 꼴을 사용하게 되는데 이 term을 사용한다는 자체가 환경에 대한 dynamics를 알고 있다는 점에 유의하면서 보도록 하자.

 ## Policy Evaluation (Prediction)