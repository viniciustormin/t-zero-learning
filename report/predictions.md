# Previsões — escritas antes de olhar os resultados da varredura

Congeladas (e commitadas) com a varredura ainda em execução, antes de qualquer
figura ser gerada. O relatório final diz, para cada questão, se a previsão se
confirmou.

## Q1 — `dqn.target_network_frequency` ∈ {1, 500 (baseline), 5000}

Com `tnf=1` e `tau=1.0` a rede-alvo vira uma cópia da rede online a cada passo:
o alvo persegue a própria predição e o viés de maximização do `max_a Q` se
realimenta. Esperamos a curva de retorno degradar e `losses/q_values` crescer
muito acima do teto real (~100 para γ=0.99 e recompensa 1 por passo), enquanto
`losses/td_loss` permanece *baixo* — o erro é medido contra um alvo que se move
junto com a predição, então ele não mede mais nada. Com `tnf=5000` o alvo fica
quase congelado (só 100 sincronizações em 500k passos): esperamos `td_loss` em
dentes de serra, saltando a cada sincronização, e aprendizado mais lento, porque
o valor só se propaga um passo de bootstrap por sincronização. O baseline (500)
deve ser o melhor dos três.

## Q2 — `dqn.buffer_size` ∈ {500, 10 000 (baseline), 500 000}

Com `buffer=500` o minibatch de 128 sai de uma janela de ~1–2 episódios: as
amostras são fortemente correlacionadas e quase on-policy (o gradiente perde o
efeito de descorrelação que o replay existe para dar), e as transições antigas
são despejadas antes de a rede consolidar o que aprendeu com elas. Esperamos uma
curva de retorno instável — sobe e desaba repetidamente (esquecimento
catastrófico) — e `q_values` ruidoso, possivelmente inflado. Com `buffer=500 000`
nada é despejado em 500k passos: esperamos a curva mais estável das três, talvez
um pouco mais lenta no começo, já que o batch continua carregando transições
antigas de uma política ruim. O baseline deve ficar entre os dois, mais perto do
buffer grande.

## Q3 — `dqn.exploration_fraction` ∈ {0.05, 0.5 (baseline), 0.9}

Com `ef=0.05` o epsilon chega ao piso de 0.05 por volta de 25k passos, logo após
`learning_starts=10000`: esperamos aprendizado inicial rápido, com risco de
estacionar numa política subótima por falta de cobertura do espaço de estados.
Com `ef=0.9` o epsilon ainda vale ≈0.14 aos 400k passos: esperamos a curva de
retorno de *treino* deprimida o tempo todo — porque o que ela mede é o retorno da
política de comportamento, que continua jogando ~14% de ações aleatórias — mas
`eval/mean_return` (greedy) bem melhor do que a curva de treino sugere.

Gráficos escolhidos para a Q3 e por quê: `charts/epsilon` porque é ele que mostra
o cronograma *efetivamente* usado, e sem ele as outras duas curvas não são
interpretáveis; `charts/episodic_return_mean_last100` como a métrica de interesse;
e `losses/q_values` porque é o que separa "não aprendeu" de "aprendeu, a política
greedy é boa, mas o comportamento ainda é aleatório" — um Q alto com retorno de
treino baixo só é explicável pela segunda hipótese.
