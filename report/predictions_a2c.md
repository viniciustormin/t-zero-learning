# Previsões — A2C (escritas antes de olhar os resultados da varredura)

Congeladas e commitadas junto com a implementação, antes de qualquer run completo
do sweep e antes de qualquer figura existir. O relatório final diz, para cada
questão, se a previsão se confirmou.

Baseline: `configs/a2c_cartpole.yml` — `num_envs=8`, `a2c.num_steps=5`,
`ent_coef=0.01`, `use_baseline=true`, 500k passos, CartPole-v1.

## Q1 — `num_envs` ∈ {1, 8 (baseline), 32} + controle `num_envs=1, num_steps=40`

Com 1 ambiente o lote de um update são 5 transições consecutivas do mesmo
episódio: o gradiente estimado é praticamente o gradiente de um único estado,
com variância enorme e forte correlação temporal. Esperamos `losses/policy_loss`
com ruído muito maior (oscilando de sinal a cada update) e a curva de retorno
instável ou francamente pior. Com `num_envs=32` o lote é de 160 transições de 32
episódios independentes: gradiente muito mais próximo do verdadeiro, curva mais
suave — mas como `num_iterations = 500k / (num_envs × num_steps)`, são só 3125
updates contra 12500 do baseline, então esperamos aprendizado *mais lento em
passos de ambiente*, possivelmente sem chegar a 500 em 500k. O controle
`num_envs=1, num_steps=40` tem exatamente o mesmo `batch_size` (40) e o mesmo
número de updates do baseline — se ele ficar pior que o baseline, a diferença é
*descorrelação entre atores*, não tamanho de lote. Em `charts/SPS` esperamos
ganho grande de 1 → 8 (o custo por update é amortizado sobre mais transições) e
ganho menor de 8 → 32 em CPU.

## Q2 — `a2c.num_steps` ∈ {1, 5 (baseline), 32, 128}

`num_steps=1` dá o alvo `r + γV(s')`: variância mínima, viés máximo (o alvo é
quase inteiramente a própria estimativa do crítico). `num_steps=128` é quase
Monte Carlo: viés quase nulo, variância alta, e o crédito demora 128 passos para
ser atribuído. Esperamos que `losses/explained_variance` fique *enganosamente
alta* com `num_steps=1` — o crítico está explicando a variância de um alvo que
ele mesmo gerou — e mais baixa com `num_steps=128`, onde o alvo é o retorno real
e ruidoso. `losses/value_loss` deve *crescer* com `num_steps` simplesmente porque
a escala do alvo cresce (retornos de até ~100 contra ~1+γV). Esperamos o retorno
melhor no meio da varredura (5 ou 32) e pior nos extremos: `num_steps=1` porque o
viés do bootstrap trava a propagação de valor, `num_steps=128` porque a variância
do peso do gradiente domina.

## Q3 — `a2c.ent_coef` ∈ {0, 0.01 (baseline), 0.1} (2 seeds em `ent_coef=0`)

O bônus de entropia impede o colapso prematuro da política para uma
determinística antes de o crítico ser confiável — no gradiente de política existe
um laço de realimentação positiva ("ação amostrada com vantagem positiva → sua
probabilidade sobe → é amostrada mais"), e a entropia é o termo que o contrapõe.
Com `ent_coef=0.1` esperamos `losses/entropy` presa perto de ln 2 ≈ 0,69
(uniforme): a perda total passa a ser dominada por −0,1·H, e uma política
determinística — que é a ótima no CartPole — é penalizada demais; o retorno deve
estacionar bem abaixo de 500. Com `ent_coef=0` esperamos que o CartPole *funcione
mesmo assim* na maioria dos seeds (2 ações, recompensa densa e sempre positiva,
e o próprio ambiente gera estados novos quando a política é ruim), com entropia
caindo mais rápido e mais cedo, e maior variância entre seeds.

## Q4 — `a2c.use_baseline=false` (2 seeds)

Subtrair V(s) não muda o gradiente esperado (E[∇log π · b] = 0 para b independente
da ação), mas muda a variância. Esperamos `charts/advantage_mean` ≈ 0 com baseline
e fortemente positivo sem ele (≈ o retorno descontado médio, da ordem de dezenas),
enquanto `charts/advantage_std` fica na mesma ordem de grandeza nos dois — ou seja,
a razão sinal/ruído do peso piora muito sem baseline, porque o peso passa a ser
"offset grande + variação pequena". Na entropia esperamos colapso *mais rápido*
sem baseline: no CartPole toda recompensa é positiva, então o peso é positivo para
*qualquer* ação amostrada, e o update sobe log π(a|s) de tudo que foi tentado — a
política reforça o que foi amostrado, não o que foi bom, e fica determinística cedo
por um motivo errado. Esperamos retorno pior e mais instável entre os dois seeds.
