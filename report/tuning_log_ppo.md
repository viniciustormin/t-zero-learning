# PPO / LunarLander-v3 — diário de tuning (Parte 4)

Cada rodada: o que os gráficos do run anterior mostravam → mudança → resultado.
Números: `python tools/summarize_ppo.py` (métricas de `sweep_results_ppo/<arm>__seed<N>/metrics.jsonl`;
"0→1" = média do primeiro → último 20% do treino). Critério: `eval/mean_return` ≥ 200 e
`charts/episodic_return_mean_last100` ≥ 200 no fim, seeds 1 e 2, ≤ 2M passos.

## Rodada 0 — `base` (configs/ppo_lunarlander.yml sem mudanças, 1M)

| seed | ret100 final | máx | eval | approx_kl 0→1 | clipfrac | entropy 0→1 | expl_var 0→1 | grad_norm mediana |
|---|---|---|---|---|---|---|---|---|
| 1 | 2.7 | 62.0 | −10.6 | 0.0013 → 0.0001 | 0.00 | 1.16 → 0.83 | 0.04 → 0.64 | 6.0 (p10 2.2, p90 26) |
| 2 | 7.9 | 46.3 | −46.5 | 0.0014 → 0.0001 | 0.00 | 1.18 → 0.79 | 0.03 → 0.58 | 6.5 |

Leitura: o retorno sobe até ~0–50 em 400k e estaciona. A política quase não se move por
rollout: `approx_kl` ~1e-4 e `clipfrac` = 0 do começo ao fim, então o clipping, que é a razão de
ser do PPO, nunca é ativado. O motivo aparece em `grad_norm`: mediana de 6 contra
`max_grad_norm = 0.5`, cortado em 100% dos passos. Com `value_loss` ~50 (retornos do LunarLander
na casa das centenas) e `vf_coef = 0.5`, o gradiente do crítico ocupa quase toda a norma, e o
corte reduz o passo da política ~12×. A duração dos episódios sobe para ~1000 (truncamento) em
300k e depois cai para ~500: a política aprende a pairar sem cair, mas não a pousar.

## Rodada 1 — uma mudança por braço (1M, 2 seeds)

Hipótese: o gargalo é o corte da norma do gradiente, não o tamanho do passo nem o número de épocas.
- `r1_gn5`: `max_grad_norm` 0.5 → 5 (tira o corte)
- `r1_vf0.1`: `vf_coef` 0.5 → 0.1 (o crítico pesa menos na norma)
- `r1_ep10`: `update_epochs` 4 → 10 (controle: mais reuso, com o corte mantido)
- `r1_lr1e-3`: `learning_rate` 2.5e-4 → 1e-3 (controle: passo maior, com o corte mantido)

| braço | ret100 s1 / s2 | eval s1 / s2 | approx_kl início | clipfrac início | grad_norm mediana | duração final |
|---|---|---|---|---|---|---|
| base | 2.7 / 7.9 | −10.6 / −46.5 | 0.0013 | 0.00 | 6.0–6.5 | 524 / 438 |
| r1_gn5 | −35.4 / −43.6 | −29.3 / −60.2 | 0.004 | 0.01–0.02 | 4.5–4.9 | 785 / 907 |
| r1_vf0.1 | −23.8 / −18.8 | 3.8 / −41.7 | 0.002–0.003 | 0.00–0.01 | 1.2–1.4 | 672 / 645 |
| r1_ep10 | 90.1 / 104.7 | 81.2 / 53.7 | 0.005 | 0.02–0.03 | 3.3–3.7 | 260 / 232 |
| r1_lr1e-3 | **173.4 / 110.0** | 107.3 / 158.2 | 0.0076 | 0.04–0.05 | 4.5–5.3 | 256 / 280 |

Hipótese **refutada**. Tirar o corte (`gn5`) ou encolher o crítico na norma (`vf0.1`) não ajuda
nada; os dois terminam *abaixo* do base, pairando (episódios de 650–900 passos). O motivo:
`DiscreteActorCritic` tem ator e crítico em MLPs separados, então o `vf_coef` só chega ao ator
pelo corte global, que multiplica o gradiente do ator por uma constante. O Adam divide cada
parâmetro pela raiz da média móvel do quadrado do gradiente e desfaz quase toda essa constante.
O passo efetivo da política é fixado por `learning_rate` e pelo número de passos de gradiente
por rollout, que são justamente os dois controles que funcionaram. Os dois sobem o `approx_kl`
inicial de ~0.001 para 0.005–0.008 e tiram o `clipfrac` do zero. A política passa a *pousar*
(duração de 1000 para ~250). Em todos os braços o `approx_kl` cai para ~5e-4 no último quinto,
porque o `anneal_lr` leva a lr a zero.

## Rodada 2 — a partir de `lr=1e-3` (2 seeds)

- `r2_lr_ep10`: + `update_epochs` 10 (os dois controles da rodada 1 juntos)
- `r2_lr_mb16`: + `num_minibatches` 16 (4× mais passos de gradiente por rollout, minibatch de 32)
- `r2_lr_g999`: + `gamma` 0.999 (com episódios de ~250 passos, γ=0.99 desconta o bônus do pouso a ~0.08 no início do episódio)
- `r2_lr_2m`: + `total_timesteps` 2M (o orçamento todo, com o anneal mais lento)

| braço | ret100 s1 / s2 | máx | eval s1 / s2 | approx_kl início | 1º passo com ret100 ≥ 200 |
|---|---|---|---|---|---|
| r1_lr1e-3 (ref.) | 173.4 / 110.0 | 214 / 122 | 107 / 158 | 0.0076 | 0.80M / — |
| r2_lr_ep10 | **213.5 / 177.7** | 250 / 246 | **234 / 181** | 0.010 | **0.29M / 0.28M** |
| r2_lr_mb16 | 189.5 / 155.2 | 221 / 174 | 152 / 157 | 0.011 | 0.85M / — |
| r2_lr_g999 | 122.7 / 118.9 | 221 / 150 | 132 / 168 | 0.006 | 0.62M / — |
| r2_lr_2m | 175.1 / 174.7 | 206 / 208 | 179 / 199 | 0.007 | 1.10M / 1.54M |

`lr_ep10` é de longe o mais rápido: os dois seeds chegam a 200 em ~0.28M. Mas não sustenta. O
seed 1 cai para 34 em 0.71M e volta, e o seed 2 desce devagar de 238 (0.57M) para 150 (0.78M). As
quedas coincidem com picos de `approx_kl` (0.018–0.025, contra ~0.006 típico) e com a entropia
voltando a subir (0.37 → 0.70). Com 10 épocas e lr 1e-3, algumas atualizações levam a política
longe demais e ela demora a se recuperar. `mb16` (o mesmo número de passos de gradiente por rollout
de `ep10`, só que com minibatches 4× menores) fica atrás, com o KL inicial parecido mas mais
ruidoso. `gamma 0.999` sozinho não ajuda, e o `value_loss` final dobra no seed 1 (239). Estender
para 2M sem mais reuso só deixa o aprendizado mais lento (o anneal mantém a lr alta por mais
tempo, mas o KL continua baixo).

## Rodada 3 — estabilizar `lr 1e-3 + ep10`, já em 2M

- `r3_2m`: só o orçamento (controle)
- `r3_envs16`: `num_envs` 4 → 16 (batch de 2048: vantagens e gradientes menos ruidosos, 4× menos iterações)
- `r3_kl02`: `target_kl` = 0.02 (interrompe as épocas quando o KL dispara)
- `r3_lr5e-4`: `learning_rate` 1e-3 → 5e-4 (passo menor, com o reuso de 10 épocas mantido)

| braço | ret100 s1 / s2 | máx | eval s1 / s2 | approx_kl início | t200 s1 / s2 | menor ret100 após 1M |
|---|---|---|---|---|---|---|
| r3_2m | 240.3 / 234.5 | 258 / 269 | 254.4 / 250.1 | 0.011 | 0.29M / 0.65M | 144 / 157 (seed 1 cai para 66 em 0.7M) |
| r3_envs16 | 239.0 / 257.6 | 265 / 268 | 204.1 / 282.6 | 0.007 | 0.54M / 0.42M | ~205 / ~204 |
| r3_lr5e-4 | 210.8 / 194.8 | 247 / 226 | 178.6 / 184.6 | 0.007 | 0.40M / 0.73M | — |
| r3_kl02 | falhou: `--override ppo.target_kl=0.02` vira *string* (o parser tira o tipo do valor atual, `None`) → refeito em YAML na rodada 4 | | | | | |

`r3_2m` e `r3_envs16` cumprem o critério. O `r3_2m` mostra que o pico-e-queda da rodada 2 ainda
existe (seed 1: 185 → 66 → 170 entre 0.3M e 0.8M), e o anneal de 2M só dá tempo de recuperar. Com
`num_envs=16` (batch de 2048), as quedas somem: os dois seeds ficam em ≥ 200 a partir de ~1M. O
KL inicial cai de 0.011 para 0.007, e a mesma lr com 10 épocas passa a dar um passo mais
consistente, porque as vantagens normalizadas por minibatch de 512 são menos ruidosas que as de
128. Como bônus, o SPS dobra (10.7k contra 5.4k), porque são menos iterações e mais ambientes
vetorizados por chamada. `lr 5e-4` é mais lento e não chega lá em 2M.

## Rodada 4 — `target_kl` (YAML) e o config final em 4 seeds

- `r4_kl02` = `r3_2m` + `target_kl` 0.02 (`configs/ppo_lunarlander_targetkl.yml`)
- `final` = `configs/ppo_lunarlander_tuned.yml` (= `r3_envs16`), seeds 1–4

| braço | seed | ret100 final | eval (10 ep.) | eval refeito, 100 ep. (`evaluate.py`) |
|---|---|---|---|---|
| r4_kl02 | 1 / 2 | 258.8 / 213.2 | 212.0 / 242.6 | — |
| final | 1 | 239.0 | **197.6** (204.1 no r3_envs16) | 228.3 ± 96.8 |
| final | 2 | 257.6 | 220.1 (282.6 no r3_envs16) | 241.7 ± 77.7 |
| final | 3 | 265.7 | 270.1 | 254.4 ± 80.8 |
| final | 4 | 244.3 | 269.3 | 253.8 ± 69.3 |
| r3_2m | 1 / 2 | — | — | 246.6 ± 95.0 / 242.0 ± 108.5 |
| a2c_ll | 1 / 2 | — | — | 200.1 ± 99.9 / 189.7 ± 105.8 |

`target_kl` corta em média 0.4 das 10 épocas (`epochs_run` 9.6). O seed 2 ainda despenca de 236
para 168 entre 1.0M e 1.7M: o limite só para a atualização *depois* do minibatch que passou do
alvo, o que não segura o movimento ruim. O treino do `final` é determinístico (as curvas dos seeds
1/2 são idênticas às do `r3_envs16`), mas a avaliação final **não é**: são 10 episódios
*amostrados* da política, com desvio-padrão de ~100 quando um ou dois terminam em queda. O mesmo
modelo deu 204.1 e 197.6. Com 100 episódios, os 4 seeds ficam em 228–254 (erro padrão ~8–10),
então o 197.6 é ruído de amostragem. Mas o critério é literal.

## Rodada 5 — margem no eval estocástico (sobre o `final`, 2 seeds)

- `r5_ent0`: `ent_coef` 0.01 → 0 (a entropia termina em ~0.5; o eval amostra dessa política)
- `r5_g999`: `gamma` 0.99 → 0.999 (agora que a política pousa, o episódio inteiro de ~300 passos conta)

| braço | ret100 s1 / s2 | eval s1 / s2 | eval 100 ep. s1 / s2 | entropia final | t200 |
|---|---|---|---|---|---|
| final (ref.) | 239.0 / 257.6 | 197.6 / 220.1 | 228.3 / 241.7 | 0.54 / 0.48 | 0.54M / 0.42M |
| r5_ent0 | 155.0 / 176.2 | 204.5 / 247.4 | 184.0 / 148.5 | 0.52 / 0.55 | — |
| r5_g999 | 211.4 / 222.5 | 242.9 / 262.5 | 210.3 / 229.3 | 0.66 / 0.67 | 0.46M / 0.63M |

Nenhum dos dois ajuda. Com `ent_coef=0` a entropia final é a *mesma* (~0.5), então não era o bônus
que mantinha a política ruidosa (é a própria incerteza sobre quando acionar os motores laterais).
Sem o bônus, ela se fixa cedo numa política pior: ret100 de 155/176 e eval de 100 episódios de
184/149. O `gamma 0.999` termina com a entropia *mais alta* (0.67) e com o eval de 100 episódios
abaixo do `final`. Os 10 episódios de eval do r5 ficaram acima dos do `final`, mas em 100 episódios
ficam abaixo, o que é mais um exemplo de quanto esse número oscila.

## Decisão

Config final = `configs/ppo_lunarlander_tuned.yml` (`r3_envs16` = `final`).

## Parte 5 — A2C e ablações (para o relatório)

- `a2c_cp` = `a2c_cartpole` + `env_id=LunarLander-v3`, 2M; `a2c_ll` = `configs/a2c_lunarlander.yml` (o piloto da atividade de A2C), 2M.
- `bench_ppo` / `bench_a2c` / `bench_a2c_cp`: seed 1 de cada, um run por vez (`--jobs 1`), só para tempo e SPS.
  O `bench_ppo` reproduz o treino do `final` seed 1 e recebeu eval de 215.0 (é o terceiro valor desse mesmo modelo: 204.1, 197.6 e 215.0).
- Ablações sobre o `final`: `abl_reuse` (1 época, 1 minibatch) −75/−89; `abl_gae` (λ=1) 230/212, t200 de 0.71M/0.58M;
  `abl_clip` (ε=100) 215/141, KL inicial de 0.3–0.4, eval de 197/44.
