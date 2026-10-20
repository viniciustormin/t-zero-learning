# Diário de ajuste — SAC no Hopper-v5 (100k passos)

Todos os números saem de `python tools/summarize_sac.py` (uma linha por run) ou
de leituras do `metrics.jsonl` de um run citado. `run = braço/seed`.
Máquina: Mac com 8 núcleos P; até 8 runs em paralelo, 1 thread cada.

## Partes 1 e 2

- `tests/test_sac.py`: 16/16.
- O `train.py` do fork (e o do upstream) quebrava em todo treino SAC:
  `print(f"Learning rate: {args.algo.learning_rate}")`, e o `SACConfig` tem
  `policy_lr`/`q_lr`. Corrigido só esse print (harness, não o algoritmo).
- Sanidade, `pend/seed1` (Pendulum-v1, 20k): `eval/mean_return` = **−129,9**
  (amostrado −122,6); ret100 −528,8 (a janela inclui a fase aleatória).
  OK, acima de −200.

## Rodada 0 — `base` (sac_hopper sem mudanças), seeds 1–3, 4 runs em paralelo

| run | ret100 | eval | eval amostr. | alpha fim | ent (últ. 20%) | Q1 | Q−G (últ. 20%) | qf1_loss | min |
|---|---|---|---|---|---|---|---|---|---|
| base/1 | 542,0 | 2869,4 | 2705,5 | 0,0933 | −3,00 | 197,4 | 181,4 | 21,49 | 4,8 |
| base/2 | 1115,9 | 988,2 | 1474,8 | 0,0904 | −3,00 | 210,4 | −26,0 | 16,52 | 4,8 |
| base/3 | 1194,5 | 2965,1 | 3019,7 | 0,0652 | −3,01 | 212,4 | 3,5 | 12,53 | 4,8 |

Braço: eval 2274,2 (988,2–2965,1); ret100 950,8 (542,0–1194,5).

Leituras (tabela a cada 10k passos, lida dos `metrics.jsonl`):

- **alpha**: igual nos 3 seeds até 10k: 0,90 no 1º log (5,1k) → 0,41 (6,1k)
  → 0,18 (8,1k) → 0,10 (10,1k). Ou seja, em ~5k passos de gradiente a
  temperatura sai do valor inicial (log α = 0 → α = 1) e entra no regime.
  Depois oscila entre 0,06 e 0,16; mediana da 2ª metade: 0,0999 / 0,0911 / 0,0739.
  Nunca cai abaixo de 0,05 em nenhum log.
- **entropia**: de −2,0 em 10k (acima do alvo; α ainda caindo) para ≈ −3 a
  partir de ~20k e fica colada no alvo (−3,00/−3,00/−3,01 nos últimos 20%).
  O autotune faz o seu trabalho.
- **Q vs retorno**: Q1 sobe de ≈46 (10k) a ≈200–220 (100k), acompanhando o
  `soft_return_ep_start`. Exceção: **base/1 entre 70k e 80k**: ret100
  705 → 68, duração média dos episódios 259 → 93, `q_minus_return` ≈ 195–199
  (Q ≈ 180–207 contra G ≈ 57–66). Os críticos continuaram dando ≈200 a
  estados de onde a política agora cai em ~90 passos: superestimação depois de
  uma mudança da política. Antes disso, `qf1_loss` de base/1 teve picos de
  52,3 (30k) e 56,6 (50k), contra 10–25 nos outros seeds. base/1 se recupera
  para 542 em 100k.
- **ainda subindo em 100k?** Sim: ret100 de base/2 vai de 847 (80k) a 990
  (90k) e 1116 (100k); o de base/3, de 886 a 1027 e 1195 (+126 a +167 por 10k); os episódios
  ainda estão ficando mais longos (321 e 401 passos, de 1000 possíveis). O
  orçamento acaba no meio do aprendizado.
- **eval vs ret100**: o eval (política final, ação média) fica muito acima do
  ret100 em base/1 e base/3 (2869,4 / 542,0; 2965,1 / 1194,5), não em base/2,
  porque a janela de 100 episódios cobre ~30k passos de treino. O eval
  determinístico e o amostrado são próximos em base/1 e base/3 (2869/2706,
  2965/3020), mas em base/2 o amostrado é *maior* (988 vs 1475): 10
  episódios de uma política que às vezes cai cedo; o eval é ruidoso.
- **tempo**: 4,8 min por run com 4 em paralelo (SPS ≈ 350), muito abaixo dos
  ~17 min do comentário do config.

PPO (`ppo_hopper`, seeds 1–2, rodado junto): eval 338,4 / 368,0, ret100
325,8 / 356,1, 7680 passos de gradiente, 0,3 min. Termina em 98.304 passos
(24 iterações × 4096; 100k não é múltiplo do batch).

## Rodada 1 — uma mudança por braço, 3 seeds

Motivos (cada um citando o run onde foi visto):

- `r1_ups2` — `sac.updates_per_step` 1 → 2: base/2 e base/3 ainda ganham
  ~+130 de ret100 por 10k passos em 100k; o aprendizado está limitado por
  passos de gradiente, não por dados. Dobra os passos de gradiente por transição.
- `r1_ups4` — `sac.updates_per_step` 1 → 4: a mesma leitura, mais longe;
  risco de superestimação maior (já visto em base/1, Q−G ≈ 195 em 70–80k).
- `r1_ls1k` — `sac.learning_starts` 5000 → 1000: o alpha só entra no regime
  ≈5k passos de gradiente depois do início do aprendizado (0,90 em 5,1k →
  0,10 em 10,1k, nos 3 seeds); começar em 1k recupera 4% do orçamento.
- `r1_qlr3e-4` — `sac.q_lr` 1e-3 → 3e-4: o colapso de base/1 (70–80k) veio
  com Q−G ≈ 195 e picos de `qf1_loss` de 52–57; um crítico mais lento
  deve ser mais estável. (`q_lr` também é o lr do alpha.)

### Resultados da rodada 1 (12 runs, até 8 em paralelo, 20,3 min)

| braço | eval: média (mín–máx) | ret100: média (mín–máx) | eval amostr. | min/run (paralelo) |
|---|---|---|---|---|
| base | 2274,2 (988,2–2965,1) | 950,8 (542,0–1194,5) | 2400,0 (1474,8–3019,7) | 4,8 |
| r1_ups2 | 1034,1 (15,6–2793,3) | 946,8 (437,3–1353,5) | 887,2 (15,6–2342,7) | 12,1 |
| r1_ups4 | 1776,4 (1006,9–2447,4) | 1288,3 (1109,5–1434,7) | 1665,6 (1261,5–2143,8) | 20,2 |
| r1_ls1k | 1004,7 (973,3–1035,6) | 1127,6 (965,7–1426,9) | 1618,9 (970,9–2618,0) | 6,9–7,8 |
| r1_qlr3e-4 | 834,3 (347,6–1405,8) | 736,1 (335,3–1076,8) | 929,4 (336,2–1598,2) | 5,4–6,6 |

Vereditos (critério: diferença de médias maior que a variação entre seeds,
intervalos mín–máx que quase não se sobrepõem):

- **r1_ups4: indistinguível do ruído.** Maior média de ret100, mas base/2
  (1115,9) e base/3 (1194,5) caem dentro do intervalo do ups4. Diferença de médias de
  ret100 = +337,5, menor que a amplitude do base (652,5); os intervalos se
  sobrepõem de 1109,5 a 1194,5. No eval a média é *menor* que a do base
  (1776,4 vs 2274,2), intervalos sobrepostos. O mais promissor, mas não passa.
  Custo: 4× os passos de gradiente (379.600), 20,2 min/run em paralelo.
- **r1_ups2: indistinguível do ruído.** ret100 igual ao base (946,8 vs
  950,8). r1_ups2/2 colapsa entre 90k e 100k: ret100 1163 → 437, duração
  384 → 147, Q1 parado em ≈219, e a política final cai em ~15 passos
  (eval = 15,6 nos 10 episódios, determinístico e amostrado). Mesmo padrão do
  colapso de base/1.
- **r1_ls1k: indistinguível do ruído.** ret100 +176,8, intervalos
  sobrepostos; eval menor (1004,7), com os 3 seeds (973–1036) dentro da faixa do base.
  O alpha entra no regime antes (< 0,1 em 6k em vez de 10k), como previsto,
  mas isso não aparece no retorno.
- **r1_qlr3e-4: indistinguível do ruído (tendência de piora).** ret100 −214,7
  e eval −1439,9 em média, mas os intervalos se sobrepõem (r1_qlr3e-4/2 =
  1076,8 / 1405,8 está dentro do base). O crítico mais lento não evitou
  colapsos: r1_qlr3e-4/3 termina em 335,3.

**Config final = config inicial** (`configs/sac_hopper_tuned.yml`, corpo
idêntico ao `sac_hopper.yml`). Sem rodada 2: nenhum vencedor para combinar.
Na figura da Parte 3, o inicial vai ao lado do candidato mais próximo
(r1_ups4).

Observação sobre o eval: são 10 episódios de uma política final que, em
alguns seeds, acabou de colapsar (base/1 em 70–80k, r1_ups2/2 em 90–100k).
Ele vai de 15,6 a 2965,1 entre os runs da rodada 1 e do base; o ret100
é menos ruidoso, mas atrasa (janela de ~30k passos).

## Parte 4 — ablações sobre o config final (= base), seeds 1–2 (4 runs em paralelo, 4,8 min)

| braço | eval: média (mín–máx) | ret100: média (mín–máx) | alpha fim | ent (últ. 20%) | Q1 | Q−G (últ. 20%) | qf1_loss |
|---|---|---|---|---|---|---|---|
| base (s1–2) | 1928,8 (988,2–2869,4) | 829,0 (542,0–1115,9) | 0,0933 / 0,0904 | −3,00 / −3,00 | 197,4 / 210,4 | 181,4 / −26,0 | 21,49 / 16,52 |
| abl_alpha (α = 0,09) | 2120,5 (1073,4–3167,6) | 567,4 (435,2–699,7) | 0,09 / 0,09 | −3,33 / −2,89 | 213,1 / 186,2 | 120,6 / −7,0 | 15,28 / 18,78 |
| abl_1q | 2446,9 (2084,2–2809,6) | 1057,3 (901,7–1212,8) | 0,2458 / 0,2194 | −3,00 / −2,99 | 543,7 / 517,2 | 494,3 / 471,1 | 242,11 / 184,63 |

Leituras complementares (do `metrics.jsonl`):
- abl_alpha: entropia média entre 20k e 50k = −3,53 / −3,45 (base: −3,01 /
  −3,00); a partir de 50k, −3,30 / −3,00 (mín −4,22 / −3,86). Em 10k: −2,66 /
  −2,10 contra −2,06 / −1,87 do base (o base ainda tem α > 0,1 ali).
  abl_alpha/1 colapsa entre 80k e 100k (ret100 1142 → 435), com Q−G ≈ 173 em
  80k: o mesmo padrão de base/1.
- abl_1q: Q1 de 174 / 212 já em 20k (base 108 / 113) e 365 / 426 em 40k;
  Q−G médio ≥ 50k = 490 / 474 (base 119 / −19). alpha mediano ≥ 50k = 0,320 /
  0,295, ≈ 3× o do base (0,100 / 0,091); a entropia fica no alvo (−3,00).

Previsões (`predictions_sac.md`) contra o resultado:
- 4.1: **errei a deriva da entropia.** Previ que ela fosse cair para −4 a −6
  com a escala de Q subindo; ela fica 0,3–0,5 abaixo do alvo entre 20k e 50k e
  volta a −3,3 / −3,0. O que acerta a perda do ator (`compute_actor_loss`) é
  ∂Q/∂a, não o nível de Q: o Q sobe de 46 para 200 sobretudo pelo valor do
  estado, comum a todas as ações. Acertei o começo (entropia menor que a do
  base em 10k). Retorno: indistinguível (eval 2120,5 contra 1928,8, ret100
  567,4 contra 829,0, intervalos sobrepostos).
- 4.2: **acertei a superestimação, errei o retorno.** Q−G ≈ 480 e Q1 ≈ 2,6×
  o base; qf1_loss ≈ 10×. Mas o retorno não piorou com 2 seeds (eval 2446,9,
  ret100 1057,3, os dois dentro ou acima da faixa do base). O que eu não
  previ: o alpha triplica. Sem o mínimo, o ator persegue o máximo de Q1 e
  fica mais "afiado" (∂Q1/∂a maior); a entropia cairia, e o
  `compute_alpha_loss` sobe α até a entropia voltar a −3. A superestimação, que
  é em boa parte um deslocamento comum às ações, quase não muda a ordem das ações
  em 100k passos.

## Parte 5 — contagem de usos por transição

- SAC base: `num_updates` = 94.900 passos de gradiente do crítico (um por
  passo depois de `learning_starts`), batch 256: 94.900 × 256 / 100.000 =
  **242,9** usos por transição. O ator dá 2 passos a cada 2 do crítico
  (`policy_frequency` 2, compensado): 94.900 passos, nos *mesmos* batches.
  Com `updates_per_step=4`: 379.600 × 256 / 100k = **971,8**.
- PPO: 24 iterações × 4096 = 98.304 transições; 10 épocas × 32
  minibatches de 128 = 320 passos por iteração; `num_updates` = 7.680.
  7.680 × 128 / 98.304 = **10,0** = `update_epochs` (cada época embaralha o
  batch e percorre todos os minibatches, `ppo_continuous_action.py:286`).

## Parte 5 — runs isolados (um por vez, nada mais rodando; `bench.log`)

| run isolado (seed 1) | tempo (s, `_wall`) | SPS | passos de grad. | ret100 em 20 s | 60 s | 120 s | fim | s até 300 | 500 | 1000 |
|---|---|---|---|---|---|---|---|---|---|---|
| bench_ppo | 16 | 7105 | 7.680 | 325,8 | 325,8 | 325,8 | 325,8 | 15 | – | – |
| bench_sac_final (= base) | 220 | 460 | 94.900 | 151,8 | 355,5 | 559,4 | 542,0 | 44 | 110 | – |
| bench_sac_ups4 | 800 | 125 | 379.600 | 50,8 | 142,0 | 128,0 | 1320,9 | 266 | 408 | 620 |

- O treino é determinístico: bench_sac_final reproduz base/1 (ret100 542,0,
  mesmos passos até 300/500), e bench_ppo reproduz ppo_hopper/1 (325,8). O
  eval, não: o mesmo modelo de base/1 deu 2869,4 e 2699,6.
- Por passo: PPO atinge ret100 300 em 90,1k / 69,6k passos; o SAC base, em
  23,5k / 32,5k / 30,5k. Em 100k o PPO para em 325,8 / 356,1; o SAC base
  chega a 542,0 / 1115,9 / 1194,5.
- Por tempo: o PPO termina em 16 s; nesse momento (20 s) o SAC está em
  151,8. O SAC passa de 300 em 44 s e está em 355,5 em 60 s, acima do fim do
  PPO. Com o *mesmo tempo do PPO*, o PPO vai mais longe; com mais de ~1 min, o SAC.
- `updates_per_step=4`: 3,7× mais lento por passo (125 contra 460 SPS), 4×
  os passos de gradiente. Fica atrás do base em todo tempo intermediário (142 contra
  355,5 em 60 s, 128 contra 559,4 em 120 s) e só passa dele depois de 220 s
  (1320,9 em 800 s, mas é 1 seed; com 3 seeds não passou do ruído).
- SPS PPO / SAC = 7105 / 460 = 15,4×; passos de gradiente SAC / PPO =
  94.900 / 7.680 = 12,4×.

Previsão da Parte 5 contra o resultado: acertou a ordem (PPO na frente até
~20 s, SAC na frente a partir de ~1 min: passa de 300 em 44 s) e o
`updates_per_step=4` atrás em todo tempo intermediário. O SPS isolado do PPO
(7105) foi maior que o dos runs em paralelo (≈6400), e o do SAC (460) também
(≈350).
