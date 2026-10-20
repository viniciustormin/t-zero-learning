# Previsões — Partes 4 e 5 (escritas antes de cada experimento; não editadas depois)

Config de partida: `configs/sac_hopper_tuned.yml` (= `sac_hopper`, nenhum
candidato da Parte 3 passou do ruído). Comparação: seeds 1 e 2 do `base`.

## Parte 4.1 — `sac.autotune=false`, `sac.alpha=0.09` (escrita em 2026-10-20, antes de rodar)

Escolha do valor: mediana do `losses/alpha` na 2ª metade do treino (≥ 50k)
dos 3 seeds do base = 0,0999 / 0,0911 / 0,0739; a mediana das três é 0,0911,
então 0,09. É o valor em que o autotune "estaciona"; a ablação tira só o
ajuste, não o nível.

Previsão:
- No começo, o base tem α ≈ 0,9 → 0,1 nos primeiros ~5k passos de
  gradiente, ou seja, *mais* entropia que a ablação nesse trecho. Com α fixo
  em 0,09 desde o início, a entropia deve cair abaixo do alvo mais cedo.
- Depois, a escala de Q cresce (Q1 de ≈46 em 10k para ≈200 em 100k no base),
  e o termo α·log π pesa cada vez menos na perda do ator diante de
  min(Q1, Q2). Com α fixo, nada compensa isso: espero a entropia *derivando
  para baixo* do alvo −3 ao longo do treino (algo entre −4 e −6 no fim), em vez
  de colada nele como no base.
- Retorno: indistinguível do base com 2 seeds (o nível de α é o mesmo na
  maior parte do treino); talvez um pouco mais rápido no início.

## Parte 4.2 — `sac.clipped_double_q=false` (escrita em 2026-10-20, antes de rodar)

Previsão:
- Sem o mínimo, o alvo usa só Q1' (via `pick_q2`), e o ator também maximiza
  só Q1. O erro positivo de Q1 é escolhido pelo ator e propagado pelo alvo:
  `q_minus_return` positivo e crescendo ao longo do treino (no base fica perto
  de 0, salvo o colapso de base/1), e `qf1_values` acima do base no fim.
- Retorno: pior ou mais instável que o base, mas com só 100k passos e
  τ = 0,005 a rede-alvo segura parte do viés; espero o efeito em Q mais claro
  que o efeito no retorno.

## Parte 5 — SAC × PPO (escrita em 2026-10-20, antes dos runs isolados de tempo)

Nota de honestidade: os 2 seeds do PPO (`ppo_hopper`) rodaram na rodada 0,
junto com o base, como planejado, e o `summarize_sac.py` já mostrou o
resultado deles (eval 338,4 / 368,0; ret100 325,8 / 356,1). A previsão sobre
o retorno por passo, então, não é cega. O que ainda não foi medido isolado é
o tempo.

Previsão:
- Por passo de ambiente: SAC muito à frente (ret100 do base ≈ 950 contra
  ≈ 340), porque cada transição entra em ≈ (100k − 5k) × 256 / 100k ≈ 243
  passos de gradiente do crítico no SAC contra 10 (`update_epochs`) no PPO.
- Por tempo: o PPO roda a ≈ 6400 SPS (rodada 0, em paralelo) contra ≈ 350
  do SAC (mesma situação), ~18× mais rápido por passo. Em ≈ 20 s o PPO acaba
  os 100k; nesse tempo o SAC isolado ainda está na fase aleatória ou
  começando (5k passos a ~1000+ SPS e depois ~400 SPS). Espero o PPO na frente
  nos primeiros ~20–30 s e o SAC passando dele por volta de 1 min, e
  terminando muito acima: com o mesmo tempo *total* do SAC, o SAC vai mais
  longe, mas com o mesmo tempo do PPO, o PPO vai.
- `updates_per_step=4`: ≈ 4× o tempo do SAC (SPS ≈ 90–100 isolado) para um
  ret100 que não passou do ruído; na curva retorno × tempo, ele fica *atrás*
  do base em qualquer tempo intermediário.
