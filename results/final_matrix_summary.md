# Final matrix — 190 non-attention cells

generated 2026-08-03 21:07 · git HEAD e8b4be6 (main) · newest result 2026-08-03 17:43:35

done 103 · aborted 2 · pending 85 · 710.6 h wall-clock over 103 timed cell(s)

```
exp  idx  model                  prm   L   l/N  site   arch          wid    K  bench  status  nHold  fullEx      p  s  fullAp   nFlt  nFltH    fltEx   fltP  fltS    fltAp   kfold   +f     gen   hrs
399  100  pythia_70m             70M   2  0.40  resid  batchtopk   32768   16  CSQA   done     1380  +0.00%  1.000     +0.00%  10934   1366   +0.00%  1.000         +0.00%  +0.00%  0/5  +0.00%   1.7
400  101  pythia_70m             70M   2  0.40  resid  batchtopk   32768   16  ARC    done      291  +0.00%  1.000     +0.00%   2277    285   +0.00%  1.000         +0.00%  +0.00%  0/5  +0.00%   0.7
395   96  pythia_70m             70M   2  0.40  mlp    batchtopk   32768   16  CSQA   done     1380  -0.43%  0.885     -0.36%  10934   1366   -0.51%  0.919         -0.37%  -0.43%  1/5  +0.04%   1.2
396   97  pythia_70m             70M   2  0.40  mlp    batchtopk   32768   16  ARC    done      291  +0.00%  1.000     +0.00%   2277    285   +0.00%  1.000         +0.00%  +0.00%  0/5  -0.01%   0.6
401  102  pythia_70m             70M   5  1.00  resid  batchtopk   32768   16  CSQA   done     1380  +0.00%  1.000     +0.00%  10934   1366   +0.00%  1.000         +0.00%  +0.00%  0/5  -0.01%   2.2
402  103  pythia_70m             70M   5  1.00  resid  batchtopk   32768   16  ARC    done      291  +0.00%  1.000     +0.00%   2277    285   +0.00%  1.000         +0.00%  +0.00%  0/5  +0.00%   0.8
397   98  pythia_70m             70M   5  1.00  mlp    batchtopk   32768   16  CSQA   done     1380  +0.00%  1.000     +0.00%  10934   1366   +0.00%  1.000         +0.00%  +0.00%  0/5  +0.00%   2.2
398   99  pythia_70m             70M   5  1.00  mlp    batchtopk   32768   16  ARC    done      291  +0.00%  1.000     +0.00%   2277    285   +0.00%  1.000         +0.00%  +0.00%  0/5  +0.00%   0.8

411  112  pythia_70m_deduped     70M   2  0.40  resid  batchtopk   32768   16  CSQA   pend        -       -      -  -       -      -      -        -      -  -           -       -    -       -     - (drop)
412  113  pythia_70m_deduped     70M   2  0.40  resid  batchtopk   32768   16  ARC    pend        -       -      -  -       -      -      -        -      -  -           -       -    -       -     - (drop)
407  108  pythia_70m_deduped     70M   2  0.40  mlp    batchtopk   32768   16  CSQA   pend        -       -      -  -       -      -      -        -      -  -           -       -    -       -     - (drop)
408  109  pythia_70m_deduped     70M   2  0.40  mlp    batchtopk   32768   16  ARC    pend        -       -      -  -       -      -      -        -      -  -           -       -    -       -     - (drop)
413  114  pythia_70m_deduped     70M   5  1.00  resid  batchtopk   32768   16  CSQA   pend        -       -      -  -       -      -      -        -      -  -           -       -    -       -     - (drop)
414  115  pythia_70m_deduped     70M   5  1.00  resid  batchtopk   32768   16  ARC    pend        -       -      -  -       -      -      -        -      -  -           -       -    -       -     - (drop)
409  110  pythia_70m_deduped     70M   5  1.00  mlp    batchtopk   32768   16  CSQA   pend        -       -      -  -       -      -      -        -      -  -           -       -    -       -     - (drop)
410  111  pythia_70m_deduped     70M   5  1.00  mlp    batchtopk   32768   16  ARC    pend        -       -      -  -       -      -      -        -      -  -           -       -    -       -     - (drop)

306    7  gpt2_small            124M   5  0.46  resid  batchtopk  131072   32  CSQA   done     4161  -0.12%  0.941     -0.12%   3519    440   +0.00%  0.637         +0.00%  -0.12%  1/5  -0.04%   2.9
505  206  gpt2_small            124M   5  0.46  resid  batchtopk  131072   32  ARC    pend        -       -      -  -       -      -      -        -      -  -           -       -    -       -     - (drop)
506  207  gpt2_small            124M  11  1.00  resid  batchtopk  131072   32  CSQA   pend        -       -      -  -       -      -      -        -      -  -           -       -    -       -     - (drop)
507  208  gpt2_small            124M  11  1.00  resid  batchtopk  131072   32  ARC    pend        -       -      -  -       -      -      -        -      -  -           -       -    -       -     - (drop)

311   12  smollm2_135m          135M  15  0.52  mlp    batchtopk   36864   32  CSQA   done     2761  +0.36%  0.015  *  +0.07%   7252    907   +1.10%  0.015  *      +0.22%  +0.36%  4/5  +0.12%  11.5
415  116  smollm2_135m          135M  15  0.52  mlp    batchtopk   36864   32  ARC    done      357  +0.84%  0.254     +0.28%   2103    263   +1.14%  0.254         +0.38%  +0.84%  2/5  +0.03%   5.2 (run) agree
416  117  smollm2_135m          135M  27  0.93  mlp    batchtopk   36864   32  CSQA   done     2761  +0.00%  0.598     +0.04%   7252    907   +0.00%  0.598         +0.11%  +0.00%  1/5  -0.04%  17.2 (run) agree
417  118  smollm2_135m          135M  27  0.93  mlp    batchtopk   36864   32  ARC    done      357  +0.56%  0.344     +0.28%   2103    263   +0.76%  0.344         +0.38%  +0.56%  2/5  +0.04%   4.7 (run) agree

361   62  pythia_160m           160M   5  0.46  resid  batchtopk   32768   32  CSQA   done     3728  -0.05%  0.806     -0.08%   4673    584   -0.17%  1.000         -0.17%  -0.05%  2/5  -0.05%   3.8
362   63  pythia_160m           160M   5  0.46  resid  batchtopk   32768   32  ARC    done      840  -0.24%  0.938     -0.36%    813    102   +0.98%  0.500         +0.00%  -0.24%  1/5  -0.00%   1.2
308    9  pythia_160m           160M   6  0.55  mlp    batchtopk   32768   64  CSQA   done     3538  -0.11%  1.000     -0.08%   5180    648   +0.00%  1.000         +0.00%  -0.11%  0/5  +0.07%   3.2
377   78  pythia_160m           160M   6  0.55  mlp    batchtopk  131072   64  CSQA   pend        -       -      -  -       -      -      -        -      -  -           -       -    -       -     - (drop)
380   81  pythia_160m           160M   6  0.55  mlp    batchtopk    4084   64  CSQA   pend        -       -      -  -       -      -      -        -      -  -           -       -    -       -     - (drop)
382   83  pythia_160m           160M   6  0.55  mlp    batchtopk   65536   64  CSQA   pend        -       -      -  -       -      -      -        -      -  -           -       -    -       -     - (drop)
384   85  pythia_160m           160M   6  0.55  mlp    batchtopk     768   64  CSQA   pend        -       -      -  -       -      -      -        -      -  -           -       -    -       -     - (drop)
378   79  pythia_160m           160M   6  0.55  mlp    batchtopk  131072   64  ARC    pend        -       -      -  -       -      -      -        -      -  -           -       -    -       -     - (drop)
379   80  pythia_160m           160M   6  0.55  mlp    batchtopk   32768   64  ARC    pend        -       -      -  -       -      -      -        -      -  -           -       -    -       -     - (drop)
381   82  pythia_160m           160M   6  0.55  mlp    batchtopk    4084   64  ARC    pend        -       -      -  -       -      -      -        -      -  -           -       -    -       -     - (drop)
383   84  pythia_160m           160M   6  0.55  mlp    batchtopk   65536   64  ARC    pend        -       -      -  -       -      -      -        -      -  -           -       -    -       -     - (drop)
385   86  pythia_160m           160M   6  0.55  mlp    batchtopk     768   64  ARC    pend        -       -      -  -       -      -      -        -      -  -           -       -    -       -     - (drop)
386   87  pythia_160m           160M  10  0.91  mlp    batchtopk     768   64  CSQA   pend        -       -      -  -       -      -      -        -      -  -           -       -    -       -     - (drop)
387   88  pythia_160m           160M  10  0.91  mlp    batchtopk     768   64  ARC    pend        -       -      -  -       -      -      -        -      -  -           -       -    -       -     - (drop)
363   64  pythia_160m           160M  11  1.00  resid  batchtopk   32768   32  CSQA   done     3728  -0.11%  0.717     +0.00%   4673    584   -0.17%  0.726         -0.51%  -0.11%  2/5  -0.28%   3.8
364   65  pythia_160m           160M  11  1.00  resid  batchtopk   32768   32  ARC    done      840  +0.00%  1.000     +0.00%    813    102   +0.00%  1.000         +0.00%  +0.00%  0/5  +0.00%   1.2

373   74  pythia_160m_deduped   160M   5  0.46  resid  batchtopk   65536   32  CSQA   done     3211  +0.06%  0.430     +0.12%   6052    757   -0.13%  0.812         +0.13%  +0.06%  2/5  +0.07%   3.7
374   75  pythia_160m_deduped   160M   5  0.46  resid  batchtopk   65536   32  ARC    done      806  +0.00%  0.750     +0.00%    905    113   +0.00%  1.000         +0.00%  -0.00%  1/5  +0.03%   1.2
369   70  pythia_160m_deduped   160M   5  0.46  mlp    batchtopk   65536   32  CSQA   done     3211  -0.06%  0.806     +0.09%   6052    757   +0.00%  1.000         -0.26%  -0.06%  2/5  +0.07%   3.6
370   71  pythia_160m_deduped   160M   5  0.46  mlp    batchtopk   65536   32  ARC    done      806  +0.12%  0.500     +0.12%    905    113   +0.00%  1.000         +0.00%  +0.12%  1/5  +0.03%   1.2
375   76  pythia_160m_deduped   160M  11  1.00  resid  batchtopk   65536   32  CSQA   done     3211  -0.19%  0.830     -0.12%   6052    757   -0.92%  0.998         -0.26%  -0.19%  1/5  +0.03%   3.8
376   77  pythia_160m_deduped   160M  11  1.00  resid  batchtopk   65536   32  ARC    done      806  +0.12%  0.500     +0.12%    905    113   +0.00%  1.000         -0.88%  +0.12%  1/5  +0.01%   1.0
371   72  pythia_160m_deduped   160M  11  1.00  mlp    batchtopk   65536   32  CSQA   done     3211  -0.03%  0.655     -0.03%   6052    757   -0.13%  1.000         -0.13%  -0.03%  2/5  +0.03%   3.8
372   73  pythia_160m_deduped   160M  11  1.00  mlp    batchtopk   65536   32  ARC    done      806  +0.12%  0.500     +0.12%    905    113   +0.00%  1.000         +0.00%  +0.12%  1/5  -0.00%   1.2

450  151  gemma3_270m           270M   9  0.53  resid  jump_relu   16384    -  CSQA   done     2810  +0.36%  0.006  *  -0.04%   7121    890   +1.12%  0.006  *      -0.11%  +0.36%  5/5  -0.08%   8.8
451  152  gemma3_270m           270M   9  0.53  resid  jump_relu   16384    -  ARC    done      391  -0.51%  1.000     -0.51%   2012    251   -0.80%  1.000         -0.80%  -0.51%  0/5  +0.03%   2.7
446  147  gemma3_270m           270M   9  0.53  mlp    jump_relu   16384    -  CSQA   done     2810  +0.04%  0.500     +0.07%   7121    890   +0.11%  0.500         +0.22%  +0.04%  1/5  -0.13%  12.5
447  148  gemma3_270m           270M   9  0.53  mlp    jump_relu   16384    -  ARC    done      391  -0.51%  0.891     -0.77%   2012    251   -0.80%  0.891         -1.20%  -0.52%  1/5  -0.01%   5.1 (run) agree
452  153  gemma3_270m           270M  15  0.88  resid  jump_relu   16384    -  CSQA   done     2810  -0.07%  0.891     +0.07%   7121    890   -0.22%  0.891         +0.22%  -0.07%  1/5  +0.02%  12.2
453  154  gemma3_270m           270M  15  0.88  resid  jump_relu   16384    -  ARC    done      391  -0.77%  1.000     -0.26%   2012    251   -1.20%  1.000         -0.40%  -0.77%  0/5  +0.05%   3.4
448  149  gemma3_270m           270M  15  0.88  mlp    jump_relu   16384    -  CSQA   done     2810  +0.11%  0.227     +0.04%   7121    890   +0.34%  0.227         +0.11%  +0.11%  1/5  -0.07%  11.7
449  150  gemma3_270m           270M  15  0.88  mlp    jump_relu   16384    -  ARC    done      391  -0.51%  0.938     +0.00%   2012    251   -0.80%  0.938         +0.00%  -0.52%  1/5  +0.02%   4.7 (run) agree

438  139  gemma3_270m_it        270M   9  0.53  resid  jump_relu   16384    -  CSQA   pend        -       -      -  -       -      -      -        -      -  -           -       -    -       -     - (run later) depending on gemma3_270m results
439  140  gemma3_270m_it        270M   9  0.53  resid  jump_relu   16384    -  ARC    pend        -       -      -  -       -      -      -        -      -  -           -       -    -       -     - (run later) depending on gemma3_270m results
440  141  gemma3_270m_it        270M  15  0.88  resid  jump_relu   16384    -  CSQA   pend        -       -      -  -       -      -      -        -      -  -           -       -    -       -     - (run later) depending on gemma3_270m results
441  142  gemma3_270m_it        270M  15  0.88  resid  jump_relu   16384    -  ARC    pend        -       -      -  -       -      -      -        -      -  -           -       -    -       -     - (run later) depending on gemma3_270m results

388   89  pythia_410m           410M  11  0.48  mlp    batchtopk   65536   32  CSQA   done     4625  -0.02%  0.773     -0.09%   2281    286   +0.00%  0.656         -0.70%  -0.02%  2/5  -0.01%  14.6 (run) agree
309   10  pythia_410m           410M  11  0.48  mlp    batchtopk   65536   32  ARC    done      819  +0.24%  0.250     +0.37%    869    109   +1.83%  0.250         +2.75%  +0.24%  2/5  -0.01%   2.6
389   90  pythia_410m           410M  23  1.00  mlp    batchtopk   65536   32  CSQA   done     4625  +0.04%  0.250     +0.04%   2281    286   +0.70%  0.250         +0.70%  +0.04%  1/5  +0.00%  15.3 (run) agree
390   91  pythia_410m           410M  23  1.00  mlp    batchtopk   65536   32  ARC    done      819  +0.00%  0.750     +0.12%    869    109   +0.00%  0.750         +0.92%  +0.00%  1/5  -0.10%   3.8 (run) agree

331   32  gemma3_1b               1B  12  0.48  resid  jump_relu   32768    -  CSQA   done     1932  -0.10%  0.708     +0.31%   9462   1183   -0.17%  0.708         +0.51%  -0.10%  3/5  -0.15%  16.1
332   33  gemma3_1b               1B  12  0.48  resid  jump_relu   32768    -  ARC    done      294  -0.34%  0.746     -0.34%   2269    283   -0.35%  0.746         -0.35%  -0.34%  1/5  -0.15%   5.4
434  135  gemma3_1b               1B  13  0.52  resid  jump_relu   16384    -  CSQA   done     1932  +0.36%  0.155     -0.10%   9462   1183   +0.59%  0.155         -0.17%  +0.36%  4/5  -0.15%  11.8
435  136  gemma3_1b               1B  13  0.52  resid  jump_relu   16384    -  ARC    pend        -       -      -  -       -      -      -        -      -  -           -       -    -       -     - (run later) but only if money allows it
430  131  gemma3_1b               1B  13  0.52  mlp    jump_relu   16384    -  CSQA   pend        -       -      -  -       -      -      -        -      -  -           -       -    -       -     - (run later) but only if money allows it
431  132  gemma3_1b               1B  13  0.52  mlp    jump_relu   16384    -  ARC    done      294  +1.02%  0.188     -1.36%   2269    283   +1.06%  0.188         -1.41%  +0.97%  2/5  -0.05%   6.2 (run later) but only if money allows it
436  137  gemma3_1b               1B  22  0.88  resid  jump_relu   16384    -  CSQA   done     1932  -0.36%  0.939     -0.21%   9462   1183   -0.59%  0.939         -0.34%  -0.36%  0/5  +0.04%  20.1
437  138  gemma3_1b               1B  22  0.88  resid  jump_relu   16384    -  ARC    pend        -       -      -  -       -      -      -        -      -  -           -       -    -       -     - (run later) but only if money allows it
432  133  gemma3_1b               1B  22  0.88  mlp    jump_relu   16384    -  CSQA   pend        -       -      -  -       -      -      -        -      -  -           -       -    -       -     - (run later) but only if money allows it
433  134  gemma3_1b               1B  22  0.88  mlp    jump_relu   16384    -  ARC    pend        -       -      -  -       -      -      -        -      -  -           -       -    -       -     - (run later) but only if money allows it
333   34  gemma3_1b               1B  24  0.96  resid  jump_relu   32768    -  CSQA   abrt        -       -      -  -       -      -      -        -      -  -           -       -    -       -     - (run later) but only if money allows it
334   35  gemma3_1b               1B  24  0.96  resid  jump_relu   32768    -  ARC    done      294  +0.00%  0.750     +0.34%   2269    283   +0.00%  0.750         +0.35%  +0.00%  0/5  -0.06%   6.1

422  123  gemma3_1b_it            1B  13  0.52  resid  jump_relu   16384    -  CSQA   pend        -       -      -  -       -      -      -        -      -  -           -       -    -       -     - (drop)
423  124  gemma3_1b_it            1B  13  0.52  resid  jump_relu   16384    -  ARC    pend        -       -      -  -       -      -      -        -      -  -           -       -    -       -     - (drop)
424  125  gemma3_1b_it            1B  22  0.88  resid  jump_relu   16384    -  CSQA   pend        -       -      -  -       -      -      -        -      -  -           -       -    -       -     - (drop)
425  126  gemma3_1b_it            1B  22  0.88  resid  jump_relu   16384    -  ARC    pend        -       -      -  -       -      -      -        -      -  -           -       -    -       -     - (drop)

355   56  llama32_1b              1B   7  0.47  mlp    batchtopk  131072   32  CSQA   done     3350  +3.70%  0.000  *  +3.67%   5683    711  +15.47%  0.000  *     +18.00%  +3.70%  5/5  +7.70%   5.9
356   57  llama32_1b              1B   7  0.47  mlp    batchtopk  131072   32  ARC    done      771  +1.95%  0.062     -0.39%    998    125  +31.20%  0.000  *     +42.40%  +1.94%  3/5  +2.72%   1.9
304    5  llama32_1b              1B   9  0.60  resid  relu        32768    -  CSQA   done     3304  +0.30%  0.163     +0.42%   5804    726   +2.20%  0.044  *      +2.75%  +0.30%  3/5  +0.43%   1.9
510  211  llama32_1b              1B   9  0.60  resid  relu        32768    -  ARC    done      747  +1.74%  0.005  *  +1.07%   1061    132  +11.36%  0.001  *     +12.12%  +1.74%  4/5  +2.95%   2.9 (run) agree
302    3  llama32_1b              1B  15  1.00  mlp    batchtopk  131072   32  CSQA   done     3304  +1.45%  0.000  *  +1.36%   5804    726   +8.54%  0.000  *      +8.82%  +1.45%  5/5  +5.74%   2.4
357   58  llama32_1b              1B  15  1.00  mlp    batchtopk  131072   32  ARC    done      771  +2.08%  0.004  *  +2.08%    998    125  +14.40%  0.000  *     +14.40%  +2.08%  4/5  +3.85%   2.0

335   36  llama32_1b_base         1B   7  0.47  resid  relu        16384    -  CSQA   done     2487  -0.16%  0.731     -0.04%   7983    998   -0.40%  0.731         -0.10%  -0.16%  3/5  -0.12%   4.9
336   37  llama32_1b_base         1B   7  0.47  resid  relu        16384    -  ARC    done      501  +0.20%  0.500     +0.60%   1718    214   +0.93%  0.416         +1.87%  +0.20%  2/5  -0.20%   1.5
337   38  llama32_1b_base         1B   8  0.53  resid  relu        16384    -  CSQA   done     2487  +0.40%  0.061     +0.28%   7983    998   +1.00%  0.061         +0.70%  +0.40%  4/5  +0.06%   4.4
338   39  llama32_1b_base         1B   8  0.53  resid  relu        16384    -  ARC    done      501  +0.00%  0.656     -0.60%   1718    214   +0.00%  0.656         -1.40%  +0.00%  2/5  -0.05%   1.5
515  216  llama32_1b_base         1B  11  0.73  resid  batchtopk   14336   48  CSQA   done     2487  +0.56%  0.170     +0.08%   7983    998   +1.40%  0.170         +0.20%  +0.56%  4/5  +0.73%   8.5 (run) agree
516  217  llama32_1b_base         1B  11  0.73  resid  batchtopk   14336   48  ARC    done      501  +1.60%  0.076     +1.80%   1718    214   +3.74%  0.076         +4.21%  +1.60%  3/5  +0.06%   2.9 (run) agree

339   40  deepseek_r1_qwen_15b  1.5B  13  0.48  mlp    batchtopk   65536   32  CSQA   done     2436  +0.12%  0.437     +0.16%   8120   1015   +0.30%  0.437         +0.39%  +0.12%  2/5  -0.06%   8.3
340   41  deepseek_r1_qwen_15b  1.5B  13  0.48  mlp    batchtopk   65536   32  ARC    done      633  +1.74%  0.013  *  +1.74%   1367    170   +6.47%  0.013  *      +6.47%  +1.73%  5/5  -0.12%   3.0
341   42  deepseek_r1_qwen_15b  1.5B  27  1.00  mlp    batchtopk   65536   32  CSQA   done     2436  +0.08%  0.387     +0.00%   8120   1015   +0.20%  0.387         +0.00%  +0.08%  1/5  -0.11%   7.0
342   43  deepseek_r1_qwen_15b  1.5B  27  1.00  mlp    batchtopk   65536   32  ARC    done      633  +0.47%  0.125     +0.00%   1367    170   +1.76%  0.125         +0.00%  +0.47%  3/5  -0.07%   2.5

329   30  gemma2_2b               2B   1  0.04  resid  batchtopk   18432   50  CSQA   done     3774  +0.72%  0.000  *  +0.69%   4551    569   +5.10%  0.000  *      +5.10%  +0.72%  4/5  +1.17%  14.6
330   31  gemma2_2b               2B   1  0.04  resid  batchtopk   18432   50  ARC    done      738  +0.81%  0.090     +0.54%   1085    136   +4.41%  0.090         +3.68%  +0.81%  3/5  +0.24%   4.1
321   22  gemma2_2b               2B  12  0.48  resid  jump_relu   32768    -  CSQA   done     3774  +0.64%  0.000  *  +0.56%   4551    569   +4.22%  0.000  *      +3.87%  +0.64%  5/5  +1.16%  16.8
325   26  gemma2_2b               2B  12  0.48  resid  jump_relu   32768    -  CSQA   done     3774  +0.37%  0.010  *  +0.24%   4551    569   +2.64%  0.005  *      +1.76%  +0.37%  4/5  +0.81%  16.4
478  179  gemma2_2b               2B  12  0.48  resid  jump_relu   16384    -  CSQA   done     3774  +0.34%  0.024  *  +0.32%   4551    569   +2.64%  0.008  *      +2.46%  +0.34%  4/5  +0.72%  18.4 (run) agree
322   23  gemma2_2b               2B  12  0.48  resid  jump_relu   32768    -  ARC    done      738  +0.54%  0.212     +0.27%   1085    136   +3.68%  0.133         +1.47%  +0.54%  2/5  +0.43%   4.8
326   27  gemma2_2b               2B  12  0.48  resid  jump_relu   32768    -  ARC    done      738  +0.54%  0.109     +0.00%   1085    136   +2.94%  0.109         +0.00%  +0.54%  2/5  +0.14%   5.0
479  180  gemma2_2b               2B  12  0.48  resid  jump_relu   16384    -  ARC    done      738  +0.14%  0.500     +0.27%   1085    136   +1.47%  0.344         +1.47%  +0.13%  2/5  -0.04%   5.7 (run) agree
474  175  gemma2_2b               2B  12  0.48  mlp    jump_relu   16384    -  CSQA   done     3774  +0.21%  0.092     +0.21%   4551    569   +1.41%  0.092         +1.41%  +0.21%  3/5  +0.43%  19.7 (run) agree
475  176  gemma2_2b               2B  12  0.48  mlp    jump_relu   16384    -  ARC    done      738  -0.14%  0.773     -0.41%   1085    136   -0.74%  0.773         -2.21%  -0.14%  1/5  -0.01%   4.9 (run) agree
312   13  gemma2_2b               2B  19  0.76  resid  jump_relu   16384    -  CSQA   done     3774  +0.26%  0.072     +0.24%   4551    569   +1.93%  0.049  *      +1.76%  +0.27%  3/5  +1.18%  15.4
480  181  gemma2_2b               2B  19  0.76  resid  jump_relu   16384    -  CSQA   done     3774  +0.48%  0.003  *  +0.40%   4551    569   +3.34%  0.002  *      +2.81%  +0.48%  5/5  +1.11%  19.6 (run) agree
511  212  gemma2_2b               2B  19  0.76  resid  batchtopk   18432   64  CSQA   done     3774  +0.48%  0.003  *  +0.45%   4551    569   +3.16%  0.002  *      +3.16%  +0.48%  5/5  +1.25%  18.9 (run) agree
313   14  gemma2_2b               2B  19  0.76  resid  jump_relu   16384    -  ARC    done      738  +0.41%  0.274     +0.41%   1085    136   +2.94%  0.172         +2.94%  +0.41%  2/5  +0.41%   5.0
481  182  gemma2_2b               2B  19  0.76  resid  jump_relu   16384    -  ARC    done      738  +0.00%  0.688     -0.41%   1085    136   +0.00%  0.688         -2.21%  -0.00%  1/5  +0.00%   5.8 (run) agree
512  213  gemma2_2b               2B  19  0.76  resid  batchtopk   18432   64  ARC    done      738  +0.27%  0.387     +0.54%   1085    136   +2.21%  0.274         +2.94%  +0.27%  2/5  +0.49%   5.6 (run) agree
476  177  gemma2_2b               2B  19  0.76  mlp    jump_relu   16384    -  CSQA   done     3774  +0.21%  0.067     +0.11%   4551    569   +1.41%  0.067         +0.70%  +0.21%  4/5  +0.72%  19.0 (run) agree
477  178  gemma2_2b               2B  19  0.76  mlp    jump_relu   16384    -  ARC    done      738  -0.27%  0.855     -0.27%   1085    136   -1.47%  0.855         -1.47%  -0.27%  1/5  +0.31%   5.9 (run) agree
323   24  gemma2_2b               2B  24  0.96  resid  jump_relu   32768    -  CSQA   done     3774  +0.26%  0.061     +0.34%   4551    569   +1.93%  0.040  *      +2.46%  +0.27%  3/5  +0.68%  15.8
327   28  gemma2_2b               2B  24  0.96  resid  jump_relu   32768    -  CSQA   done     3774  +0.26%  0.026  *  +0.16%   4551    569   +1.93%  0.013  *      +1.23%  +0.27%  3/5  +0.59%  15.7
324   25  gemma2_2b               2B  24  0.96  resid  jump_relu   32768    -  ARC    done      738  +0.00%  0.656     -0.41%   1085    136   +0.00%  0.656         -2.21%  +0.00%  1/5  +0.04%   4.5
328   29  gemma2_2b               2B  24  0.96  resid  jump_relu   32768    -  ARC    done      738  -0.41%  0.969     +0.27%   1085    136   -2.21%  0.969         +1.47%  -0.41%  1/5  +0.05%   4.8

517  218  gemma2_2b_it            2B  13  0.52  resid  batchtopk   73728  100  CSQA   pend        -       -      -  -       -      -      -        -      -  -           -       -    -       -     - (run later) depending on gemma2_2b results
518  219  gemma2_2b_it            2B  13  0.52  resid  batchtopk   73728  100  ARC    pend        -       -      -  -       -      -      -        -      -  -           -       -    -       -     - (run later) depending on gemma2_2b results

500  201  gemma_2b                2B   0  0.00  resid  relu        16384    -  CSQA   done     2366  +0.08%  0.447     -0.08%   8306   1038   +0.19%  0.447         -0.19%  +0.08%  4/5  -0.09%   5.7 (run) agree
501  202  gemma_2b                2B   0  0.00  resid  relu        16384    -  ARC    done      442  -0.23%  0.726     +0.00%   1874    233   -0.43%  0.726         +0.00%  -0.22%  2/5  +0.48%   1.9 (run) agree
502  203  gemma_2b                2B   6  0.35  resid  relu        16384    -  CSQA   done     2366  +0.08%  0.437     -0.17%   8306   1038   +0.19%  0.437         -0.39%  +0.08%  2/5  -0.02%   8.4 (run) agree
503  204  gemma_2b                2B   6  0.35  resid  relu        16384    -  ARC    done      442  -0.23%  0.773     +0.45%   1874    233   -0.43%  0.773         +0.86%  -0.22%  1/5  +0.34%   2.3 (run) agree
301    2  gemma_2b                2B  10  0.59  resid  relu        16384    -  CSQA   done     2366  +0.34%  0.340     +0.04%   8306   1038   +0.77%  0.340         +0.10%  +0.34%  2/5  +0.43%   2.3
498  199  gemma_2b                2B  10  0.59  resid  relu        16384    -  ARC    done      442  -1.36%  0.866     +0.68%   1874    233   -2.58%  0.866         +1.29%  -1.37%  2/5  +0.11%   2.0 (run) agree
300    1  gemma_2b                2B  17  1.00  resid  relu        16384    -  CSQA   done     2366  +8.83%  0.000  *  +8.79%   8306   1038   -7.23%  1.000         -7.32%  +8.83%  5/5  -0.70%   2.2
499  200  gemma_2b                2B  17  1.00  resid  relu        16384    -  ARC    done      442  -0.68%  0.739     -1.81%   1874    233   -1.29%  0.739         -3.43%  -0.68%  2/5  +0.91%   2.5 (run) agree

305    6  gemma_2b_it             2B  12  0.71  resid  relu        16384    -  CSQA   done     5370  +0.02%  0.500     +0.09%    296     36  +38.89%  0.000  *     +36.11%  +0.02%  3/5  -0.37%   2.6
504  205  gemma_2b_it             2B  12  0.71  resid  relu        16384    -  ARC    done     1118  +0.45%  0.250     +0.45%     72      9  +66.67%  0.016  *     +66.67%  +0.45%  4/5  -0.84%   4.7 (run later) depending on gemma_2b results

466  167  gemma3_4b               4B  17  0.52  resid  jump_relu   16384    -  CSQA   done     3669  +0.03%  0.500     +0.05%   4832    603   +0.17%  0.500         +0.33%  +0.03%  2/5  +0.05%  21.7
467  168  gemma3_4b               4B  17  0.52  resid  jump_relu   16384    -  ARC    pend        -       -      -  -       -      -      -        -      -  -           -       -    -       -     - (drop)
462  163  gemma3_4b               4B  17  0.52  mlp    jump_relu   16384    -  CSQA   pend        -       -      -  -       -      -      -        -      -  -           -       -    -       -     - (drop)
463  164  gemma3_4b               4B  17  0.52  mlp    jump_relu   16384    -  ARC    pend        -       -      -  -       -      -      -        -      -  -           -       -    -       -     - (drop)
468  169  gemma3_4b               4B  29  0.88  resid  jump_relu   16384    -  CSQA   done     3669  +0.05%  0.363     +0.00%   4832    603   +0.33%  0.363         +0.00%  +0.05%  3/5  -0.05%  24.8
469  170  gemma3_4b               4B  29  0.88  resid  jump_relu   16384    -  ARC    pend        -       -      -  -       -      -      -        -      -  -           -       -    -       -     - (drop)
464  165  gemma3_4b               4B  29  0.88  mlp    jump_relu   16384    -  CSQA   pend        -       -      -  -       -      -      -        -      -  -           -       -    -       -     - (drop)
465  166  gemma3_4b               4B  29  0.88  mlp    jump_relu   16384    -  ARC    pend        -       -      -  -       -      -      -        -      -  -           -       -    -       -     - (drop)

454  155  gemma3_4b_it            4B  17  0.52  resid  jump_relu   16384    -  CSQA   pend        -       -      -  -       -      -      -        -      -  -           -       -    -       -     - (drop)
455  156  gemma3_4b_it            4B  17  0.52  resid  jump_relu   16384    -  ARC    pend        -       -      -  -       -      -      -        -      -  -           -       -    -       -     - (drop)
456  157  gemma3_4b_it            4B  29  0.88  resid  jump_relu   16384    -  CSQA   pend        -       -      -  -       -      -      -        -      -  -           -       -    -       -     - (drop)
457  158  gemma3_4b_it            4B  29  0.88  resid  jump_relu   16384    -  ARC    pend        -       -      -  -       -      -      -        -      -  -           -       -    -       -     - (drop)

519  220  mistral_7b              7B  16  0.52  mlp    batchtopk  131072  128  CSQA   pend        -       -      -  -       -      -      -        -      -  -           -       -    -       -     - (drop)
307    8  mistral_7b              7B  16  0.52  mlp    batchtopk  131072  128  ARC    done     1046  +0.00%  0.750     +0.19%    264     33   +3.03%  0.500         +6.06%  +0.00%  1/5  -0.00%   3.0

317   18  qwen25_7b               7B  15  0.56  resid  batchtopk  131072   64  CSQA   pend        -       -      -  -       -      -      -        -      -  -           -       -    -       -     - (drop)
318   19  qwen25_7b               7B  15  0.56  resid  batchtopk  131072   64  ARC    pend        -       -      -  -       -      -      -        -      -  -           -       -    -       -     - (drop)
319   20  qwen25_7b               7B  27  1.00  resid  batchtopk  131072   64  CSQA   pend        -       -      -  -       -      -      -        -      -  -           -       -    -       -     - (drop)
320   21  qwen25_7b               7B  27  1.00  resid  batchtopk  131072   64  ARC    pend        -       -      -  -       -      -      -        -      -  -           -       -    -       -     - (drop)

508  209  deepseek_r1_llama_8b    8B  19  0.61  resid  relu        65536    -  CSQA   pend        -       -      -  -       -      -      -        -      -  -           -       -    -       -     - (drop)
509  210  deepseek_r1_llama_8b    8B  19  0.61  resid  relu        65536    -  ARC    pend        -       -      -  -       -      -      -        -      -  -           -       -    -       -     - (drop)

314   15  llama31_8b              8B  15  0.48  resid  batchtopk  131072   64  CSQA   done     5123  -0.61%  1.000     -0.61%    953    119   +1.68%  0.344         +1.68%  -0.61%  0/5  -0.37%   9.2
303    4  llama31_8b              8B  15  0.48  resid  batchtopk  131072   64  ARC    done     1076  -0.74%  0.994     -0.65%    185     23  +13.04%  0.125        +26.09%  -0.74%  0/5  -0.29%   1.4
496  197  llama31_8b              8B  19  0.61  resid  relu        65536    -  CSQA   pend        -       -      -  -       -      -      -        -      -  -           -       -    -       -     - (run later) but only if money allows it
497  198  llama31_8b              8B  19  0.61  resid  relu        65536    -  ARC    pend        -       -      -  -       -      -      -        -      -  -           -       -    -       -     - (run later) but only if money allows it
351   52  llama31_8b              8B  23  0.74  resid  batchtopk  262144   32  CSQA   pend        -       -      -  -       -      -      -        -      -  -           -       -    -       -     - (run later) but only if money allows it
352   53  llama31_8b              8B  23  0.74  resid  batchtopk  262144   32  ARC    pend        -       -      -  -       -      -      -        -      -  -           -       -    -       -     - (run later) but only if money allows it
343   44  llama31_8b              8B  23  0.74  mlp    batchtopk  131072  128  CSQA   pend        -       -      -  -       -      -      -        -      -  -           -       -    -       -     - (drop later) depending on other runs
347   48  llama31_8b              8B  23  0.74  mlp    batchtopk  262144   32  CSQA   pend        -       -      -  -       -      -      -        -      -  -           -       -    -       -     - (run later) but only if money allows it
344   45  llama31_8b              8B  23  0.74  mlp    batchtopk  131072  128  ARC    pend        -       -      -  -       -      -      -        -      -  -           -       -    -       -     - (drop later) depending on other runs
348   49  llama31_8b              8B  23  0.74  mlp    batchtopk  262144   32  ARC    pend        -       -      -  -       -      -      -        -      -  -           -       -    -       -     - (run later) but only if money allows it
315   16  llama31_8b              8B  27  0.87  resid  batchtopk  131072   64  CSQA   done     5123  -0.08%  0.927     -0.08%    953    119   +0.00%  0.750         +1.68%  -0.08%  1/5  +0.06%   9.2
316   17  llama31_8b              8B  27  0.87  resid  batchtopk  131072   64  ARC    pend        -       -      -  -       -      -      -        -      -  -           -       -    -       -     - (run later) but only if money allows it
353   54  llama31_8b              8B  29  0.94  resid  batchtopk  262144   32  CSQA   pend        -       -      -  -       -      -      -        -      -  -           -       -    -       -     - (run later) but only if money allows it
354   55  llama31_8b              8B  29  0.94  resid  batchtopk  262144   32  ARC    pend        -       -      -  -       -      -      -        -      -  -           -       -    -       -     - (run later) but only if money allows it
345   46  llama31_8b              8B  29  0.94  mlp    batchtopk  131072  128  CSQA   pend        -       -      -  -       -      -      -        -      -  -           -       -    -       -     - (drop later) depending on other runs
349   50  llama31_8b              8B  29  0.94  mlp    batchtopk  262144   32  CSQA   pend        -       -      -  -       -      -      -        -      -  -           -       -    -       -     - (run later) but only if money allows it
346   47  llama31_8b              8B  29  0.94  mlp    batchtopk  131072  128  ARC    pend        -       -      -  -       -      -      -        -      -  -           -       -    -       -     - (drop later) depending on other runs
350   51  llama31_8b              8B  29  0.94  mlp    batchtopk  262144   32  ARC    pend        -       -      -  -       -      -      -        -      -  -           -       -    -       -     - (run later) but only if money allows it

513  214  llama31_8b_base         8B  23  0.74  resid  batchtopk   16384   80  CSQA   pend        -       -      -  -       -      -      -        -      -  -           -       -    -       -     - (run later) depending on llama31_8b results
514  215  llama31_8b_base         8B  23  0.74  resid  batchtopk   16384   80  ARC    pend        -       -      -  -       -      -      -        -      -  -           -       -    -       -     - (run later) depending on llama31_8b results

310   11  llama3_8b               8B  15  0.48  resid  batchtopk  131072  192  CSQA   done     5272  -0.08%  0.837     -0.15%    556     70   +2.86%  0.312         +1.43%  -0.08%  1/5  +0.25%   9.9
358   59  llama3_8b               8B  15  0.48  resid  batchtopk  131072  192  ARC    done     1109  +0.27%  0.188     +0.09%     97     13   +7.69%  0.500         +7.69%  +0.27%  3/5  -0.01%   2.8
359   60  llama3_8b               8B  30  0.97  resid  batchtopk  131072  192  CSQA   done     5272  +0.02%  0.500     -0.02%    556     70   +0.00%  0.688         +1.43%  +0.02%  2/5  -0.00%   9.2 (run) agree
360   61  llama3_8b               8B  30  0.97  resid  batchtopk  131072  192  ARC    done     1109  +0.18%  0.312     +0.18%     97     13   +7.69%  0.500         +7.69%  +0.18%  2/5  -0.03%   2.6 (run) agree

494  195  gemma2_9b               9B   9  0.22  resid  jump_relu   16384    -  CSQA   pend        -       -      -  -       -      -      -        -      -  -           -       -    -       -     - (drop)
495  196  gemma2_9b               9B   9  0.22  resid  jump_relu   16384    -  ARC    pend        -       -      -  -       -      -      -        -      -  -           -       -    -       -     - (drop)
490  191  gemma2_9b               9B  20  0.49  resid  jump_relu   16384    -  CSQA   done     4784  +0.04%  0.377     -0.04%   1857    232   +1.29%  0.125         -0.43%  +0.04%  2/5  +0.01%  26.9
491  192  gemma2_9b               9B  20  0.49  resid  jump_relu   16384    -  ARC    pend        -       -      -  -       -      -      -        -      -  -           -       -    -       -     - (drop)
488  189  gemma2_9b               9B  20  0.49  mlp    jump_relu   16384    -  CSQA   abrt        -       -      -  -       -      -      -        -      -  -           -       -    -       -     - (drop)
489  190  gemma2_9b               9B  20  0.49  mlp    jump_relu   16384    -  ARC    pend        -       -      -  -       -      -      -        -      -  -           -       -    -       -     - (drop)
492  193  gemma2_9b               9B  31  0.76  resid  jump_relu   16384    -  CSQA   pend        -       -      -  -       -      -      -        -      -  -           -       -    -       -     - (drop)
493  194  gemma2_9b               9B  31  0.76  resid  jump_relu   16384    -  ARC    pend        -       -      -  -       -      -      -        -      -  -           -       -    -       -     - (drop)

482  183  gemma2_9b_it            9B  20  0.49  resid  jump_relu   16384    -  CSQA   pend        -       -      -  -       -      -      -        -      -  -           -       -    -       -     - (drop)
483  184  gemma2_9b_it            9B  20  0.49  resid  jump_relu   16384    -  ARC    pend        -       -      -  -       -      -      -        -      -  -           -       -    -       -     - (drop)
484  185  gemma2_9b_it            9B  31  0.76  resid  jump_relu   16384    -  CSQA   pend        -       -      -  -       -      -      -        -      -  -           -       -    -       -     - (drop)
485  186  gemma2_9b_it            9B  31  0.76  resid  jump_relu   16384    -  ARC    pend        -       -      -  -       -      -      -        -      -  -           -       -    -       -     - (drop)

418  119  gemma3_12b_it          12B  24  0.51  resid  jump_relu   16384    -  CSQA   pend        -       -      -  -       -      -      -        -      -  -           -       -    -       -     - (drop)
419  120  gemma3_12b_it          12B  24  0.51  resid  jump_relu   16384    -  ARC    pend        -       -      -  -       -      -      -        -      -  -           -       -    -       -     - (drop)
420  121  gemma3_12b_it          12B  41  0.87  resid  jump_relu   16384    -  CSQA   pend        -       -      -  -       -      -      -        -      -  -           -       -    -       -     - (drop)
421  122  gemma3_12b_it          12B  41  0.87  resid  jump_relu   16384    -  ARC    pend        -       -      -  -       -      -      -        -      -  -           -       -    -       -     - (drop)
```

**The primary result is the full holdout.** `nHold` = full-holdout questions · `fullEx`/`p`/`s` = full-exact delta, binomial p, significance · `fullAp` = full-approximated delta · Secondary, on the much smaller filtered holdout: `nFlt` = filtered questions · `nFltH` = filtered-holdout questions · `fltEx`/`fltP`/`fltS` = filtered-exact delta, binomial p, significance · `fltAp` = filtered-approximated delta · `kfold`/`+f` = k-fold mean delta and positive folds · `gen` = mean delta over the three generalisation benchmarks · `hrs` = wall-clock the cell took, from its log footer · `!` after a status = result JSON disagrees with the cell definition.

Anything in parentheses after the last column is a hand-written note. Edit those in place — this script reads them back out of the previous table and re-emits them, so regenerating never loses them. Every other character is generated.

Machine-readable form: `final_matrix_summary.jsonl` (190 lines, one JSON object per cell, EXP-id order).
