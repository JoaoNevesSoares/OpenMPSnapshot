# Overview.
This is an algorithm to process the product X=V1×V2 of an m×n matrix V1 and an n-dimensional vector V2 in parallel using **OpenMP**. The algorithm assumes m << n, specifically a large dense matrix with m in the tens and n in the millions.


## Parallelization of matrix-vector product (matvec)
Since OpenMP 4.5, array reduction has been possible in C. However, depending on your environment, you may not be able to use it because the OpenMP version is too old. In this section, we introduce a method using temporary variables.

### Main part of the code
```C
#pragma omp parallel private(i)
  {
    for (i = 0; i < m; i++) {
      k = 0.0;
#pragma omp for reduction(+ : k)
      for (j = 0; j < n; j++) {
        k += V1[i * n + j] * V2[j];
      }
      X[i] = k;
#pragma omp barrier
    }
  }
```

### Points

- The ``i`` of the outer loop must be ``private``.

If you forget to specify this, you may end up with an infinite loop or different results depending on the execution time. This is because ``i`` is shared and some thread will execute ``i=0`` and the update by ``i++`` will not be reflected for a long time.
- Proper placement of ``barrier`` synchronization

The absence of ``barrier`` in L11 can also cause erroneous results. This is because another thread executes ``v=0.0`` before assigning ``f[i]=v``, resulting in ```f[i]=v=0.0``` and the wrong value.


Translated with www.DeepL.com/Translator (free version)

# 概要
これはm×n行列V1と，n次元ベクトルV2の積X=V1×V2を```OpenMP```を用いて並列処理するアルゴリズムです．ここではm << n，具体的にはmは数十程度，nは数百万程度の大規模密行列を想定しています．


## 行列ベクトル積（matvec）の並列化
OpenMP4.5からC言語でも配列のreductionが可能になっているようですが，環境によってはOpenMPのバージョンが古くで利用できないこともあると思います．ここでは，一時変数を用いた方法を紹介します．

### コードの主要部分
```C {.line-numbers}
#pragma omp parallel private(i)
  {
    for (i = 0; i < m; i++) {
      k = 0.0;
#pragma omp for reduction(+ : k)
      for (j = 0; j < n; j++) {
        k += V1[i * n + j] * V2[j];
      }
      X[i] = k;
#pragma omp barrier
    }
  }
```

### ポイント

- 外側ループの```i```は```private```指定すること

この指定を忘れると実行時によって無限ループになったり，結果が異なったりしてしまいます．これは，```i```がsharedになっていてどこかのスレッドが```i=0```を実行し```i++```による更新がいつまでたっても反映されなくなってしまうからです．
- ```barrier```同期を適切に配置すること

L11の```barrier```がない場合も結果がおかしくなってしまいます．これは```f[i]=v```を代入する前に他のスレッドが```v=0.0```を実行してしまい，```f[i]=v=0.0```と誤った値が入ってしまうためです．

