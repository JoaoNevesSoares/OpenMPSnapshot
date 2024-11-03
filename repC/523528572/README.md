# Fibonnaci recursivo paralelizado
> Colaborador: Gabriel Mazzuco ([Github Profile](https://github.com/gabrielmazz))
---

* Implementação do algortimo de Fibonacci recursivo paralelizado
  * Usado OPENMP com suas diretrizes
  * Linguagem utilizada C
  
## Execução:

- Para executar o código, apenas use make
  - A quantidade de 'fibonnaci' é passada pelo próprio makefile via argumentação

## Detalhes do algoritmo:

- Para a resolução do problema de Fibonacci recursivo existem algumas formas de
conseguir executar está tarefa, pesquisando a fundo sobre o tema, existem alguns
métodos para sua execução, o mais comum é o uso de task’s compartilhando suas
variáveis algo como #pragma omp task shared(i). Usando task’s o problema
normalmente dobrava de tempo sendo 1 minuto sequencialmente para 2 minutos com,
sendo inviável, mas a criação de uma árvore de threads usando schedule(dynamics) foi
uma solução bastante viável e reduzindo pela metade do tempo em Fibonacci onde o
número era muito grande. Se torna bem notável quando passamos um número é maior
que Fib (30)
---
