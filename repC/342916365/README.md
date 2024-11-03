# Uso de threads em C com interface OpenMP

Este repositório faz referência ao uso de threads em linguagem C por meio  da API OpenMP.

Nesse contexto, a tarefa em questão é utilizar threads para percorrer uma fila estática que contém linhas de um arquivo de texto, e identificar o número de ocorrências de determinada palavra.

## Tecnologias utilizadas

- [Linguagem C](https://docs.microsoft.com/pt-br/cpp/c-language/?view=msvc-160)
- [OpenMP](https://www.openmp.org/#)

## Configuração do ambiente

É necessário antes adicionar o seguinte comando ao compilador, referente ao OpenMP:

~~~ c
-fopenmp
~~~

## Número de núcleos

A quantidade de threads é definida a partir do número de núcleos disponíveis do processador.

~~~ c
#ifdef _WIN32
  #define WIN32_LEAN_AND_MEAN
  #include <windows.h>
#else
  #include <unistd.h>
#endif

int main(){
  #ifdef _WIN32 
    #ifndef _SC_NPROCESSORS_ONLN
      SYSTEM_INFO info;
      GetSystemInfo(&info);
      #define sysconf(a) info.dwNumberOfProcessors
      #define _SC_NPROCESSORS_ONLN
    #endif
  #endif

  //numero de nucleos disponíveis
  long nprocs = sysconf(_SC_NPROCESSORS_ONLN);
  
  return 0;
}
 
~~~

## Execução de threads

A declaração de uma thread com OpenMP é relacionada à seguinte regra:

~~~ c
#pragma omp parallel num_threads(nthreads)
{
  //executados por cada thread
}
~~~

## Zona crítica

A zona crítica serve para definir que as threads serão executadas uma por vez.

~~~ c
#pragma omp parallel num_threads(nprocs-1)
{
  #pragma omp critical
  {
    //executados por cada thread
  }
}
~~~

## Autor
<b>Otávio Augusto Souza Silva</b>

[![Linkedin Badge](https://img.shields.io/badge/-Otávio-blue?style=flat-square&logo=Linkedin&logoColor=white&link=https://www.linkedin.com/in/otaviosilva22/)](https://www.linkedin.com/in/otaviosilva22/)
[![Gmail Badge](https://img.shields.io/badge/-otavio.ssilva22@gmail.com-c14438?style=flat-square&logo=Gmail&logoColor=white&link=mailto:otavio.ssilva22@gmail.com)](mailto:otavio.ssilva22@gmail.com)
