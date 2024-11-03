# integral

## Descripción
El ejercicio presenta una serie de funciones con implementaciones diferentes de las directivas de OpenMP como: `reduction`, `atomic` y `private` para obtener los primeros dígitos del número PI. 

### reduction
```c
#pragma omp parallel for reduction(+:sum) 
{
    ...
}
```
La directiva `reduction` crea variables privadas del parámetro definido con el resultado de la ejecución de cada hilo y luego las reduce con la operación especificada.

### atomic
```c
#pragma omp atomic
{
    ...
}
```
Asegura que la localización de memoria es accedida 'atómicamente', es decir, no la expone a múltiples escrituras/lecturas.

### atomic
```c
#pragma omp private(x)
{
    ...
}
```
Indica un parámetro que tendrá un valor único y no compartido entre los hilos que ejecutan la zona paralela.

###### Obtenido desde: https://www.openmp.org/wp-content/uploads/OpenMP-API-Specification-5.0.pdf
---
## Output
Dependiendo del método usado, se obtendrán resultados similares al siguiente:
```c
3.141593 took 0.256237
```
La directiva que desarrolló la operación en el menor tiempo resultó ser una unión entre `reduction` y `private`. El método es llamado `integral_better_reduction()`.