import argparse
import numpy as np


class Particula:
    """
    Representa una partícula en el algoritmo de Optimización por Enjambre de Partículas (PSO).
    Cada partícula tiene una posición en el espacio de soluciones, una velocidad, y una función
    objetivo que evalúa su desempeño en relación al problema de la mochila.

    Atributos
    ---------
    pesos : np.ndarray
        Array de pesos de los ítems de la mochila.
    valores : np.ndarray
        Array de valores de los ítems de la mochila.
    capacidad : int
        Capacidad máxima de la mochila.
    posicion : np.ndarray
        Posición actual de la partícula (vector binario).
    velocidad : np.ndarray
        Velocidad de la partícula (vector).
    velocidad_max : float
        Velocidad máxima que puede tener una partícula.
    valor : int
        Valor de la función objetivo en la posición actual.
    mejor_valor : int
        Mejor valor encontrado por la partícula hasta el momento.
    mejor_posicion : np.ndarray
        Mejor posición encontrada por la partícula hasta el momento.
    inercia : float
        Tendencia de la partícula a mantener su velocidad actual.
    peso_colaborativo : float
        Tasa de aprendizaje colaborativo que dirige la partícula hacia la mejor posición global.
    peso_cognitivo : float
        Tasa de aprendizaje cognitivo que dirige la partícula hacia su mejor posición personal.

    Métodos
    -------
    __init__(pesos, valores, capacidad, velocidad_max, inercia, peso_colaborativo, peso_cognitivo)
        Inicializa una nueva partícula con los parámetros dados.
    __repr__()
        Devuelve una representación en cadena de texto del estado actual de la partícula.
    funcion_objetivo(posicion)
        Evalúa la función objetivo para una posición dada.
    actualizar_velocidad(mejor_posicion_global)
        Actualiza la velocidad de la partícula basada en su propia experiencia y la del enjambre.
    mover_particula()
        Actualiza la posición de la partícula según su velocidad actual.
    actualizar_mejor()
        Actualiza la mejor posición y valor de la partícula si la posición actual es mejor.
    """

    def __init__(
        self, pesos, valores, capacidad, velocidad_max, inercia, peso_colaborativo, peso_cognitivo
    ):
        """
        Inicializa una nueva partícula.

        Parámetros
        ----------
        - pesos : np.ndarray
            Array de pesos de los ítems.
        - valores : np.ndarray
            Array de valores de los ítems.
        - capacidad : int
            Capacidad máxima de la mochila.
        - velocidad_max : float
            Velocidad máxima que puede tener una partícula.
        - inercia : float
            Tendencia de una partícula de tener la misma velocidad.
        - peso_colaborativo : float
            Tasa de aprendizaje colaborativo.
        - peso_cognitivo : float
            Tasa de aprendizaje cognitivo.
        """
        self.pesos = pesos
        self.valores = valores
        self.capacidad = capacidad
        self.posicion = np.random.randint(2, size=len(pesos))
        self.velocidad = np.zeros(len(pesos))
        self.velocidad_max = velocidad_max
        self.valor = self.funcion_objetivo(self.posicion)
        self.mejor_valor = self.valor
        self.mejor_posicion = np.copy(self.posicion)
        self.inercia = inercia
        self.peso_colaborativo = peso_colaborativo
        self.peso_cognitivo = peso_cognitivo

    def __repr__(self):
        """
        Retorna una representación en formato de cadena de texto de la partícula, mostrando su estado actual.
        """
        texto = (
            f"Partícula {id(self)}\n"
            f"-------------------------\n"
            f"Posición: {self.posicion}\n"
            f"Velocidad: {self.velocidad}\n"
            f"Valor función objetivo: {self.valor}\n"
            f"Mejor valor: {self.mejor_valor}\n"
            f"Mejor posición: {self.mejor_posicion}\n"
            f"¿Solución válida?: {'Sí' if self.valor != -1 else 'No'}\n"
            f"-------------------------\n"
        )

        return texto

    def funcion_objetivo(self, posicion):
        """
        Evalúa la función objetivo para una posición dada.

        La función objetivo calcula el valor total de la mochila basado en la posición de los ítems
        seleccionados. Si la capacidad total excede el límite, se penaliza la solución devolviendo un valor
        negativo.

        Parámetros
        ----------
        - posicion : np.ndarray
            Vector binario que indica si un ítem está seleccionado (1) o no (0).

        Retorna
        -------
        - valor_total : int
            El valor total de los ítems seleccionados, o -1 si se excede la capacidad.
        """
        valor_total = np.sum(self.valores * posicion)
        peso_total = np.sum(self.pesos * posicion)

        if peso_total > self.capacidad:
            return -1

        return valor_total

    def actualizar_velocidad(self, mejor_posicion_global):
        """
        Actualiza la velocidad de la partícula utilizando las fórmulas del PSO.

        La nueva velocidad se calcula teniendo en cuenta tres componentes:
        1. La inercia (velocidad anterior).
        2. La atracción hacia la mejor posición global.
        3. La atracción hacia la mejor posición propia de la partícula.

        Parámetros
        ----------
        - mejor_posicion_global : np.ndarray
            La mejor posición global encontrada por el enjambre.
        """
        e_1 = np.random.rand()
        e_2 = np.random.rand()

        componente_velocidad = self.inercia * self.velocidad
        componente_social = e_1 * self.peso_colaborativo * (mejor_posicion_global - self.posicion)
        componente_cognitivo = e_2 * self.peso_cognitivo * (self.mejor_posicion - self.posicion)

        nueva_velocidad = componente_velocidad + componente_social + componente_cognitivo
        self.velocidad = np.clip(nueva_velocidad, None, self.velocidad_max)

    def mover_particula(self):
        """
        Actualiza la posición de la partícula basada en su velocidad actual.

        La posición se actualiza aplicando una función sigmoide a la velocidad,
        convirtiéndola en un valor binario (0 o 1) que indica si un ítem es seleccionado.
        """
        sigmoide = 1 / (1 + np.exp(-self.velocidad))
        self.posicion = (sigmoide > np.random.rand(len(self.velocidad))).astype(int)

    def actualizar_mejor(self):
        """
        Actualiza la mejor posición de la partícula si se ha encontrado una mejor solución.

        Si el valor de la nueva posición es mejor que el anterior, se actualiza la
        mejor posición y el mejor valor encontrado.
        """
        self.valor = self.funcion_objetivo(self.posicion)

        if self.valor > self.mejor_valor:
            self.mejor_valor = self.valor
            self.mejor_posicion = np.copy(self.posicion)


class Enjambre:
    """
    Representa el conjunto de partículas que conforman el enjambre en el algoritmo PSO.
    Gestiona el proceso de optimización, actualizando las partículas y buscando la mejor solución.

    Atributos
    ---------
    - popSize : int
        Número de partículas en el enjambre.
    - particulas : list de Particula
        Lista de objetos `Particula`.
    - pesos : np.ndarray
        Array de pesos de los ítems.
    - valores : np.ndarray
        Array de valores de los ítems.
    - capacidad : int
        Capacidad máxima de la mochila.
    - velocidad_maxima : float
            Velocidad máxima que puede tener una partícula.
    - inercia : float
        Tendencia de una partícula de tener la misma dirección.
    - mejor_global : np.ndarray
        Mejor posición global encontrada por el enjambre.
    - mejor_global_valor : float
        Mejor valor de la función objetivo encontrado por el enjambre.
    - peso_colaborativo : float
        Tasa de aprendizaje colaborativo.
    - peso_cognitivo : float
        Tasa de aprendizaje cognitivo.
    - verbose : bool
        Flag para imprimir información detallada del proceso.

    Métodos
    -------
    __init__(popSize, pesos, valores, capacidad, velocidad_maxima inercia, peso_colaborativo, peso_cognitivo, verbose)
        Inicializa el enjambre de partículas con los parámetros dados.
    actualizar_mejor_global()
        Actualiza la mejor posición global del enjambre.
    optimizar(maxIter)
        Ejecuta el proceso de optimización PSO durante un número máximo de iteraciones.
    """

    def __init__(
        self,
        popSize,
        pesos,
        valores,
        capacidad,
        velocidad_maxima,
        inercia,
        peso_colaborativo,
        peso_cognitivo,
        verbose,
    ):
        """
        Inicializa el enjambre de partículas.

        Parámetros
        ----------
        - popSize : int
            Número de partículas en el enjambre.
        - pesos : np.ndarray
            Array de pesos de los ítems.
        - valores : np.ndarray
            Array de valores de los ítems.
        - capacidad : int
            Capacidad máxima de la mochila.
        - velocidad_maxima : float
            Velocidad máxima que puede tener una partícula.
        - inercia : float
            Tendencia de una partícula de tener la misma dirección.
        - peso_colaborativo : float
            Tasa de aprendizaje colaborativo.
        - peso_cognitivo : float
            Tasa de aprendizaje cognitivo.
        - verbose : bool
            Si es True, imprime información detallada del proceso.
        """
        self.particulas = [
            Particula(
                pesos,
                valores,
                capacidad,
                velocidad_maxima,
                inercia,
                peso_colaborativo,
                peso_cognitivo,
            )
            for _ in range(popSize)
        ]
        self.mejor_global = np.zeros(len(pesos))
        self.mejor_global_valor = 0
        self.verbose = verbose

    def actualizar_mejor_global(self):
        """
        Actualiza la mejor posición global del enjambre.

        Se compara el mejor valor de cada partícula con el mejor valor global
        encontrado hasta el momento.
        """
        for particula in self.particulas:
            if particula.mejor_valor > self.mejor_global_valor:
                self.mejor_global_valor = particula.mejor_valor
                self.mejor_global = np.copy(particula.mejor_posicion)

    def optimizar(self, maxIter):
        """
        Optimiza la función objetivo utilizando el algoritmo PSO.

        Este método ejecuta el proceso de optimización durante un número máximo de iteraciones.
        En cada iteración, se actualizan las velocidades y las posiciones de las partículas.
        Además cada partícula evalúa su desempeño y actualiza su mejor solución hasta el
        momento. Al terminar, se retorna la mejor solución global encontrada por el enjambre.

        Parámetros
        ----------
        - maxIter : int
            Número máximo de iteraciones a realizar.

        Retorna
        -------
        - mejor_global_valor : float
            Mejor valor de la función objetivo encontrado.
        - mejor_global : np.ndarray
            Mejor posición encontrada por el enjambre.

        El proceso incluye:
        1. Evaluación y actualización de las velocidades y posiciones de las partículas.
        2. Evaluación de la función objetivo de cada partícula.
        3. Actualización de las mejores posiciones locales y globales.

        Si el parámetro `verbose` está activado (True), se imprimirá información detallada de
        cada iteración y del estado de las partículas.
        """
        self.actualizar_mejor_global()

        for i in range(maxIter):
            if self.verbose:
                print(f"Iteración {i + 1}")
                print("--------------------------------------\n")

            for particula in self.particulas:
                if self.verbose:
                    print(particula)

                particula.actualizar_velocidad(self.mejor_global)
                particula.mover_particula()
                particula.actualizar_mejor()

            self.actualizar_mejor_global()

        if self.verbose:
            print("-----------------------------------------------------------------")
            print("-----------------------------------------------------------------\n")

        return self.mejor_global, self.mejor_global_valor


def parse_args():
    """
    Procesa los argumentos de la línea de comandos para configurar los parámetros del algoritmo
    PSO en el problema de la mochila. Esta función utiliza `argparse` para interpretar los
    parámetros y retornar un objeto con los valores configurados.

    Los argumentos posibles incluyen:
    -----------
    --verbose : bool
        Un valor booleano que indica si se debe habilitar la salida detallada del algoritmo.
    --maxIter : int
        Un entero que define el número máximo de iteraciones del algoritmo.
    --popSize : int
        Un entero que especifica el tamaño de la población (número de partículas).
    - velocidadMax : float
        Velocidad máxima que puede tener una partícula.
    --inercia : float
        Un valor flotante que representa la inercia en el algoritmo de optimización.
    --alpha : float
        Un entero que define la tasa de aprendizaje colaborativo.
    --beta : float
        Un entero que define la tasa de aprendizaje cognitivo.
    --random : bool
        Un valor booleano que indica si los datos del problema deben generarse aleatoriamente.

    Retorna
    -------
    - Namespace
        Un objeto con los argumentos parseados desde la línea de comandos.
    """
    parser = argparse.ArgumentParser(description="Problema de la mochila con PSO.")

    parser.add_argument(
        "--verbose",
        type=bool,
        default=False,
        help="Habilita la salida detallada del algoritmo.",
    )

    parser.add_argument(
        "--maxIter",
        type=int,
        default=1,
        help="Número de iteraciones.",
    )

    parser.add_argument(
        "--popSize",
        type=int,
        default=3,
        help="Tamaño de la población (número de partículas).",
    )

    parser.add_argument(
        "--velocidadMax",
        type=float,
        default=3,
        help="Velocidad máxima que puede tener una partícula.",
    )

    parser.add_argument(
        "--inercia",
        type=float,
        default=0.7,
        help="Tendencia de una partícula de cambiar su velocidad.",
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=2.0,
        help="Tasa de aprendizaje colaborativo.",
    )

    parser.add_argument(
        "--beta",
        type=float,
        default=2.0,
        help="Tasa de aprendizaje cognitivo.",
    )

    parser.add_argument(
        "--random",
        type=bool,
        default=False,
        help="Crear de manera aleatoria los datos del problema.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if args.popSize < 2:
        print("El tamaño de la población (popSize) debe ser al menos 2.")
        exit(1)

    if args.random:
        num_objetos = np.random.randint(5, 10)
        pesos = np.random.randint(20, 30, size=num_objetos)
        valores = np.random.randint(15, 25, size=num_objetos)
        capacidad = np.random.randint(50, 100)
    else:
        pesos = np.array([2, 3, 4, 5, 9, 7, 6, 8, 3, 6])
        valores = np.array([3, 4, 5, 8, 10, 7, 9, 6, 2, 5])
        capacidad = 20

    enjambre = Enjambre(
        args.popSize,
        pesos,
        valores,
        capacidad,
        args.velocidadMax,
        args.inercia,
        args.alpha,  # Peso colaborativo
        args.beta,  # Peso cognitivo
        args.verbose,
    )
    mejor_posicion, mejor_valor = enjambre.optimizar(args.maxIter)

    print("Mejor posición:", mejor_posicion)
    print("Valor total de la mochila:", mejor_valor)


if __name__ == "__main__":
    main()
