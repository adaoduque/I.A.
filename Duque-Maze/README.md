# Duque-Maze
Escrevi um programa com GUI usando a biblioteca *Tkinter*.<br />
Com esse programa, podemos criar ambientes de forma simples, com diferentes dificuldades de resolução. <br />
Eu estava com problemas para resolver um outro problema usando *aprendizagem por reforço*, então resolvi criar esse ambiente de forma que eu não precise alterar muito código e que seja realmente desafiador. <br /><br />

### Dependências
* tensorflow==2.0.0-alpha0
* matplotlib==3.3.4
* pillow==8.1.0
* numpy==1.17.0
* opencv-python==4.5.1.48
* tk==8.6.10

### Configurando o ambiente
* ```I``` - O agente inicia onde tiver a letra I
* ```X``` - Bloqueio no ambiente, funciona como uma parede
* ```" "``` - Espaço, caminho "livre" para o agente
* ```- ``` - Sinal negativo, o jogo termina, o agente perdeu.
* ```+ ``` - Sinal positivo, o jogo termina, o agente venceu.

### Variáveis de configuração
```height``` - Altura da janela do programa.
```width``` - Largura da janela do programa.
```widthSquares``` - Largura e altura dos quadrados que compõem o ambiente.
```episodes``` - Número de episódios que vai rodar o ambiente.
```animate``` - Cada ação do agente pode ser animada com uma cor diferente e exibindo o sentido da ação.
```rewardPositive``` - Recompensa positiva, usada quando o agente chega no ```+```.
```rewardNegative``` - Recompensa negativa, usada quando o agente chega no ```-```.
```rewardEachStep``` - Recompensa que o agente toma a cada passo, exceto quando perde ou ganha o jogo.
```rewardInvalidStep``` - Recompensa que o agente ganha quano faz uma ação não permitida no jogo
```image_dim``` - Dimensão da imagem e quantidade de canais respectivamente. Usada para treinamento com redes comvolucionais (Ainda não implementada).
```environment``` - Matriz com os dados que o ambiente vai usar para desenhar a interface.

# Interação do ambiente com o agente
Como *estado* o ambiente te retorna um array, contendo observações básicas, como a posição atual na matriz e dados de alguns *sensores* que mostram se o próximo passo nas 4 direções está bloequeado (0) ou livre (1):<br />
Estado: ```[1, 0, 1, 0, 0]```
0. Se a ação *para cima* está bloqueada ou não
1. Se a ação *Para a direita* está bloqueada ou não
2. Se a ação *para baixo* está bloqueada ou não
3. Se a ação *Para a esquerda* está bloqueada ou não
4. Posição atual no jogo, iniciando em 0.

Ações:
0. Para cima
1. Para a direita
2. Para baixo
3. Para a esquerda

Com esses dados é possível treinar um modelo nesse ambiente. <br />

 

### Rodando o ambiente 
```
import numpy as np
from environment.Maze import Maze

if __name__ == "__main__":
    # Config for environment
    config = {
        "height": 600,  # Height for canvas
        "width": 600,  # Width for canvas
        "widthSquares": 100,  # Width and height for square
        "episodes": 100000, # Run this number of episodes
        "animate": False, # Animate action
        "rewardPositive": 10,
        "rewardNegative": -10,
        "rewardEachStep": -0.01,
        "rewardInvalidStep": -1,
        "image_dim": (64, 64, 2),
        "environment": np.array([
            [[' '], [' '], [' '], [' '], [' '], [' ']],
            [[' '], ['X'], ['X'], [' '], ['X'], ['X']],
            [['I'], ['X'], [' '], [' '], [' '], [' ']],
            [[' '], ['X'], ['X'], ['X'], ['X'], ['+']],
            [[' '], [' '], [' '], [' '], [' '], [' ']],
            [['-'], ['-'], ['-'], ['-'], ['-'], ['-']]
        ])
    }

    # Instance Game Maze
    appMaze = Maze(config, mode='train-qtable')

    # Run train
    # appMaze.after(2, train)

    # Show Tkinter GUI
    appMaze.mainloop()
```
Essa configuração cria um ambiente semelhante a esse:<br />
![enviroment](https://drive.google.com/uc?export=download&id=1R7NFtksb5m2hd5lCtZ0rnIHn4gGDnewN)

### Plus
Incluí 2 agentes, um baseado em tabela (Q-Table) e outro em rede neural artificial com Tensorflow, ambos funcionando perfeitamente.<br /><br />


Duvidas, criticas ou sugestões: <br />
```Telegram```: @adaoduque<br />
```E-mail```: adaoduquesn@gmail.com