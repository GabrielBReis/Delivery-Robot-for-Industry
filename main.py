import pygame
import random
import heapq
import sys
import argparse
from abc import ABC, abstractmethod

# ==========================
# CLASSES DE PLAYER
# ==========================
class BasePlayer(ABC):
    """
    Classe base para o jogador (robô).
    Para criar uma nova estratégia de jogador, basta herdar dessa classe e implementar o método escolher_alvo.
    """
    def __init__(self, position):
        self.position = position  # Posição no grid [x, y]
        self.cargo = 0            # Número de pacotes atualmente carregados

    @abstractmethod
    def escolher_alvo(self, world):
        """
        Retorna o alvo (posição) que o jogador deseja ir.
        Recebe o objeto world para acesso a pacotes e metas.
        """
        pass

class DefaultPlayer(BasePlayer):

    # Exemplo de como acessar prioridade de um objetivo
    # Se idade > prioridade você começa a levar uma multa de -1 por passo por pacote
    def get_remaining_steps(self, goal, current_steps):
        prioridade = goal["priority"]
        idade = current_steps - goal["created_at"]  # para medir o atraso    
        print(f"Goal em {goal['pos']} tem prioridade {prioridade} e idade {idade}")    
        return prioridade - idade
    """
    Implementação padrão do jogador.
    Se não estiver carregando pacotes (cargo == 0), escolhe o pacote mais próximo.
    Caso contrário, escolhe a meta (entrega) mais próxima.
    """
    def escolher_alvo(self, world, current_steps):
        # Lógica simples 
        sx, sy = self.position
        # Se não estiver carregando pacote e houver pacotes disponíveis:
        if self.cargo == 0 and world.packages:
            best = None
            best_dist = float('inf')
            for pkg in world.packages:
                d = abs(pkg[0] - sx) + abs(pkg[1] - sy)
                if d < best_dist:
                    best_dist = d
                    best = pkg
            return best
        else:
            # Se estiver carregando ou não houver mais pacotes, vai para a meta de entrega (se existir)
            if world.goals:
                best = None
                best_dist = float('inf')
                for goal in world.goals:
                    gx, gy = goal["pos"]
                    d = abs(gx - sx) + abs(gy - sy)
                    if d < best_dist:
                        best_dist = d
                        best = goal["pos"]
                
                steps_for_deadline = self.get_remaining_steps(goal, current_steps)    
                return best
            else:
                return None

# ==========================
# CLASSE WORLD (MUNDO)
# ==========================
class World:
    def __init__(self, seed=None):
        if seed is not None:
            random.seed(seed)
        # Parâmetros do grid e janela
        self.maze_size = 30
        self.width = 600
        self.height = 600
        self.block_size = self.width // self.maze_size

        # Cria uma matriz 2D para planejamento de caminhos:
        # 0 = livre, 1 = obstáculo
        self.map = [[0 for _ in range(self.maze_size)] for _ in range(self.maze_size)]
        # Geração de obstáculos com padrão de linha (assembly line)
        self.generate_obstacles()
        # Gera a lista de paredes a partir da matriz
        self.walls = []
        for row in range(self.maze_size):
            for col in range(self.maze_size):
                if self.map[row][col] == 1:
                    self.walls.append((col, row))

        # Número total de entregas (metas) planejadas ao longo do jogo
        # 2 iniciais + 1 após 2–5 passos + 3 extras com janelas de 10–15 passos = 6
        self.total_items = 6

        # Geração dos locais de coleta (pacotes)
        # Mantemos uma folga de um a mais que o total de entregas
        self.packages = []
        while len(self.packages) < self.total_items + 1:
            x = random.randint(0, self.maze_size - 1)
            y = random.randint(0, self.maze_size - 1)
            if self.map[y][x] == 0 and [x, y] not in self.packages:
                self.packages.append([x, y])

        # Metas (goals) com surgimento ao longo do tempo
        # Estrutura de cada goal: {"pos":[x,y], "priority":int, "created_at":steps_int}
        self.goals = []

        # Cria o jogador usando a classe DefaultPlayer (pode ser substituído por outra implementação)
        self.player = self.generate_player()

        # Inicializa a janela do Pygame
        pygame.init()
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Delivery Bot")

        # Carrega imagens para pacote e meta a partir de arquivos
        self.package_image = pygame.image.load("images/cargo.png")
        self.package_image = pygame.transform.scale(self.package_image, (self.block_size, self.block_size))

        self.goal_image = pygame.image.load("images/operator.png")
        self.goal_image = pygame.transform.scale(self.goal_image, (self.block_size, self.block_size))

        # Cores utilizadas para desenho (caso a imagem não seja usada)
        self.wall_color = (100, 100, 100)
        self.ground_color = (255, 255, 255)
        self.player_color = (0, 255, 0)
        self.path_color = (200, 200, 0)

    def generate_obstacles(self):
        """
        Gera obstáculos com sensação de linha de montagem:
         - Cria vários segmentos horizontais curtos com lacunas.
         - Cria vários segmentos verticais curtos com lacunas.
         - Cria um obstáculo em bloco grande (4x4 ou 6x6) simulando uma estrutura de suporte.
        """
        # Barragens horizontais curtas:
        for _ in range(7):
            row = random.randint(5, self.maze_size - 6)
            start = random.randint(0, self.maze_size - 10)
            length = random.randint(5, 10)
            for col in range(start, start + length):
                if random.random() < 0.7:
                    self.map[row][col] = 1

        # Barragens verticais curtas:
        for _ in range(7):
            col = random.randint(5, self.maze_size - 6)
            start = random.randint(0, self.maze_size - 10)
            length = random.randint(5, 10)
            for row in range(start, start + length):
                if random.random() < 0.7:
                    self.map[row][col] = 1

        # Obstáculo em bloco grande: bloco de tamanho 4x4 ou 6x6.
        block_size = random.choice([4, 6])
        max_row = self.maze_size - block_size
        max_col = self.maze_size - block_size
        top_row = random.randint(0, max_row)
        top_col = random.randint(0, max_col)
        for r in range(top_row, top_row + block_size):
            for c in range(top_col, top_col + block_size):
                self.map[r][c] = 1

    def generate_player(self):
        # Cria o jogador em uma célula livre que não seja de pacote ou meta.
        while True:
            x = random.randint(0, self.maze_size - 1)
            y = random.randint(0, self.maze_size - 1)
            if self.map[y][x] == 0 and [x, y] not in self.packages:
                return DefaultPlayer([x, y])

    def random_free_cell(self):
        # Retorna uma célula livre que não colida com paredes, pacotes, jogador ou metas existentes
        while True:
            x = random.randint(0, self.maze_size - 1)
            y = random.randint(0, self.maze_size - 1)
            occupied = (
                self.map[y][x] == 1 or
                [x, y] in self.packages or
                [x, y] == self.player.position or
                any(g["pos"] == [x, y] for g in self.goals)
            )
            if not occupied:
                return [x, y]

    def add_goal(self, created_at_step):
        pos = self.random_free_cell()
        priority = random.randint(40, 110)
        self.goals.append({"pos": pos, "priority": priority, "created_at": created_at_step})

    def can_move_to(self, pos):
        x, y = pos
        if 0 <= x < self.maze_size and 0 <= y < self.maze_size:
            return self.map[y][x] == 0
        return False

    def draw_world(self, path=None):
        self.screen.fill(self.ground_color)
        # Desenha os obstáculos (paredes)
        for (x, y) in self.walls:
            rect = pygame.Rect(x * self.block_size, y * self.block_size, self.block_size, self.block_size)
            pygame.draw.rect(self.screen, self.wall_color, rect)
        # Desenha os locais de coleta (pacotes) utilizando a imagem
        for pkg in self.packages:
            x, y = pkg
            self.screen.blit(self.package_image, (x * self.block_size, y * self.block_size))
        # Desenha os locais de entrega (metas) utilizando a imagem
        for goal in self.goals:
            x, y = goal["pos"]
            self.screen.blit(self.goal_image, (x * self.block_size, y * self.block_size))
        # Desenha o caminho, se fornecido
        if path:
            for pos in path:
                x, y = pos
                rect = pygame.Rect(x * self.block_size + self.block_size // 4,
                                   y * self.block_size + self.block_size // 4,
                                   self.block_size // 2, self.block_size // 2)
                pygame.draw.rect(self.screen, self.path_color, rect)
        # Desenha o jogador (retângulo colorido)
        x, y = self.player.position
        rect = pygame.Rect(x * self.block_size, y * self.block_size, self.block_size, self.block_size)
        pygame.draw.rect(self.screen, self.player_color, rect)
        pygame.display.flip()

# ==========================
# CLASSE MAZE: Lógica do jogo e planejamento de caminhos (A*)
# ==========================
class Maze:
    def __init__(self, seed=None):
        self.world = World(seed)
        self.running = True
        self.score = 0
        self.steps = 0
        self.delay = 100  # milissegundos entre movimentos
        self.path = []
        self.num_deliveries = 0  # contagem de entregas realizadas

        # Spawn de metas (goals) ao longo do tempo:
        # 2 metas iniciais no passo 0
        self.world.add_goal(created_at_step=0)

        # Fila de intervalos para novas metas:
        # +1 meta após 2–5 passos; +3 metas com intervalos de 10–15 passos entre si
        self.spawn_intervals = [random.randint(2, 5)] + [random.randint(5, 10)] + [random.randint(10, 15) for _ in range(3)]
        self.next_spawn_step = self.spawn_intervals.pop(0)  # passo absoluto do próximo spawn

        # O alvo corrente é fixado até ser alcançado (não muda se surgirem novas metas)
        self.current_target = None

    def heuristic(self, a, b):
        # Distância de Manhattan
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def astar(self, start, goal):
        maze = self.world.map
        size = self.world.maze_size
        neighbors = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        close_set = set()
        came_from = {}
        gscore = {tuple(start): 0}
        fscore = {tuple(start): self.heuristic(start, goal)}
        oheap = []
        heapq.heappush(oheap, (fscore[tuple(start)], tuple(start)))
        while oheap:
            current = heapq.heappop(oheap)[1]
            if list(current) == goal:
                data = []
                while current in came_from:
                    data.append(list(current))
                    current = came_from[current]
                data.reverse()
                return data
            close_set.add(current)
            for dx, dy in neighbors:
                neighbor = (current[0] + dx, current[1] + dy)
                tentative_g = gscore[current] + 1
                if 0 <= neighbor[0] < size and 0 <= neighbor[1] < size:
                    if maze[neighbor[1]][neighbor[0]] == 1:
                        continue
                else:
                    continue
                if neighbor in close_set and tentative_g >= gscore.get(neighbor, 0):
                    continue
                if tentative_g < gscore.get(neighbor, float('inf')) or neighbor not in [i[1] for i in oheap]:
                    came_from[neighbor] = current
                    gscore[neighbor] = tentative_g
                    fscore[neighbor] = tentative_g + self.heuristic(neighbor, goal)
                    heapq.heappush(oheap, (fscore[neighbor], neighbor))
        return []

    def maybe_spawn_goal(self):
        # Spawna metas conforme a agenda de passos
        while self.next_spawn_step is not None and self.steps >= self.next_spawn_step:
            self.world.add_goal(created_at_step=self.steps)
            if self.spawn_intervals:
                self.next_spawn_step += self.spawn_intervals.pop(0)
            else:
                self.next_spawn_step = None  # sem mais spawns

    def delayed_goals_penalty(self):
        # Conta quantas metas abertas estouraram sua prioridade
        delayed = 0
        for g in self.world.goals:
            age = self.steps - g["created_at"]
            if age > g["priority"]:
                delayed += 1
        return delayed  # -1 por goal atrasado

    def get_goal_at(self, pos):
        for g in self.world.goals:
            if g["pos"] == pos:
                return g
        return None

    def idle_tick(self):
        # Um "passo" sem movimento: avança tempo, aplica penalidades e redesenha
        self.steps += 1
        # Custo base por passo
        self.score -= 1
        # Penalidade adicional por metas atrasadas
        self.score -= self.delayed_goals_penalty()
        # Spawns que podem acontecer neste passo
        self.maybe_spawn_goal()
        self.world.draw_world(self.path)
        pygame.time.wait(self.delay)

    def game_loop(self):
        # O jogo termina quando o número de entregas realizadas é igual ao total de itens.
        while self.running:
            if self.num_deliveries >= self.world.total_items:
                self.running = False
                break

            # Spawns podem ocorrer antes mesmo de escolher alvo
            self.maybe_spawn_goal()

            # Escolhe o alvo apenas quando não há alvo corrente
            if self.current_target is None:
                target = self.world.player.escolher_alvo(self.world, self.steps)
                # Se não há nada para fazer agora, aguardamos (tick ocioso) até surgir algo
                if target is None:
                    self.idle_tick()
                    continue
                self.current_target = target

            # Planeja caminho até o alvo corrente
            self.path = self.astar(self.world.player.position, self.current_target)
            if not self.path:
                print("Nenhum caminho encontrado para o alvo", self.current_target)
                self.running = False
                break

            # Segue o caminho calculado (não muda o alvo durante o trajeto)
            for pos in self.path:
                # Move
                self.world.player.position = pos
                self.steps += 1

                # Custo base por movimento
                self.score -= 1

                # Penalidade por metas atrasadas
                self.score -= self.delayed_goals_penalty()

                # Spawns podem ocorrer durante o trajeto
                self.maybe_spawn_goal()

                # Desenha
                self.world.draw_world(self.path)
                pygame.time.wait(self.delay)

                # Eventos do pygame (fechar janela, etc.)
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.running = False
                        break
                if not self.running:
                    break

            if not self.running:
                break

            # Ao chegar ao alvo, processa a coleta ou entrega:
            if self.world.player.position == self.current_target:
                # Se for local de coleta, pega o pacote.
                if self.current_target in self.world.packages:
                    self.world.player.cargo += 1
                    self.world.packages.remove(self.current_target)
                    print("Pacote coletado em", self.current_target, "Cargo agora:", self.world.player.cargo)
                else:
                    # Se for local de entrega e o jogador tiver pelo menos um pacote, entrega.
                    goal = self.get_goal_at(self.current_target)
                    if goal is not None and self.world.player.cargo > 0:
                        self.world.player.cargo -= 1
                        self.num_deliveries += 1
                        self.world.goals.remove(goal)
                        self.score += 50
                        print(
                            f"Pacote entregue em {self.current_target} | "
                            f"Cargo: {self.world.player.cargo} | "
                            f"Priority: {goal['priority']} | "
                            f"Age: {self.steps - goal['created_at']}"
                        )

            # Reset do alvo para permitir nova decisão no próximo ciclo (sem trocar durante o trajeto)
            self.current_target = None

            # Log simples de estado
            delayed_count = sum(1 for g in self.world.goals if (self.steps - g["created_at"]) > g["priority"])
            print(
                f"Passos: {self.steps}, Pontuação: {self.score}, Cargo: {self.world.player.cargo}, "
                f"Entregas: {self.num_deliveries}, Goals ativos: {len(self.world.goals)}, "
                f"Atrasados: {delayed_count}"
            )

        print("Fim de jogo!")
        print("Total de passos:", self.steps)
        print("Pontuação final:", self.score)
        pygame.quit()

# ==========================
# PONTO DE ENTRADA PRINCIPAL
# ==========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Delivery Bot: Navegue no grid, colete pacotes e realize entregas."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Valor do seed para recriar o mesmo mundo (opcional)."
    )
    args = parser.parse_args()

    maze = Maze(seed=args.seed)
    maze.game_loop()

