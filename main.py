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
    def escolher_alvo(self, world, current_steps):
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


class ClusterSweepPlayer(BasePlayer):
    """
    Implementação padrão do jogador.
    Prioriza pegar pacotes adicionais em um raio próximo antes de ir para a entrega.
    """
    def escolher_alvo(self, world, current_steps):
        sx, sy = self.position
        
        # Lógica de decisão
        if self.cargo == 0:
            # Se não tem pacote, busca o pacote mais próximo
            if world.packages:
                best = None
                best_dist = float('inf')
                for pkg in world.packages:
                    d = abs(pkg[0] - sx) + abs(pkg[1] - sy)
                    if d < best_dist:
                        best_dist = d
                        best = pkg
                return best
            else:
                return None
        else: # self.cargo > 0
            # Se já tem um pacote, verifica se há outro próximo para pegar
            nearby_packages = []
            for pkg in world.packages:
                d = abs(pkg[0] - sx) + abs(pkg[1] - sy)
                if d <= 6:  # Raio de 6 blocos
                    nearby_packages.append(pkg)

            if nearby_packages:
                # Se encontrou pacotes próximos, vai para o mais perto deles
                best_pkg = None
                best_dist = float('inf')
                for pkg in nearby_packages:
                    d = abs(pkg[0] - sx) + abs(pkg[1] - sy)
                    if d < best_dist:
                        best_dist = d
                        best_pkg = pkg
                return best_pkg
            
            else:
                # Se não encontrou pacotes próximos, vai para a entrega mais prioritária
                if world.goals:
                    best_goal = None
                    best_cost = float('inf')

                    for goal in world.goals:
                        gx, gy = goal["pos"]
                        distance = abs(gx - sx) + abs(gy - sy)
                        age = current_steps - goal["created_at"]
                        priority_limit = goal["priority"]
                        
                        urgency_penalty = max(0, age - priority_limit) * 1.5
                        cost = distance + urgency_penalty
                        
                        if cost < best_cost:
                            best_cost = cost
                            best_goal = goal["pos"]
                    
                    return best_goal
                else:
                    return None

# ==========================
# CLASSE WORLD (MUNDO)
# ==========================
class World:
    def __init__(self, seed=None, player_class=DefaultPlayer):
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
        
        self.player = self.generate_player(player_class)


        # Inicializa a janela do Pygame
        pygame.init()
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Delivery Bot")

        # Carrega imagens para pacote e meta a partir de arquivos
        try:
            self.package_image = pygame.image.load("images/cargo.png")
            self.package_image = pygame.transform.scale(self.package_image, (self.block_size, self.block_size))

            self.goal_image = pygame.image.load("images/operator.png")
            self.goal_image = pygame.transform.scale(self.goal_image, (self.block_size, self.block_size))
        except pygame.error:
            print("Não foi possível carregar as imagens. Certifique-se de que os arquivos 'images/cargo.png' e 'images/operator.png' existem.")
            self.package_image = None
            self.goal_image = None

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

    def generate_player(self, player_class):
        while True:
            x = random.randint(0, self.maze_size - 1)
            y = random.randint(0, self.maze_size - 1)
            if self.map[y][x] == 0 and [x, y] not in self.packages:
                return player_class([x, y])

    def random_free_cell(self):
        # Retorna uma célula livre que não colide com paredes, pacotes, jogador ou metas existentes
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
        if self.package_image:
            for pkg in self.packages:
                x, y = pkg
                self.screen.blit(self.package_image, (x * self.block_size, y * self.block_size))
        else: # Fallback para cor
            for pkg in self.packages:
                x, y = pkg
                rect = pygame.Rect(x * self.block_size, y * self.block_size, self.block_size, self.block_size)
                pygame.draw.rect(self.screen, (255, 165, 0), rect) # Laranja

        # Desenha os locais de entrega (metas) utilizando a imagem
        if self.goal_image:
            for goal in self.goals:
                x, y = goal["pos"]
                self.screen.blit(self.goal_image, (x * self.block_size, y * self.block_size))
        else: # Fallback para cor
            for goal in self.goals:
                x, y = goal["pos"]
                rect = pygame.Rect(x * self.block_size, y * self.block_size, self.block_size, self.block_size)
                pygame.draw.rect(self.screen, (0, 0, 255), rect) # Azul

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
    def __init__(self, seed=None, player_class=DefaultPlayer):
        self.world = World(seed, player_class)
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
                            #f"Tempo restante: {tempo_restante} | "
                            f"Score: {self.score}"
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


import pandas as pd 
import matplotlib
matplotlib.use('Agg')  # evita problemas em ambientes sem GUI
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class Simulator:
    def __init__(self, player_class, num_runs=40, seed=None):
        self.player_class = player_class
        self.num_runs = num_runs
        self.seed = seed
        self.results = []

    def run_single_game(self, seed=None):
        maze = Maze(seed=seed, player_class=self.player_class)
        maze.delay = 0  # acelera simulação
        maze.running = True
        pygame.display.set_mode((1,1), pygame.NOFRAME)
        maze.game_loop()
        return {
            "seed": seed,
            "score": maze.score,
            "steps": maze.steps,
            "deliveries": maze.num_deliveries,
            "delayed_goals": maze.delayed_goals_penalty()
        }

    def run(self, output_csv="simulation_results.csv"):
        for i in range(1, self.num_runs + 1):
            run_seed = None
            if self.seed is not None:
                run_seed = self.seed + i
            print(f"Rodando simulação {i}/{self.num_runs} com seed {run_seed}...")
            result = self.run_single_game(seed=run_seed)
            result["run"] = i
            self.results.append(result)

        df = pd.DataFrame(self.results)
        df.to_csv(output_csv, index=False)
        print(f"Simulações concluídas. Resultados salvos em {output_csv}")
        return df

    # ==========================
    # Análise Estatística
    # ==========================
    @staticmethod
    def summary_stats(csv_file, label):
        df = pd.read_csv(csv_file)
        desc = df.describe()
        print(f"\n===== Estatísticas ({label}) =====")
        print(desc)
        return desc

    # ==========================
    # NOVOS GRÁFICOS
    # ==========================
    @staticmethod
    def compare_boxplot_with_points(csv_files, labels, column, ylabel, title, save_as):
        dfs = [pd.read_csv(f).assign(Jogador=label) for f, label in zip(csv_files, labels)]
        df_all = pd.concat(dfs)

        plt.figure(figsize=(8,5))
        sns.boxplot(data=df_all, x="Jogador", y=column, showfliers=False)
        sns.stripplot(data=df_all, x="Jogador", y=column, color="black", alpha=0.5)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.savefig(save_as)
        plt.close()

    @staticmethod
    def compare_violinplot(csv_files, labels, column, ylabel, title, save_as):
        dfs = [pd.read_csv(f).assign(Jogador=label) for f, label in zip(csv_files, labels)]
        df_all = pd.concat(dfs)

        plt.figure(figsize=(8,5))
        sns.violinplot(data=df_all, x="Jogador", y=column, inner="quartile")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.savefig(save_as)
        plt.close()

    @staticmethod
    def compare_line_per_seed(csv_files, labels, column, ylabel, title, save_as):
        df1, df2 = [pd.read_csv(f) for f in csv_files]
        plt.figure(figsize=(10,5))
        plt.plot(df1["run"], df1[column], marker="o", label=labels[0])
        plt.plot(df2["run"], df2[column], marker="s", label=labels[1])
        plt.xlabel("Execução (seed)")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend()
        plt.savefig(save_as)
        plt.close()

    @staticmethod
    def compare_difference_barplot(csv_files, labels, column, ylabel, title, save_as):
        df1, df2 = [pd.read_csv(f) for f in csv_files]
        merged = pd.merge(df1, df2, on="run", suffixes=(f"_{labels[0]}", f"_{labels[1]}"))
        merged["diff"] = merged[f"{column}_{labels[1]}"] - merged[f"{column}_{labels[0]}"]

        plt.figure(figsize=(10,5))
        plt.bar(merged["run"], merged["diff"], color=np.where(merged["diff"]>=0, "green", "red"))
        plt.axhline(0, color="black", linestyle="--")
        plt.xlabel("Execução (seed)")
        plt.ylabel(f"Diferença de {ylabel} ({labels[1]} - {labels[0]})")
        plt.title(title)
        plt.savefig(save_as)
        plt.close()

    @staticmethod
    def compare_histograms(csv_files, labels, column, xlabel, title, save_as):
        """
        Plota histogramas sobrepostos (mesmos bins) para os arquivos CSV informados.
        Mantém a assinatura que seu código chama: compare_histograms(csv_files, labels, column, ...)
        """
        # certifique-se de ter importado: import numpy as np, import pandas as pd, import matplotlib.pyplot as plt
        dfs = [pd.read_csv(f) for f in csv_files]

        # se a coluna não existir em algum CSV, avisa e retorna
        for i, df in enumerate(dfs):
            if column not in df.columns:
                raise ValueError(f"Coluna '{column}' não encontrada no arquivo {csv_files[i]}")

        # calcula bins comuns
        col_min = min(df[column].min() for df in dfs)
        col_max = max(df[column].max() for df in dfs)
        if col_min == col_max:
            # caso todos tenham o mesmo valor, cria uma pequena faixa para o hist
            col_min -= 0.5
            col_max += 0.5
        bins = np.linspace(col_min, col_max, 12)  # 11 intervals (ajustável)

        plt.figure(figsize=(8,5))
        for df, label in zip(dfs, labels):
            plt.hist(df[column], bins=bins, alpha=0.5, label=label, edgecolor='black')
        plt.xlabel(xlabel)
        plt.ylabel("Frequência")
        plt.title(title)
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_as)
        plt.close()

    @staticmethod
    def compare_histograms(csv_files, labels, column, xlabel, title, save_as):
        dfs = [pd.read_csv(f) for f in csv_files]
        plt.figure(figsize=(8,5))
        for df, label in zip(dfs, labels):
            plt.hist(df[column], bins=10, alpha=0.5, label=label)
        plt.xlabel(xlabel)
        plt.ylabel("Frequência")
        plt.title(title)
        plt.legend()
        plt.savefig(save_as)
        plt.close()

    @staticmethod
    def compare_boxplot(csv_files, labels, column, ylabel, title, save_as):
        dfs = [pd.read_csv(f) for f in csv_files]
        data = [df[column] for df in dfs]
        plt.figure(figsize=(8,5))
        plt.boxplot(data, labels=labels)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.savefig(save_as)
        plt.close()

    @staticmethod
    def compare_seed_scatter(csv_files, labels, save_as):
        df1, df2 = [pd.read_csv(f) for f in csv_files]
        merged = pd.merge(df1, df2, on="run", suffixes=(f"_{labels[0]}", f"_{labels[1]}"))

        plt.figure(figsize=(6,6))
        plt.scatter(merged[f"score_{labels[0]}"], merged[f"score_{labels[1]}"], alpha=0.7)
        plt.plot([merged[f"score_{labels[0]}"].min(), merged[f"score_{labels[0]}"].max()],
                 [merged[f"score_{labels[0]}"].min(), merged[f"score_{labels[0]}"].max()],
                 color="red", linestyle="--", label="y=x")
        plt.xlabel(f"Pontuação {labels[0]}")
        plt.ylabel(f"Pontuação {labels[1]}")
        plt.title("Comparação Pareada por Seed")
        plt.legend()
        plt.savefig(save_as)
        plt.close()

    @staticmethod
    def compare_deliveries_barplot(csv_files, labels, save_as):
        dfs = [pd.read_csv(f) for f in csv_files]
        means = [df["deliveries"].mean() for df in dfs]

        plt.figure(figsize=(6,5))
        plt.bar(labels, means, color=["skyblue", "orange"])
        plt.ylabel("Entregas Médias")
        plt.title("Número Médio de Entregas por Jogador")
        plt.savefig(save_as)
        plt.close()

    @staticmethod
    def compare_difference_histogram(csv_files, labels, column, ylabel, title, save_as):
        df1, df2 = [pd.read_csv(f) for f in csv_files]
        merged = pd.merge(df1, df2, on="run", suffixes=(f"_{labels[0]}", f"_{labels[1]}"))
        merged["diff"] = merged[f"{column}_{labels[1]}"] - merged[f"{column}_{labels[0]}"]

        plt.figure(figsize=(8,5))
        plt.hist(merged["diff"], bins=10, alpha=0.7, color="purple")
        plt.axvline(0, color="black", linestyle="--")
        plt.xlabel(f"Diferença de {ylabel} ({labels[1]} - {labels[0]})")
        plt.ylabel("Frequência")
        plt.title(title)
        plt.savefig(save_as)
        plt.close()
    
    @staticmethod
    def compare_histograms(csv_files, labels, column, xlabel, title, save_as):
        dfs = [pd.read_csv(f) for f in csv_files]
        plt.figure(figsize=(8,5))
        for df, label in zip(dfs, labels):
            plt.hist(df[column], bins=10, alpha=0.5, label=label)
        plt.xlabel(xlabel)
        plt.ylabel("Frequência")
        plt.title(title)
        plt.legend()
        plt.savefig(save_as)
        plt.close()

    @staticmethod
    def compare_boxplot(csv_files, labels, column, ylabel, title, save_as):
        dfs = [pd.read_csv(f) for f in csv_files]
        data = [df[column] for df in dfs]
        plt.figure(figsize=(8,5))
        plt.boxplot(data, tick_labels=labels)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.savefig(save_as)
        plt.close()

    @staticmethod
    def compare_scatter(csv_files, labels, save_as):
        dfs = [pd.read_csv(f) for f in csv_files]
        plt.figure(figsize=(8,5))
        for df, label in zip(dfs, labels):
            plt.scatter(df["steps"], df["score"], alpha=0.7, label=label)
        plt.xlabel("Passos")
        plt.ylabel("Pontuação")
        plt.title("Pontuação x Passos (Comparação)")
        plt.legend()
        plt.savefig(save_as)
        plt.close()



# ==========================
# MAIN
# ==========================
if __name__ == "__main__":
    # Rodar simulações
    sim_default = Simulator(DefaultPlayer, num_runs=40, seed=0)
    csv_default = "default_results.csv"
    sim_default.run(output_csv=csv_default)

    sim_cluster = Simulator(ClusterSweepPlayer, num_runs=40, seed=0)
    csv_cluster = "cluster_results.csv"
    sim_cluster.run(output_csv=csv_cluster)

    # Estatísticas descritivas
    Simulator.summary_stats(csv_default, "Default")
    Simulator.summary_stats(csv_cluster, "ClusterSweep")

    # 🔹 Novos gráficos úteis
    Simulator.compare_boxplot_with_points(
        [csv_default, csv_cluster],
        ["Default", "ClusterSweep"],
        column="score",
        ylabel="Pontuação",
        title="Distribuição de Pontuação com Pontos Individuais",
        save_as="comparison_boxplot_points.png"
    )

    Simulator.compare_violinplot(
        [csv_default, csv_cluster],
        ["Default", "ClusterSweep"],
        column="score",
        ylabel="Pontuação",
        title="Distribuição de Pontuação (Violin Plot)",
        save_as="comparison_violinplot.png"
    )

    Simulator.compare_line_per_seed(
        [csv_default, csv_cluster],
        ["Default", "ClusterSweep"],
        column="score",
        ylabel="Pontuação",
        title="Pontuação por Execução (Seed)",
        save_as="comparison_line_per_seed.png"
    )

    Simulator.compare_difference_barplot(
        [csv_default, csv_cluster],
        ["Default", "ClusterSweep"],
        column="score",
        ylabel="Pontuação",
        title="Diferença de Pontuação por Execução",
        save_as="comparison_difference_barplot.png"
    )

    Simulator.compare_difference_histogram(
        [csv_default, csv_cluster],
        ["Default", "ClusterSweep"],
        column="score",
        ylabel="Pontuação",
        title="Histograma das Diferenças de Pontuação",
        save_as="comparison_difference_histogram.png"
    )

    # Comparações gráficas
    Simulator.compare_histograms(
        [csv_default, csv_cluster],
        ["Default", "ClusterSweep"],
        column="score",
        xlabel="Pontuação",
        title="Distribuição de Pontuação",
        save_as="comparison_score_histogram.png"
    )

    Simulator.compare_histograms(
        [csv_default, csv_cluster],
        ["Default", "ClusterSweep"],
        column="deliveries",
        xlabel="Entregas concluídas",
        title="Distribuição de Entregas",
        save_as="comparison_deliveries_histogram.png"
    )

    Simulator.compare_boxplot(
        [csv_default, csv_cluster],
        ["Default", "ClusterSweep"],
        column="score",
        ylabel="Pontuação",
        title="Boxplot de Pontuação entre Jogadores",
        save_as="comparison_score_boxplot.png"
    )

    Simulator.compare_boxplot(
        [csv_default, csv_cluster],
        ["Default", "ClusterSweep"],
        column="deliveries",
        ylabel="Entregas",
        title="Boxplot de Entregas entre Jogadores",
        save_as="comparison_deliveries_boxplot.png"
    )

    Simulator.compare_scatter(
        [csv_default, csv_cluster],
        ["Default", "ClusterSweep"],
        save_as="comparison_score_vs_steps.png"
    )



    # parser = argparse.ArgumentParser(
    #     description="Delivery Bot: Navegue no grid, colete pacotes e realize entregas."
    # )
    # parser.add_argument(
    #     "--seed",
    #     type=int,
    #     default=None,
    #     help="Valor do seed para recriar o mesmo mundo (opcional)."
    # )
    # parser.add_argument(
    #     "--player",
    #     type=str,
    #     choices=["default", "bestplayer", "clustersweep"],
    #     default="default",
    #     help="Escolha o tipo de player: 'default', 'bestplayer' ou 'clustersweep'."
    # )

    # args = parser.parse_args()

    # # Escolhe a classe do player com base no argumento
    # if args.player == "default":
    #     player_class = DefaultPlayer
    # elif args.player == "bestplayer":
    #     player_class = BestPlayer
    # else:
    #     player_class = ClusterSweepPlayer

    # maze = Maze(seed=args.seed, player_class=player_class)
    # maze.game_loop()