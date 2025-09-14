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
class BestPlayer(BasePlayer):
    def get_remaining_steps(self, goal, current_steps):
        prioridade = goal["priority"]
        idade = current_steps - goal["created_at"]  # para medir o atraso   
        print(f"Goal em {goal['pos']} tem prioridade {prioridade} e idade {idade}")     
        return prioridade - idade

    """
    Estratégia avançada:
    - Se não estiver carregando pacotes → pega o pacote mais próximo.
    - Se estiver carregando → escolhe meta considerando prioridade e urgência.
    """
    def escolher_alvo(self, world, current_steps):
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
            # 🔹 NOVA REGRA: Se tiver carga e houver mais de uma entrega próxima (<= 3 blocos)
            if self.cargo > 0:
                nearby_goals = [
                    g for g in world.goals
                    if abs(g["pos"][0] - sx) + abs(g["pos"][1] - sy) <= 3
                ]
                if len(nearby_goals) > 1:
                    # entrega primeiro a mais próxima
                    closest = min(nearby_goals, key=lambda g: abs(g["pos"][0] - sx) + abs(g["pos"][1] - sy))
                    return closest["pos"]

            # Estratégia normal: escolher meta com base em prioridade/urgência
            if world.goals:
                best = None
                best_score = float('-inf')
                
                for goal in world.goals:
                    gx, gy = goal["pos"]
                    distancia = abs(gx - sx) + abs(gy - sy)
                    
                    # Calcula urgência
                    tempo_restante = self.get_remaining_steps(goal, current_steps)
                    
                    if tempo_restante < 0:
                        score = 10000 + abs(tempo_restante)  # Altíssima prioridade se atrasado
                    else:
                        score = (100 / (distancia + 1)) + (50 / (tempo_restante + 1))
                    
                    if score > best_score:
                        best_score = score
                        best = goal["pos"]
                
                return best
            else:
                return None

class ClusterSweepPlayer(BestPlayer):
    FORCE_SWEEP_RADIUS = 6

    def _safe_path_length(self, world, a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _overdue_count_at(self, world, steps_at_time):
        cnt = 0
        for g in world.goals:
            age = steps_at_time - g["created_at"]
            if age > g["priority"]:
                cnt += 1
        return cnt

    def _forced_sweep_next(self, world, current_pos):
        if not world.packages:
            return None
        cand = []
        for p in world.packages:
            L = self._safe_path_length(world, current_pos, p)
            if L <= self.FORCE_SWEEP_RADIUS:
                cand.append((L, p))
        if not cand:
            return None
        cand.sort(key=lambda t: t[0])
        return cand[0][1]

    def _forced_sweep_decide(self, world, current_steps):
        if self._overdue_count_at(world, current_steps) > 0:
            return None
        nxt = self._forced_sweep_next(world, self.position)
        if nxt is not None:
            return nxt
        return None

    def escolher_alvo(self, world, current_steps):
        sweep_target = self._forced_sweep_decide(world, current_steps)
        if sweep_target:
            return sweep_target
        return super().escolher_alvo(world, current_steps)
    
# ==========================
# CLASSE WORLD (MUNDO)
# ==========================
class World:
    def __init__(self, seed=None, player_class=DefaultPlayer):
        if seed is not None:
            random.seed(seed)
        self.maze_size = 30
        self.width = 600
        self.height = 600
        self.block_size = self.width // self.maze_size

        self.map = [[0 for _ in range(self.maze_size)] for _ in range(self.maze_size)]
        self.generate_obstacles()
        self.walls = [(col, row) for row in range(self.maze_size) for col in range(self.maze_size) if self.map[row][col] == 1]

        self.total_items = 6
        self.packages = []
        while len(self.packages) < self.total_items + 1:
            x = random.randint(0, self.maze_size - 1)
            y = random.randint(0, self.maze_size - 1)
            if self.map[y][x] == 0 and [x, y] not in self.packages:
                self.packages.append([x, y])

        self.goals = []
        self.player = self.generate_player(player_class)

        pygame.init()
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Delivery Bot")

        self.package_image = pygame.image.load("images/cargo.png")
        self.package_image = pygame.transform.scale(self.package_image, (self.block_size, self.block_size))
        self.goal_image = pygame.image.load("images/operator.png")
        self.goal_image = pygame.transform.scale(self.goal_image, (self.block_size, self.block_size))

        self.wall_color = (100, 100, 100)
        self.ground_color = (255, 255, 255)
        self.player_color = (0, 255, 0)
        self.path_color = (200, 200, 0)
        self.package_color = (0, 0, 255)
        self.goal_color = (255, 0, 0)

    def generate_obstacles(self):
        for _ in range(7):
            row = random.randint(5, self.maze_size - 6)
            start = random.randint(0, self.maze_size - 10)
            length = random.randint(5, 10)
            for col in range(start, start + length):
                if random.random() < 0.7:
                    self.map[row][col] = 1

        for _ in range(7):
            col = random.randint(5, self.maze_size - 6)
            start = random.randint(0, self.maze_size - 10)
            length = random.randint(5, 10)
            for row in range(start, start + length):
                if random.random() < 0.7:
                    self.map[row][col] = 1

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
        return 0 <= x < self.maze_size and 0 <= y < self.maze_size and self.map[y][x] == 0

    def draw_world(self, path=None):
        self.screen.fill(self.ground_color)
        for (x, y) in self.walls:
            rect = pygame.Rect(x * self.block_size, y * self.block_size, self.block_size, self.block_size)
            pygame.draw.rect(self.screen, self.wall_color, rect)
        
        for pkg in self.packages:
            x, y = pkg
            if self.package_image:
                self.screen.blit(self.package_image, (x * self.block_size, y * self.block_size))
            else:
                rect = pygame.Rect(x * self.block_size, y * self.block_size, self.block_size, self.block_size)
                pygame.draw.rect(self.screen, self.package_color, rect)
        
        for goal in self.goals:
            x, y = goal["pos"]
            idade = pygame.time.get_ticks() // 1000 - goal["created_at"]
            urgência = max(0, min(255, int(255 * (idade / goal["priority"]))))
            cor_meta = (255, 255 - urgência, 255 - urgência)
            
            if self.goal_image:
                self.screen.blit(self.goal_image, (x * self.block_size, y * self.block_size))
            else:
                rect = pygame.Rect(x * self.block_size, y * self.block_size, self.block_size, self.block_size)
                pygame.draw.rect(self.screen, cor_meta, rect)
        
        if path:
            for pos in path:
                x, y = pos
                rect = pygame.Rect(x * self.block_size + self.block_size // 4,
                                   y * self.block_size + self.block_size // 4,
                                   self.block_size // 2, self.block_size // 2)
                pygame.draw.rect(self.screen, self.path_color, rect)
        
        x, y = self.player.position
        rect = pygame.Rect(x * self.block_size, y * self.block_size, self.block_size, self.block_size)
        pygame.draw.rect(self.screen, self.player_color, rect)
        
        font = pygame.font.SysFont(None, 24)
        for goal in self.goals:
            x, y = goal["pos"]
            tempo_restante = goal["priority"] - (pygame.time.get_ticks() // 1000 - goal["created_at"])
            text = font.render(str(tempo_restante), True, (0, 0, 0))
            self.screen.blit(text, (x * self.block_size + 5, y * self.block_size + 5))
        
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
        self.delay = 100
        self.path = []
        self.num_deliveries = 0

        self.world.add_goal(created_at_step=0)
        self.world.add_goal(created_at_step=0)

        self.spawn_intervals = [random.randint(2, 5)] + [random.randint(5, 10)] + [random.randint(10, 15) for _ in range(3)]
        self.next_spawn_step = self.spawn_intervals.pop(0)

        self.current_target = None

    def heuristic(self, a, b):
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
        while self.next_spawn_step is not None and self.steps >= self.next_spawn_step:
            self.world.add_goal(created_at_step=self.steps)
            if self.spawn_intervals:
                self.next_spawn_step += self.spawn_intervals.pop(0)
            else:
                self.next_spawn_step = None

    def delayed_goals_penalty(self):
        delayed = 0
        for g in self.world.goals:
            age = self.steps - g["created_at"]
            if age > g["priority"]:
                delayed += 1
        return delayed

    def get_goal_at(self, pos):
        for g in self.world.goals:
            if g["pos"] == pos:
                return g
        return None

    def idle_tick(self):
        self.steps += 1
        self.score -= 1
        self.score -= self.delayed_goals_penalty()
        self.maybe_spawn_goal()
        self.world.draw_world(self.path)
        pygame.time.wait(self.delay)

    def game_loop(self):
        while self.running:
            if self.num_deliveries >= self.world.total_items:
                self.running = False
                break

            self.maybe_spawn_goal()

            if self.current_target is None:
                target = self.world.player.escolher_alvo(self.world, self.steps)
                if target is None:
                    self.idle_tick()
                    continue
                self.current_target = target

            self.path = self.astar(self.world.player.position, self.current_target)
            if not self.path:
                print("Nenhum caminho encontrado para o alvo", self.current_target)
                self.running = False
                break

            for pos in self.path:
                self.world.player.position = pos
                self.steps += 1
                self.score -= 1
                self.score -= self.delayed_goals_penalty()
                self.maybe_spawn_goal()
                self.world.draw_world(self.path)
                pygame.time.wait(self.delay)

                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.running = False
                        break
                if not self.running:
                    break

            if not self.running:
                break

            if self.world.player.position == self.current_target:
                if self.current_target in self.world.packages:
                    self.world.player.cargo += 1
                    self.world.packages.remove(self.current_target)
                    print("Pacote coletado em", self.current_target, "Cargo agora:", self.world.player.cargo)
                else:
                    goal = self.get_goal_at(self.current_target)
                    if goal is not None and self.world.player.cargo > 0:
                        self.world.player.cargo -= 1
                        self.num_deliveries += 1
                        self.world.goals.remove(goal)
                        
                        # tempo_restante = goal["priority"] - (self.steps - goal["created_at"])
                        # bonus = 50 + max(0, tempo_restante)
                        bonus = 50
                        self.score += bonus
                        
                        print(
                            f"Pacote entregue em {self.current_target} | "
                            f"Cargo: {self.world.player.cargo} | "
                            f"Priority: {goal['priority']} | "
                            f"Age: {self.steps - goal['created_at']}"
                            #f"Tempo restante: {tempo_restante} | "
                            f"Bônus: {bonus} | "
                            f"Score: {self.score}"
                        )

                self.current_target = None

            # Log simples de estado
            delayed_count = sum(1 for g in self.world.goals if (self.steps - g["created_at"]) > g["priority"])
            print(
                f"Passos: {self.steps}| Pontuação: {self.score}| Cargo: {self.world.player.cargo}| "
                f"Entregas: {self.num_deliveries}| Goals ativos: {len(self.world.goals)}| "
                f"Atrasados: {delayed_count}"
            )
            
        print(f"Entregas concluídas: {self.num_deliveries}/{self.world.total_items}")
        print("Total de passos:", self.steps)
        print("Pontuação final:", self.score)
        pygame.quit()

# ==========================
# MAIN
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
    parser.add_argument(
        "--player",
        type=str,
        choices=["default", "bestplayer", "clustersweep"],
        default="default",
        help="Escolha o tipo de player: 'default', 'bestplayer' ou 'clustersweep'."
    )

    args = parser.parse_args()

    # Escolhe a classe do player com base no argumento
    if args.player == "default":
        player_class = DefaultPlayer
    elif args.player == "bestplayer":
        player_class = BestPlayer
    else:
        player_class = ClusterSweepPlayer

    maze = Maze(seed=args.seed, player_class=player_class)
    maze.game_loop()