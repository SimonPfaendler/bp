# bachelorproject_simon_pfaendler
**Benchmarking Deep Reinforcement Learning in RoboCup SSL: An Analysis of Action Spaces, Reward Shaping, and Algorithmic Performance in 1v1 Scenarios**



https://github.com/user-attachments/assets/3fa07e65-bb2d-4827-990b-79da1eaf4871



## Projektbeschreibung
Dieses Repository enthält den Code für meine Bachelorprojekts am Neurorobotics Lab Freiburg. Ziel des Projekts ist es, kontinuierliche RL-Algorithmen (wie SAC und CrossQ) in einer 1-gegen-1-Situation in der `rSoccer`-Umgebung zu trainieren und gegen komplexe handgeschriebene Heuristiken zu evaluieren. 

Dabei wird insbesondere der Einfluss von vordefinierten Skills im Vergleich zu rohen Low-Level-Motorbefehlen (Action Spaces) sowie die Auswirkung von Dense vs. Sparse Rewards untersucht.

## Hauptmerkmale & Architektur

* **High-Performance Vektorisierung:** Die Umgebung nutzt `SubprocVecEnv` für asynchrones Multiprocessing. 24 CPU-Kerne berechnen parallel die Physik-Engine (`rSim`), um die Experience Collection drastisch zu beschleunigen.
* **Batched Inference:** Das Modell-Update und die Aktionsfindung finden asynchron über Batched Processing auf einer Nvidia H100 GPU statt, was Trainingsgeschwindigkeiten von bis zu **2.500 FPS** ermöglicht.
* **Curriculum Learning:** Der Agent wird schrittweise an komplexe Situationen herangeführt (Level 1: Penalty, Level 2: Free Ball, Level 3: 1v1 Attack, Level 4: Full Game Chaos).
* **Dynamische Heuristiken:** Der Agent evaluiert gegen einen adaptiven Baseline, der dynamisch zwischen Torwart-Verteidigung, Interceptor-Modus und aggressivem Pressing wechselt.
* **Optimierter Observation Space:** Nutzung egozentrischer Koordinaten und relativer Winkel für stabile Lernkurven.

## System Performance & Profiling
Eine Laufzeitanalyse mittels `cProfile` bestätigt die Effizienz der Architektur. Der System-Overhead der eigenen Python-Umgebung (`env`) inklusive komplexem Reward-Shaping ist minimal. Der Großteil der Rechenzeit fließt deterministisch in die physikalische Simulation (`rSim` / CPU) und die Matrixmultiplikationen des Deep Learning Frameworks (PyTorch / GPU).

## Installation & Setup

1. **Repository klonen:**
   ```bash
   git clone [https://github.com/SimonPfaendler/bp.git](https://github.com/SimonPfaendler/bp.git)
 
## TODO
Dieses Projekt befindet sich in der aktiven Entwicklungs- und Evaluierungsphase. Die nächsten geplanten Schritte sind:

* **Upgrade der Baseline-Heuristik:** Erweiterung des blauen gegnerischen Roboters von einer reinen Ballverfolgung hin zu einer dynamischen und reaktiven Verteidigung. Geplant sind die Implementierung eines Torwart-Modus (Winkelverkürzung, Abdecken der Torlinie) sowie ein Interceptor-Verhalten zum Abfangen freier Bälle.
* **Erneute Evaluierung & Großes Benchmarking:** Durchführung umfangreicher Trainingsläufe auf dem Cluster gegen die verbesserte, schwere Heuristik. Ziel ist es, die Performance-Unterschiede zwischen SAC und CrossQ sowie den Einfluss der Action Spaces (Low-Level Control vs. vordefinierte Skills) unter maximalem taktischen Druck zu quantifizieren.
* **Integration von Self-Play:** Sobald der Agent fähig ist, die dynamische Heuristik konsistent zu schlagen, soll das Training auf kompetitives Self-Play umgestellt werden. Dadurch wird das Netz gezwungen, tiefgreifende Mikrotaktiken ohne vorgegebene Muster zu erlernen.
