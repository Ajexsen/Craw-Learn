# Crawlearn

## Aufgabe

Implementiere einen Reinforcement Learning Algorithmus und löse damit eine kontinuierliche ML-Agents Domäne.

## Ausführung

Wir haben die Proximal Policy Optimization (PPO) implementiert und damit den Crawler gelöst.
Dafür haben wir verschiedene Tools genutzt:
- UnityToGymWrapper
- Pytorch
- Optuna
- Tensorboard

## Starte einen Trainingslauf

Um den Lauf zu starten, installiere alle Anforderungen, die in `requirements.txt` enthalten sind.

Wir haben zwei leicht variierte PPOs, eine Version mit Mininbatch und eine ohne Minibatch (Default Version).

### gutes Parametersetting für Version mit Minibatch
* lr = 3e-4
* tau = 0.95
* clip = 0.2
* hidden_units = 512
* minibatch_size = 32
* update_episodes = 15
* ppo_epochs = 8
* beta = 0.05
* gamma = 0.99

### gutes Parametersetting für Version ohne Minibatch
* lr = 3e-4
* tau = 0.95
* clip = 0.2
* hidden_units = 512
* update_episodes = 15
* ppo_epochs = 5
* beta = 0.05
* gamma = 0.99
