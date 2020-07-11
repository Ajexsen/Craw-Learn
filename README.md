# Federer-Jr. / Crawlearn

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

### gutes Parametersetting (für main.py)
* lr = 3e-4
* tau = 0.95
* clip = 0.2
* hidden_units = 512
* minibatch_size = 32
* update_time_steps = 12500
* ppo_epochs = 16
* beta = 0.05
* gamma = 0.99
