# Federer-Jr.

Crawler
UnityToGymWrapper
Pytorch
Optuna
Tensorboard

## Fragen
* Warum funktioniert unser Algorithmus bei Windows und nicht bei Linux?
* Haben bei Windows Tanh() nicht gebraucht, warum?
* Wie lange sollte es bis zu einem guten Ergebnis dauern?

## gutes Parametersetting
* alpha = 3e-4
* tau = 0.95
* clip = 0.2
* hidden_units = 512
* minibatch_size = 32
* update_time_steps = 12500
* ppo_epochs = 16
* beta = 0.05
* gamma = 0.99

#### Parameter

* update_time_steps 2048 - open end
* epochs 3-32
* beta 0.001 - 0.1
* gamma 0.9 - 0.99
* layers 2 - 8
* (clip 0.1 - 0.3)
