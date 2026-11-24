"""Generación, ajuste y evaluación reproducible del controlador neuronal en Colab."""

from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
import json
import random

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from controller import FEATURES, GreenTimeMLP, TimeController, traffic_features
from sumo_experiment import BACKLOG_WEIGHT, GREEN_GRID, HORIZON, WARMUP, IntersectionSimulation, make_network, make_routes
from traffic import ALL_RED, MAX_GREEN, MIN_GREEN, YELLOW, demonstration_greens


def write_report(path, content):
    partial = path.with_suffix(path.suffix + '.part')
    partial.write_text(json.dumps(content, ensure_ascii=False, indent=2), encoding='utf-8')
    partial.replace(path)


def generate_sumo_data(work: Path, output: Path, scenarios=192, seed=42):
    """Simula cada combinación y recupera escenarios ya calculados tras una desconexión."""
    import libsumo

    output.mkdir(parents=True, exist_ok=True)
    cache = output / 'escenarios'
    cache.mkdir(exist_ok=True)
    simulation = IntersectionSimulation(make_network(work))
    spec = {'semilla': seed, 'escenarios': scenarios, 'calentamiento_s': WARMUP,
            'horizonte_s': HORIZON, 'penalizacion_pendientes': BACKLOG_WEIGHT, 'version_sumo': libsumo.getVersion()[1],
            'rejilla': [list(pair) for pair in GREEN_GRID], 'amarillo_s': YELLOW, 'todo_rojo_s': ALL_RED}
    manifest = output / 'protocolo.json'
    if manifest.exists() and json.loads(manifest.read_text(encoding='utf-8')) != spec:
        raise ValueError('Para cambiar el protocolo, usa otra carpeta de resultados.')
    write_report(manifest, spec)
    labels, trials = [], []
    for index in range(scenarios):
        scenario = seed * 100_000 + index
        cached = cache / f'{scenario}.json'
        if cached.exists():
            record = json.loads(cached.read_text(encoding='utf-8'))
        else:
            routes, scheduled = make_routes(work, scenario)
            plans = []
            for a, b in GREEN_GRID:
                result = simulation.run(routes, scheduled, scenario, (a, b))
                if result.collision_events:
                    raise RuntimeError(f'SUMO registró una colisión en el escenario {scenario}.')
                plans.append({'escenario': scenario, 'verde_a': a, 'verde_b': b,
                              **asdict(result), 'objetivo': result.objective})
            if len({plan['counts'] for plan in plans}) != 1:
                raise RuntimeError('Las pruebas de un escenario no empezaron con el mismo estado.')
            # El desempate prefiere menor espera y, después, un ciclo más corto.
            best = min(plans, key=lambda p: (p['objetivo'], p['queue_seconds'],
                                             p['verde_a'] + p['verde_b'], p['verde_a']))
            features = traffic_features(*best['counts'])
            label = {'escenario': scenario, **dict(zip(FEATURES, features)), 'verde_a': best['verde_a'],
                     'verde_b': best['verde_b'], 'objetivo_rejilla': best['objetivo']}
            record = {'etiqueta': label, 'planes': plans}
            write_report(cached, record)
        labels.append(record['etiqueta'])
        trials.extend(record['planes'])
        if (index + 1) % 12 == 0 or index + 1 == scenarios:
            print(f'SUMO · {index + 1}/{scenarios} escenarios', flush=True)
    dataset = pd.DataFrame(labels)
    dataset.to_csv(output / 'datos_mlp.csv', index=False)
    pd.DataFrame(trials).to_csv(output / 'candidatos_sumo.csv', index=False)
    return dataset


def train_mlp(dataset: pd.DataFrame, output: Path, weights: Path, seed=42, epochs=120):
    """Entrena, selecciona por pérdida de validación y mide sobre escenarios de prueba."""
    from sklearn.model_selection import train_test_split
    import matplotlib.pyplot as plt

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(2)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    # El estado sin vehículos activa el todo rojo y queda fuera del ajuste y evaluación MLP.
    eligible = dataset[dataset['demanda_total'] > 0].copy()
    train, rest = train_test_split(eligible, test_size=0.30, random_state=seed)
    valid, test = train_test_split(rest, test_size=0.50, random_state=seed)
    membership = pd.concat([table[['escenario']].assign(particion=name)
                            for name, table in [('train', train), ('valid', valid), ('test', test)]])
    membership.to_csv(output / 'particiones.csv', index=False)

    def tensors(table):
        return (torch.tensor(table[list(FEATURES)].to_numpy(), dtype=torch.float32),
                torch.tensor(table[['verde_a', 'verde_b']].to_numpy(), dtype=torch.float32))

    x_train, y_train = tensors(train)
    x_valid, y_valid = tensors(valid)
    mean, scale = x_train.mean(dim=0), x_train.std(dim=0, unbiased=False).clamp_min(1e-6)
    train_batches = DataLoader(TensorDataset((x_train - mean) / scale, y_train), batch_size=32, shuffle=True,
                               generator=torch.Generator().manual_seed(seed))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GreenTimeMLP().to(device)
    loss_fn = nn.SmoothL1Loss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=0.001)
    x_valid, y_valid = ((x_valid - mean) / scale).to(device), y_valid.to(device)
    history, best_loss = [], float('inf')
    best_state = None
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for x, y in train_batches:
            optimizer.zero_grad()
            loss = loss_fn(model(x.to(device)), y.to(device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(x)
        model.eval()
        with torch.inference_mode():
            val_loss = loss_fn(model(x_valid), y_valid).item()
        history.append({'epoca': epoch, 'entrenamiento': total_loss / len(train), 'validacion': val_loss})
        if val_loss < best_loss:
            best_loss, best_epoch = val_loss, epoch
            best_state = deepcopy(model.state_dict())
    if best_state is None:
        raise RuntimeError('El MLP no produjo un checkpoint con pérdida finita.')

    weights.parent.mkdir(parents=True, exist_ok=True)
    pending = weights.with_suffix('.pt.part')
    torch.save({'format_version': 1, 'features': list(FEATURES), 'green_limits': [MIN_GREEN, MAX_GREEN],
                'state_dict': {key: value.cpu() for key, value in best_state.items()}, 'mean': mean, 'scale': scale,
                'count_limits': [int(train['vehiculos_a'].max()), int(train['vehiculos_b'].max())],
                'best_epoch': best_epoch, 'seed': seed}, pending)
    pending.replace(weights)
    history = pd.DataFrame(history)
    history.to_csv(output / 'aprendizaje.csv', index=False)
    ax = history.plot(x='epoca', y=['entrenamiento', 'validacion'], ylabel='Pérdida Smooth L1', xlabel='Época')
    ax.figure.savefig(output / 'curva_mlp.png', dpi=140, bbox_inches='tight')
    plt.close(ax.figure)

    predictor = TimeController(weights)
    predictions = [predictor.predict(row.vehiculos_a, row.vehiculos_b, row.pesados_a, row.pesados_b)
                   for row in test.itertuples()]
    test = test.assign(predicho_a=[row[0] for row in predictions], predicho_b=[row[1] for row in predictions])
    test.to_csv(output / 'predicciones_prueba.csv', index=False)
    error = np.array(predictions) - test[['verde_a', 'verde_b']].to_numpy()
    metrics = {'entrenamiento': len(train), 'validacion': len(valid), 'prueba': len(test),
               'epocas': epochs, 'mejor_epoca': best_epoch, 'dispositivo': str(device),
               'mae_a_s': float(np.abs(error[:, 0]).mean()), 'mae_b_s': float(np.abs(error[:, 1]).mean()),
               'rmse_a_s': float(np.sqrt((error[:, 0] ** 2).mean())),
               'rmse_b_s': float(np.sqrt((error[:, 1] ** 2).mean()))}
    write_report(output / 'metricas_mlp.json', metrics)
    print('Métricas del MLP:', json.dumps(metrics, ensure_ascii=False), flush=True)
    return test


def evaluate_sumo(test: pd.DataFrame, work: Path, output: Path):
    """Re-simula cada predicción y sus referencias con las llegadas de la prueba reservada."""
    sim = IntersectionSimulation(make_network(work))
    comparisons = []
    for row in test.itertuples():
        routes, scheduled = make_routes(work, row.escenario)
        plans = {'mlp': (row.predicho_a, row.predicho_b), 'fijo': (22, 23),
                 'proporcional': demonstration_greens(row.vehiculos_a, row.vehiculos_b),
                 'mejor_rejilla': (row.verde_a, row.verde_b)}
        for method, greens in plans.items():
            result = sim.run(routes, scheduled, row.escenario, greens)
            comparisons.append({'escenario': row.escenario, 'metodo': method, 'verde_a': greens[0],
                                'verde_b': greens[1], **asdict(result), 'objetivo': result.objective})
    comparisons = pd.DataFrame(comparisons)
    comparisons.to_csv(output / 'comparacion_prueba_sumo.csv', index=False)
    summary = comparisons.groupby('metodo')[['objetivo', 'queue_seconds', 'arrived', 'unfinished', 'collision_events']].mean()
    summary.to_csv(output / 'resumen_prueba_sumo.csv')
    print(summary.to_string(), flush=True)
    return summary
