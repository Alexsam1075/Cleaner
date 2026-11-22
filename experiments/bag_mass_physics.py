"""Train a synthetic thin-film mass model; RGB inference requires a separate estimator.

Mass labels come from film area, thickness and density, not the enclosing volume.
This experiment does not change production predictions.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn


def generate(seed, n, area_range, thickness_range, density_range):
    rng = np.random.default_rng(seed)
    area = np.exp(rng.uniform(np.log(area_range[0]), np.log(area_range[1]), n))
    thickness = np.exp(rng.uniform(np.log(thickness_range[0]), np.log(thickness_range[1]), n))
    density = rng.uniform(*density_range, n)
    # Two layers plus handles and gussets, parameterised as extra film area.
    extra = rng.uniform(0.02, 0.25, n)
    film_area = 2 * area * (1 + extra)
    mass = film_area * thickness * density  # m² * µm = cm³
    features = np.stack([np.log(area), np.log(thickness), np.log(density), extra], 1).astype('float32')
    return features, np.log(mass).astype('float32'), mass


def main():
    p = argparse.ArgumentParser(); p.add_argument('--output', required=True)
    p.add_argument('--device', default='cuda:0'); p.add_argument('--epochs', type=int, default=80)
    a = p.parse_args(); torch.manual_seed(71)
    out = Path(a.output); out.mkdir(parents=True, exist_ok=True)
    train = generate(71, 100000, (.015, .8), (6, 100), (.89, 1.05))
    val = generate(83, 12000, (.015, .8), (6, 100), (.89, 1.05))
    shift = generate(97, 12000, (.01, 1.0), (4, 130), (.86, 1.10))
    mean = train[0].mean(0); std = train[0].std(0).clip(1e-6)
    device = torch.device(a.device)
    model = nn.Sequential(nn.Linear(4, 64), nn.GELU(), nn.Linear(64, 64), nn.GELU(), nn.Linear(64, 1)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.002)
    x = torch.from_numpy((train[0]-mean)/std).to(device); y = torch.from_numpy(train[1]).to(device)
    history=[]
    for epoch in range(a.epochs):
        model.train(); order=torch.randperm(len(x), device=device); total=0.
        for inds in order.split(2048):
            pred=model(x[inds]).squeeze(1); loss=nn.functional.mse_loss(pred,y[inds]); optimizer.zero_grad(); loss.backward(); optimizer.step(); total+=float(loss.detach())*len(inds)
        history.append({'epoch':epoch+1,'train_log_mse':total/len(x)})
        if (epoch+1)%10==0: print(json.dumps(history[-1]),flush=True)
    model.eval(); metrics={}
    with torch.no_grad():
        for name, data in [('synthetic_heldout',val),('synthetic_parameter_shift',shift)]:
            pred=model(torch.from_numpy((data[0]-mean)/std).to(device)).squeeze(1).cpu().numpy()
            estimates=np.exp(pred); truth=data[2]
            metrics[name]={'n':len(truth),'mae_g':float(np.mean(np.abs(estimates-truth))),
                           'median_absolute_relative_error':float(np.median(np.abs(estimates-truth)/truth)),
                           'mass_range_g':[float(truth.min()),float(truth.max())]}
    torch.save({'state_dict':model.state_dict(),'mean':mean,'std':std,'features':['log_panel_area_m2','log_thickness_um','log_density_g_cm3','extra_area_fraction'],'synthetic_only':True},out/'bag_mass_physics.pt')
    report={'training_samples':len(x),'epochs':a.epochs,'device':str(device),'metrics':metrics,'history':history,
            'limitations':['Synthetic physics parameters are inputs, not inferred from RGB.','No real-object validation or calibrated prediction for r_0073.','Generated labels exclude contents and other attachments.']}
    (out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(metrics,indent=2),flush=True)


if __name__=='__main__': main()
