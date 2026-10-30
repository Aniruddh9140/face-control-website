"""Replay held-out public samples through the controller with fake mouse output."""
from pathlib import Path
import sys
from unittest.mock import Mock
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import control_engine as ce
from public_pose import PoseRegressor, PublicPointerModel

with np.load(ROOT/'data/public-training/pose_features.npz',allow_pickle=False) as data:
    candidates=np.flatnonzero((data['subjects']=='F04') & (np.max(np.abs(data['y']),axis=1)<10))
    neutral=data['x'][candidates[0]]
    turning=np.flatnonzero((data['subjects']=='F04') & (data['y'][:,0]>15) & (np.abs(data['y'][:,1])<20))
    turn=data['x'][turning[0]]

c=ce.Controller((1920,1080))
c.mouse=Mock()
c.model=PublicPointerModel(PoseRegressor.load(),c.screen)
c.state.update(face=True,running=True,pointer_mode='public')
c.request_enable()
for i in range(100): c._track((neutral,1.,.02,.3,.3),i/30)
assert c.model.ready and c.state.read().enabled
before=np.array([c.state.read().x,c.state.read().y])
c.mouse.reset_mock()
for i in range(100,160): c._track((turn,1.,.02,.3,.3),i/30)
after=np.array([c.state.read().x,c.state.read().y])
assert c.mouse.move.called and np.linalg.norm(after-before)>100
c.mouse.click.assert_not_called()
c.pause()
c.mouse.reset_mock()
for i in range(160,180): c._track((neutral,1.,.02,.3,.3),i/30)
c.mouse.move.assert_not_called()
print(f'PASS: held-out public replay centered, enabled, moved {np.linalg.norm(after-before):.0f} simulated pixels and paused; no hardware/input used')
