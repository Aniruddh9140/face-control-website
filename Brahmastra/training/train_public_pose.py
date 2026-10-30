"""Reproducible person-separated BIWI training with a final untouched test set."""
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cv2
import numpy as np
import face_engine as fe
from public_pose import PoseRegressor, pose_features, MODEL_PATH

DATA = ROOT / "data" / "public-training"
SUBJECTS = dict(zip(range(1,25), ['F01','F02','F03','F04','F05','F06','M01','M02','M03','M04','M05','M06','M07','M08','F03','M09','M10','F05','M11','M12','F02','M01','M13','M14']))
VALIDATION = {'F02','M02','M07'}
TEST = {'F04','M04','M10','M14'}

def extract():
    cache = DATA / 'pose_features.npz'
    if cache.exists():
        with np.load(cache, allow_pickle=False) as d:
            return {k:d[k].copy() for k in d.files}
    features, targets, subjects, sequences, names = [], [], [], [], []
    attempted = missed = outside = 0
    for seq in range(1,25):
        files = sorted((DATA / 'biwi_head_pose' / f'{seq:02d}').glob('*_rgb.jpg'))[::5]
        mesh = fe.FaceMeshBackend()
        try:
            for index, path in enumerate(files):
                attempted += 1
                rotation = np.loadtxt(path.with_name(path.name.replace('_rgb.jpg','_pose.txt')))[:3,:3]
                # BIWI stores rows of the rotation; use its forward vector.
                v = rotation[:,2]
                angles = np.degrees([np.arctan2(v[0], v[2]), np.arctan2(-v[1], np.hypot(v[0],v[2]))])
                if abs(angles[0]) > 50 or abs(angles[1]) > 40:
                    outside += 1
                    continue
                image = cv2.imread(str(path))
                image = cv2.resize(image, (320,240))
                mesh.process(image, ts_ms=index * 200)
                if mesh.last_xyz is None:
                    missed += 1
                    continue
                try:
                    f = pose_features(mesh.last_xyz,320,240)
                except ValueError:
                    missed += 1
                    continue
                features.append(f); targets.append(angles)
                subjects.append(SUBJECTS[seq]); sequences.append(seq); names.append(path.name)
        finally:
            mesh.close()
        print(f'Sequence {seq:02d}/24: {len(features)} usable frames', flush=True)
    data = dict(x=np.asarray(features), y=np.asarray(targets), subjects=np.asarray(subjects),
                sequences=np.asarray(sequences), names=np.asarray(names),
                attempted=np.array(attempted), missed=np.array(missed), outside=np.array(outside))
    np.savez(cache, **data)
    return data

def metrics(predicted, target):
    error = np.abs(predicted-target)
    return dict(yaw_mae_deg=float(error[:,0].mean()), pitch_mae_deg=float(error[:,1].mean()),
                mean_mae_deg=float(error.mean()), p90_vector_error_deg=float(np.quantile(np.linalg.norm(error,axis=1),.9)))

def main():
    data = extract()
    x,y,s = data['x'],data['y'],data['subjects']
    val, test = np.isin(s,list(VALIDATION)),np.isin(s,list(TEST))
    train = ~(val|test)
    if min(train.sum(),val.sum(),test.sum()) < 50:
        raise RuntimeError('Insufficient examples in a split')
    trials=[]
    for quadratic in (False,True):
        for strength in (.1,1.,10.,100.):
            model=PoseRegressor.fit(x[train],y[train],strength,quadratic)
            score=metrics(model.predict(x[val]),y[val])
            trials.append(dict(quadratic=quadratic,strength=strength,**score))
    best=min(trials,key=lambda t:t['mean_mae_deg'])
    model=PoseRegressor.fit(x[~test],y[~test],best['strength'],best['quadratic'])
    score=metrics(model.predict(x[test]),y[test])
    baseline=metrics(np.broadcast_to(y[~test].mean(axis=0),y[test].shape),y[test])
    passed=score['mean_mae_deg'] < 8 and score['p90_vector_error_deg'] < 20
    report=dict(dataset='BIWI fast.ai RGB repack',python=sys.version, stride=5,
                input_resolution=[320,240],domain_degrees=dict(yaw=[-50,50],pitch=[-40,40]),
                attempted=int(data['attempted']), no_face=int(data['missed']),outside_domain=int(data['outside']),
                usable=len(x),split_counts=dict(train=int(train.sum()),validation=int(val.sum()),test=int(test.sum())),
                split_subjects=dict(train=sorted(set(s[train])),validation=sorted(VALIDATION),test=sorted(TEST)),
                trials=trials,selected=best,test=score,constant_baseline=baseline,acceptance_passed=passed,
                note='Person-separated internal benchmark, not screen-pointer accuracy or an official BIWI leaderboard result.')
    candidate=MODEL_PATH if passed else MODEL_PATH.with_name('biwi_head_pose_candidate.npz')
    model.save(candidate)
    report['model_sha256']=hashlib.sha256(candidate.read_bytes()).hexdigest()
    report['model_file']=candidate.name
    (ROOT/'docs'/'PUBLIC_TRAINING_RESULTS.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2),flush=True)
    if not passed:
        raise RuntimeError('Public model did not meet the predeclared acceptance threshold')

if __name__=='__main__':
    main()
