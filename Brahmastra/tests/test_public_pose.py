import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock
import numpy as np

from public_pose import PoseRegressor, PublicPointerModel, pose_features, FEATURE_COUNT
import control_engine as ce

class PublicPoseTests(unittest.TestCase):
    def test_regression_save_load_predicts_unseen_inputs(self):
        rng=np.random.default_rng(20)
        x=rng.normal(size=(200,FEATURE_COUNT))
        y=x[:,:2]*12
        model=PoseRegressor.fit(x,y,.1)
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'model.npz'
            model.save(path)
            restored=PoseRegressor.load(path)
            unseen=rng.normal(size=(30,FEATURE_COUNT))
            np.testing.assert_allclose(restored.predict(unseen),unseen[:,:2]*12,atol=.1)

    def model(self):
        regressor=Mock()
        regressor.predict.side_effect=lambda x: np.asarray(x[:2])
        regressor.signs=np.array([-1.,-1.])
        return PublicPointerModel(regressor,(1920,1080))

    def test_neutral_hold_is_required_then_head_turn_spans_screen(self):
        m=self.model()
        for t in np.arange(0,1.9,1/30):
            self.assertFalse(m.observe(np.zeros(2),t))
        for t in np.arange(1.9,2.2,1/30):
            m.observe(np.zeros(2),t)
        self.assertTrue(m.ready)
        np.testing.assert_allclose(m.predict([0,0]),[959.5,539.5])
        np.testing.assert_allclose(m.predict([-22,-16]),[1919,1079])

    def test_motion_and_face_loss_do_not_complete_neutral_hold(self):
        m=self.model()
        for i,t in enumerate(np.arange(0,4,1/30)):
            self.assertFalse(m.observe(np.array([0 if i%8 else 10,0]),t))
        m.cancel_hold()
        self.assertEqual(m.progress,0)
        self.assertFalse(m.ready)

    def test_extreme_pose_suspends_prediction_and_recenter_is_guarded(self):
        m=self.model()
        self.assertFalse(m.observe([60,0],0))
        with self.assertRaises(ValueError): m.recenter([60,0])
        m.recenter([10,5])
        np.testing.assert_allclose(m.predict([10,5]),[959.5,539.5])

    def test_invalid_landmarks_rejected(self):
        for xyz in (np.zeros((20,3)),np.zeros((478,3)),np.full((478,3),np.nan)):
            with self.assertRaises(ValueError): pose_features(xyz,320,240)

    def test_no_mouse_input_before_public_neutral_is_ready(self):
        c=ce.Controller((1920,1080))
        c.model=self.model()
        c.mouse=Mock()
        c.state.update(enabled=True,pointer_mode='public')
        for t in np.arange(0,1.8,1/30):
            c._track((np.zeros(2),1.,.02,.3,.3),t)
        c.mouse.move.assert_not_called()
        c.mouse.click.assert_not_called()

if __name__=='__main__': unittest.main()
