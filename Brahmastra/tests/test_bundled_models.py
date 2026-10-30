"""Native model load checks work both from source and inside the one-file EXE."""
import unittest
import numpy as np
import face_engine
import voice_typing

class BundledModelsTests(unittest.TestCase):
    def test_face_landmarker_loads_and_processes_without_camera(self):
        mesh=face_engine.FaceMeshBackend()
        try:
            mesh.process(np.zeros((240,320,3),np.uint8))
            self.assertIsNone(mesh.last_xyz)
        finally:
            mesh.close()

    def test_speech_detector_loads_and_processes_silence(self):
        vad=voice_typing.SileroVad()
        try:
            probability=vad(np.zeros(vad.CHUNK,np.float32))
            self.assertTrue(np.isfinite(probability))
            self.assertGreaterEqual(probability,0.)
            self.assertLessEqual(probability,1.)
        finally:
            vad.close()

if __name__=='__main__': unittest.main()
