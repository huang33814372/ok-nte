import sys
import unittest

import numpy as np

from src.sound_trigger.capture.base import AudioCaptureSource
from src.sound_trigger.capture.process_resolver import name_set


class _TestCaptureSource(AudioCaptureSource):
    @property
    def name(self):
        return "test"

    def _produce(self, push):
        pass


class AudioCaptureSourceTests(unittest.TestCase):
    def test_read_returns_latest_chunk_and_drops_backlog(self):
        source = _TestCaptureSource(queue_max=4)

        for value in range(6):
            source._push(np.array([value], dtype=np.float32))

        self.assertEqual(source.read(timeout=0.01).tolist(), [5.0])
        self.assertIsNone(source.read(timeout=0.01))


class ProcessResolverTests(unittest.TestCase):
    def test_name_set_normalizes_scalar_and_collection(self):
        self.assertEqual(name_set("HTGame.exe"), {"htgame.exe"})
        self.assertEqual(
            name_set(["HTGame.exe", "NTEGame.exe"]),
            {"htgame.exe", "ntegame.exe"},
        )
        self.assertEqual(name_set(["", None]), set())


class ProcessLoopbackTests(unittest.TestCase):
    @unittest.skipUnless(sys.platform == "win32", "WASAPI loopback is Windows-only")
    def test_to_mono_averages_stereo_float32_samples(self):
        from src.sound_trigger.capture.process_loopback import _to_mono

        stereo = np.array([1.0, -1.0, 0.25, 0.75], dtype=np.float32)

        np.testing.assert_allclose(_to_mono(stereo.tobytes(), is_float=True), [0.0, 0.5])

    @unittest.skipUnless(sys.platform == "win32", "WASAPI loopback is Windows-only")
    def test_to_mono_converts_stereo_pcm16_samples(self):
        from src.sound_trigger.capture.process_loopback import _to_mono

        stereo = np.array([32767, -32768, 16384, 0], dtype=np.int16)

        np.testing.assert_allclose(
            _to_mono(stereo.tobytes(), is_float=False),
            [-1.0 / 65536.0, 0.25],
            atol=1e-6,
        )


class SoundListenerTests(unittest.TestCase):
    def test_missing_sample_file_fails_fast(self):
        from src.sound_trigger.SoundListener import SoundListener

        with self.assertRaises(RuntimeError):
            SoundListener(
                sample_path="assets/sounds/__missing_dodge__.wav",
                counter_attack_sample_path="",
            )


if __name__ == "__main__":
    unittest.main()
