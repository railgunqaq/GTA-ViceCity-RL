import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analyze_runs import split_runs, summarize


class AnalysisTests(unittest.TestCase):
    def row(self, ep, dist='3', **kw):
        return dict(episode=str(ep), min_dist=dist, steps='62', **{'return': '-2'}, **kw)

    def test_restart_and_run_id_split(self):
        rows = [self.row(1), self.row(2), self.row(1),
                self.row(2, run_id='new'), self.row(3, run_id='new')]
        self.assertEqual([len(r) for r in split_runs(rows)], [2, 1, 2])

    def test_invalid_distance_cannot_count_as_proximity_or_success(self):
        stats = summarize([self.row(1, '-1'), self.row(2, 'nan'), self.row(3, '1')])
        self.assertEqual(stats['valid_distances'], 1)
        self.assertEqual(stats['mean_min_distance'], 1)
        self.assertIsNone(stats['success_rate'])

    def test_explicit_success_only(self):
        stats = summarize([self.row(1, '1', success='0'), self.row(2, '80', success='1')])
        self.assertEqual(stats['success_rate'], 0.5)


if __name__ == '__main__':
    unittest.main()
