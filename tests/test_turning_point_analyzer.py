from turning_points import TurningPointAnalyzer

def test_get_results():
    trajectories = [[(0, 0), (1, 1), (2, 2), (3, 3), (3, 4), (3, 5)], [(0, 0), (1, 1), (3, 3), (4, 3), (5, 3)]]
    analyzer = TurningPointAnalyzer(trajectories, auto_plot=False)
    results = analyzer.get_results()
    assert len(results) == 4
