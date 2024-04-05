# ----------------------------------------------------------------------------------------------------------------------
# Import generic
# ----------------------------------------------------------------------------------------------------------------------
import time

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde, poisson
import math
import os

# ----------------------------------------------------------------------------------------------------------------------
# GLOBAL
# ----------------------------------------------------------------------------------------------------------------------
FORMATS = ("eps", "svg", "png", "jpg")

def get_turning_events(trajectories, do_show=True, do_block=False, name="", path_save_plots="."):
    """
    Get turning events from trajectories and visualize them.

    Parameters:
        trajectories (list or list of lists): List of trajectories or list of lists of trajectories.
        do_show (bool): Whether to display the plots (default: True).
        do_block (bool): Whether to block code execution until plots are closed (default: False).
        name (str): Name for saving plots (default: "").
        path_save_plots (str): Path to save plots (default: ".").

    Returns:
        tuple: Turning points, transition directions, original index , and probabilities.
    """
    if all(isinstance(traj, (list, np.ndarray)) and all(isinstance(sub_traj, (list, np.ndarray)) for sub_traj in traj) for traj in trajectories):
        trajectories = trajectories
    else:
        trajectories = [trajectories]

    turning_points = TurningPointAnalyzer(trajectories, tolerance=2)

    fig, ax = turning_points.plot_transition(do_block=False)
    if do_show:
        plt.show(block=do_block)
    save_figure(f"turning_transition_probability_{name}", figure=fig, path_save_plots=path_save_plots)

    fig, ax = turning_points.plot_frequency(do_block=False, do_fit=False)
    if do_show:
        plt.show(block=do_block)
    save_figure(f"turning_frequency_{name}", figure=fig, path_save_plots=path_save_plots)

    return turning_points.get_results()


def save_figure(name, path_save_plots=None, figure=None, dpi=360):
    """Save a figure to the various formats specified"""
    for saving_format in FORMATS:
        if path_save_plots:
            target_path = os.path.join(path_save_plots, f"{name}.{saving_format}")
        else:
            target_path = f"{name}.{saving_format}"
        if not figure:
            plt.savefig(target_path, format=saving_format, dpi=dpi)
        else:
            figure.savefig(target_path, format=saving_format, dpi=dpi)

def pldist(point, start, end):
    """
    Calculates the distance from ``point`` to the line given
    by the points ``start`` and ``end``.

    Parameters:
        point (numpy.ndarray): A point.
        start (numpy.ndarray): A point of the line.
        end (numpy.ndarray): Another point of the line.

    Returns:
        float: The distance from the point to the line.
    """
    if np.all(np.equal(start, end)):
        return np.linalg.norm(point - start)

    return np.divide(
        np.abs(np.linalg.norm(np.cross(end - start, start - point))),
        np.linalg.norm(end - start))


def _rdp_iter(M, start_index, last_index, epsilon, dist=pldist):
    """
    Iterative version of the Ramer-Douglas-Peucker algorithm.

    Parameters:
        M (numpy.ndarray): An array of points.
        start_index (int): Index of the start point.
        last_index (int): Index of the last point.
        epsilon (float): Epsilon in the RDP algorithm.
        dist (function): Distance function (default: pldist).

    Returns:
        numpy.ndarray: Boolean mask indicating points to keep.
    """
    stk = []
    stk.append([start_index, last_index])
    global_start_index = start_index
    indices = np.ones(last_index - start_index + 1, dtype=bool)

    while stk:
        start_index, last_index = stk.pop()

        dmax = 0.0
        index = start_index

        for i in range(index + 1, last_index):
            if indices[i - global_start_index]:
                d = dist(M[i], M[start_index], M[last_index])
                if d > dmax:
                    index = i
                    dmax = d

        if dmax > epsilon:
            stk.append([start_index, index])
            stk.append([index, last_index])
        else:
            for i in range(start_index + 1, last_index):
                indices[i - global_start_index] = False

    return indices


def rdp_iter(M, epsilon, dist=pldist, return_mask=False):
    """
    Simplifies a given array of points using the Ramer-Douglas-Peucker algorithm.

    Iterative version.

    Parameters:
        M (numpy.ndarray): An array of points.
        epsilon (float): Epsilon in the RDP algorithm.
        dist (function): Distance function (default: pldist).
        return_mask (bool): Whether to return the mask of points to keep instead (default: False).

    Returns:
        Union[numpy.ndarray, Tuple[numpy.ndarray, numpy.ndarray]]: Simplified array of points or tuple of simplified array and mask.
    """
    mask = _rdp_iter(M, 0, len(M) - 1, epsilon, dist)

    if return_mask:
        return mask

    return M[mask], mask


def angle(dir):
    """
    Returns the turn angles between vectors.

    Parameters:
        dir (numpy.ndarray): A 2D-array of shape (N,M) representing N vectors in M-dimensional space.

    Returns:
        numpy.ndarray: A 1D-array of values of shape (N-1,), with each value between -pi and pi.
            Negative angle implies right turn, positive angle implies left turn.
    """
    dir2 = dir[1:]
    dir1 = dir[:-1]

    # Compute the cross product magnitude (determinant of 2x2 matrix)
    cross_product = dir1[:, 0] * dir2[:, 1] - dir1[:, 1] * dir2[:, 0]

    # Compute the dot product
    dot_product = np.sum(dir1 * dir2, axis=1)

    # Compute the angle
    angles = np.arctan2(cross_product, dot_product)
    return angles


def angle_from_trajectory(trajectory):
    """
    Compute turn angles from trajectory points.

    Parameters:
        trajectory (numpy.ndarray): Trajectory points.

    Returns:
        numpy.ndarray: Array of turn angles.
    """
    directions = np.diff(trajectory, axis=0)
    return angle(directions)


class TurningPointAnalyzer:
    """
    Analyzes turning points in trajectories and calculates transition probabilities.
    """

    def __init__(self, trajectories, tolerance=5, min_angle=np.pi*0.05, auto_compute=True, auto_plot=True, target_folder=None, do_plot_individual_track=False):
        """
        Initialize the TurningPointAnalyzer.

        Parameters:
            trajectories (list or list of lists): List of trajectories or list of lists of trajectories.
            tolerance (float): Tolerance parameter for simplifying trajectories (default: 5).
            min_angle (float): Minimum change in direction to be considered a turning point (default: π*0.05).
            auto_compute (bool): Whether to automatically compute turning points, transition directions, and probabilities (default: True).
            auto_plot (bool): Whether to automatically plot the results (default: True).
            target_folder (str): Path to the folder to save plots (default: None).
        """
        self.target_folder = target_folder

        # Ensure trajectories is a list of lists
        if all(isinstance(traj, (list, np.ndarray, tuple)) and all(isinstance(sub_traj, (list, np.ndarray, tuple)) for sub_traj in traj) for traj in trajectories):
            self.trajectories = trajectories
        else:
            self.trajectories = [trajectories]

        self.proba_same = None

        self.pos_idx = []
        self.neg_idx = []

        self.frequency = []

        self.theta = []
        self.counts = []
        self.lists = []
        self.ori_idx = []

        self.count_dico = {"pos_neg": 0, "neg_pos": 0, "pos_pos": 0, "neg_neg": 0}
        self.lists = []

        self.tolerance = tolerance
        self.min_angle = min_angle

        # Automatically compute turning points, transition directions, and probabilities if auto_compute is True
        if auto_compute:
            self.turning_points(do_plot_individual_track=do_plot_individual_track)
            self.transition_dir()
            self.calculate_probability()

        if auto_plot:
            self.plot_transition(do_block=False)
            self.plot_frequency(do_block=True)


    def turning_points(self, do_plot_individual_track=False, **kwargs):
        """
        Compute turning points for each trajectory.

        Parameters:
            do_plot_turning (bool): Whether to plot turning points (default: False).
        """
        for traj in self.trajectories:
            (pos_idx, neg_idx), theta, frequency, ori_idx = compute_turning_points(traj, tolerance=self.tolerance, min_angle=self.min_angle, do_plot_individual_track=do_plot_individual_track)
            self.pos_idx.append(pos_idx)
            self.neg_idx.append(neg_idx)
            self.theta.append(theta)
            self.frequency.append(frequency)
            self.ori_idx.append(ori_idx)
        return (self.pos_idx, self.neg_idx), self.theta, self.ori_idx

    def transition_dir(self):
        """
        Compute transition directions for each trajectory.
        """
        for pos_idx, neg_idx, theta in zip(self.pos_idx, self.neg_idx, self.theta):
            counts, lists = compute_transition_dir(pos_idx, neg_idx, theta)
            for key, items in counts.items():
                self.count_dico[key] += items
            self.lists.append(lists)
        return self.counts, self.lists

    def calculate_probability(self):
        """
        Calculate the probability of same turn direction.
        """
        try:
            self.proba_same = (self.count_dico['neg_neg'] + self.count_dico['pos_pos']) / \
                              (self.count_dico['neg_pos'] + self.count_dico['pos_neg'] + self.count_dico['neg_neg'] + self.count_dico['pos_pos'])
        except ZeroDivisionError:
            self.proba_same = 0

    def get_proba_same_turn(self):
        """
        Get the probability of same turn direction.

        Returns:
            float: Probability of same turn direction.
        """
        if self.proba_same:
            return self.proba_same
        else:
            raise RuntimeWarning("[!] Same turn probability was not computed")

    def get_results(self):
        """
        Get the results of the analysis.

        Returns:
            tuple: Results of the analysis, including turning points, angles, original indices, and probability of same turn direction.
        """
        return (self.pos_idx, self.neg_idx), self.theta, self.ori_idx, self.proba_same


    def plot_transition(self, figsize=(6, 6), do_show=True, do_block=False):
        """
        Plot transition directions.

        Parameters:
            figsize (tuple): Figure size (default: (6, 6)).
            do_show (bool): Whether to show the plot (default: True).
            do_block (bool): Whether to block code execution until the plot window is closed (default: False).

        Returns:
            tuple: Figure and axis objects.
        """
        fig, ax = plt.subplots(figsize=figsize)

        merged_list = []
        for sublist in self.lists:
            for tup in sublist:
                merged_list.extend(tup)
        x_coords, y_coords = zip(*merged_list)
        plt.scatter(x_coords, y_coords, color="darkmagenta", s=12)
        ax.set_xticks(np.arange(-np.pi, np.pi + 0.1, np.pi / 2))
        ax.set_xticklabels([r'$-\pi$', r'$-\frac{\pi}{2}$', r'$0$', r'$\frac{\pi}{2}$', r'$\pi$'], fontsize=24)
        ax.set_yticks(np.arange(-np.pi, np.pi + 0.1, np.pi / 2))
        ax.set_yticklabels([r'$-\pi$', r'$-\frac{\pi}{2}$', r'$0$', r'$\frac{\pi}{2}$', r'$\pi$'], fontsize=24)
        ax.set_xlabel('Rotation angle (t)', fontsize=24)
        ax.set_ylabel('Rotation angle (t+1)', fontsize=24)
        ax.set_xlim(-np.pi, np.pi)
        ax.set_ylim(-np.pi, np.pi)

        ax.axhline(0, color='k', linestyle='--')
        ax.axvline(0, color='k', linestyle='--')

        pad_factor = 0.2
        bbox_props = dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5')
        ax.text(-np.pi + pad_factor, -np.pi + pad_factor, f"Right-Right: {self.count_dico['neg_neg']}", fontsize=18, ha='left', va='bottom', bbox=bbox_props)
        ax.text(-np.pi + pad_factor, np.pi - pad_factor, f"Right-Left: {self.count_dico['neg_pos']}", fontsize=18, ha='left', va='top', bbox=bbox_props)
        ax.text(np.pi - pad_factor, -np.pi + pad_factor, f"Left-Right: {self.count_dico['pos_neg']}", fontsize=18, ha='right', va='bottom', bbox=bbox_props)
        ax.text(np.pi - pad_factor, np.pi - pad_factor, f"Left-Left: {self.count_dico['pos_pos']}", fontsize=18, ha='right', va='top', bbox=bbox_props)

        p = self.get_proba_same_turn()
        ax.text(0, 0, f"p={round(p, 2)}", fontsize=16, ha='center', va='center', bbox=bbox_props)

        plt.tight_layout()

        if self.target_folder:
            plt.savefig(self.target_folder, format="png", dpi=360)

        if do_show:
            plt.show(block=do_block)

        return fig, ax

    def plot_frequency(self, figsize=(6, 6), bw_method=0.25, num_values_plot=3000, do_show=True, do_block=False, do_fit=False, remove_outliers=True, outliers_method="iqr", **kwargs):
        """
        Plot frequency distribution of time between turns.

        Parameters:
            figsize (tuple): Figure size (default: (6, 6)).
            bw_method (float): Bandwidth parameter for kernel density estimation (default: 0.25).
            num_values_plot (int): Number of values to plot in KDE (default: 3000).
            do_show (bool): Whether to show the plot (default: True).
            do_block (bool): Whether to block code execution until the plot window is closed (default: False).
            do_fit (bool): Whether to fit a Poisson distribution to the data (default: False).
            remove_outliers (bool): Whether to remove outliers from the data (default: True).
            outliers_method (str): Method for outlier removal ("iqr" or "zscore", default: "iqr").

        Returns:
            tuple: Figure and axis objects.
        """
        fig, ax = plt.subplots(figsize=figsize)

        merged_list = []
        for sublist in self.frequency:
            if remove_outliers:
                if outliers_method == "iqr":
                    sublist = remove_outliers_iqr(sublist, **kwargs)
                elif outliers_method == "zscore":
                    sublist = remove_outliers_zscore(sublist, **kwargs)
            merged_list.extend(sublist)
        ax.hist(merged_list, bins=np.arange(min(merged_list), max(merged_list) + 1), density=True, facecolor="#0870d1", edgecolor='#000000', linewidth=0.5, label='Measured')

        if do_fit:
            lambda_estimate = np.mean(merged_list)
            x = np.arange(int(min(merged_list)), max(merged_list) + 1)
            pmf = poisson.pmf(x, lambda_estimate)
            ax.bar(x, pmf, width=1, align='edge', color="orange", alpha=0.85, label='Poisson Distribution (λ={})'.format(lambda_estimate))

            kde = gaussian_kde(merged_list, bw_method)
            x = np.linspace(min(merged_list), max(merged_list), num_values_plot)
            kde_values = kde(x)
            ax.plot(x, kde_values, color='crimson', label='Estimated KDE', linewidth=5, alpha=0.75)

        ax.set_xlabel('Time between turns (frames)', fontsize=24)
        ax.set_xticks(np.arange(min(merged_list), max(merged_list) + 1, 2))
        ax.set_xticks(np.arange(0, 2 + 1, 0.5))
        ax.set_ylabel('PDF', fontsize=24)

        plt.tight_layout()

        if self.target_folder:
            plt.savefig(self.target_folder, format="png", dpi=360)

        if do_show:
            plt.show(block=do_block)

        return fig, ax


def remove_outliers_iqr(data, k_iqr=1.5, do_print=False, **kwargs):
    """
    Remove outliers from a list using IQR (Interquartile Range).

    Parameters:
        data (list): The input list.
        k_iqr (float): The scale factor to determine the outlier threshold. Typically, 1.5 is used.
        do_print (bool): Whether to print the removed outliers.

    Returns:
        list: The list with outliers removed.
    """
    # Calculate quartiles and IQR
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1

    # Determine lower and upper bounds
    lower_bound = q1 - k_iqr * iqr
    upper_bound = q3 + k_iqr * iqr

    # Remove outliers
    filtered_data = [x for x in data if lower_bound <= x <= upper_bound]

    # Print removed outliers if required
    if do_print:
        outliers = [x for x in data if lower_bound >= x >= upper_bound]
        print(f"Removed {len(outliers)} outliers: {outliers}")

    return filtered_data


def remove_outliers_zscore(data, threshold=5, do_print=False, **kwargs):
    """
    Remove outliers from a list using Z-score.

    Parameters:
        data (list): The input list.
        threshold (float): The threshold for Z-score. Data points with a Z-score greater than this threshold are considered outliers.
        do_print (bool): Whether to print the removed outliers.

    Returns:
        list: The list with outliers removed.
    """
    # Calculate Z-scores
    z_scores = np.abs((data - np.mean(data)) / np.std(data))

    # Filter data based on Z-scores
    filtered_data = data[(z_scores < threshold)]

    # Print removed outliers if required
    if do_print:
        outliers = data[(z_scores >= threshold)]
        print(f"Removed {len(outliers)} outliers: {outliers}")

    return filtered_data

def compute_turning_points(pos_lst, tolerance=2, min_angle=np.pi * 0.05, do_plot_individual_track=False,
                           do_plot_cumangle=False):
    """
    Compute turning points along a trajectory.

    Parameters:
        pos_lst (list): List of 2D positions [(x1, y1), (x2, y2), ...].
        tolerance (float): Tolerance parameter for Ramer-Douglas-Peucker algorithm.
        min_angle (float): Minimum change in direction to be considered as a turning point.
        do_plot_individual_track (bool): Whether to plot individual tracks or not.
        do_plot_cumangle (bool): Whether to plot cumulative angles or not.

    Returns:
        tuple: Tuple containing:
            - Tuple of positive and negative turning point indices.
            - Array of direction angles.
            - Array of frequencies of points between consecutive turning points.
    """
    pos_lst = np.array(pos_lst)
    x, y = pos_lst.T

    # Simplify the trajectory using Ramer-Douglas-Peucker algorithm
    simplified, mask = rdp_iter(pos_lst, tolerance)
    sx, sy = np.array(simplified).T

    # Compute direction vectors on the simplified curve
    theta = angle_from_trajectory(simplified)

    # Select the indices of points with significant change in direction
    pos_idx = np.where(theta > min_angle)[0] + 1
    neg_idx = np.where(theta < min_angle)[0] + 1

    ori_idx = np.where(mask)[0]

    frequency = np.diff(ori_idx)

    if do_plot_individual_track:
        # Plot individual tracks
        plt.figure(figsize=(8, 9))
        plt.plot(x, y, "olivedrab", label='original path', linewidth=3)
        plt.plot(sx, sy, color='crimson', linestyle='--', label='simplified path')
        plt.scatter(sx[pos_idx], sy[pos_idx], color="coral", edgecolor="black", s=16,
                    label='clockwise turning points', zorder=4)
        plt.scatter(sx[neg_idx], sy[neg_idx], color="blue", edgecolor="black", s=16,
                    label='counter clockwise turning points', zorder=4)
        plt.gca().invert_yaxis()
        plt.legend(loc='best')
        plt.gca().set_aspect('auto')
        plt.show()

    if do_plot_cumangle:
        plt.figure(figsize=(6, 9))
        # Plot cumulative angle
        cum_angle = np.cumsum(theta)
        plt.plot(cum_angle)
        plt.show()

    return (pos_idx, neg_idx), theta, frequency, ori_idx


def compute_transition_dir(pos, neg, theta):
    """
    Compute transition directions and count occurrences.

    Parameters:
        pos (numpy.ndarray or list): Indices of positive turning points.
        neg (numpy.ndarray or list): Indices of negative turning points.
        theta (numpy.ndarray): Array of direction angles.

    Returns:
        tuple: Tuple containing:
            - Dictionary containing counts of transitions between directions.
            - Lists of angle transitions for each direction pair.
    """
    if isinstance(pos, np.ndarray) and isinstance(neg, np.ndarray):
        # If inputs are numpy arrays, concatenate and sort
        merged_array = np.concatenate((pos, neg))
        merged = np.sort(merged_array)
    elif isinstance(pos, list) and isinstance(neg, list):
        # If inputs are lists, concatenate and sort
        merged = sorted(pos + neg)

    # Initialize counters for transitions
    count_dico = {"pos_neg": 0, "neg_pos": 0, "pos_pos": 0, "neg_neg": 0}
    list_a_to_b = []
    list_b_to_a = []
    list_a_to_a = []
    list_b_to_b = []

    # Iterate through the merged list
    for i in range(len(merged) - 1):
        current_item = merged[i]
        current_angle = theta[i]
        next_item = merged[i + 1]
        next_angle = theta[i + 1]

        # Check for transitions and increment counters
        if current_item in pos and next_item in neg:
            count_dico["pos_neg"] += 1
            list_a_to_b.append([current_angle, next_angle])
        elif current_item in neg and next_item in pos:
            count_dico["neg_pos"] += 1
            list_b_to_a.append([current_angle, next_angle])
        elif current_item in pos and next_item in pos:
            count_dico["pos_pos"] += 1
            list_a_to_a.append([current_angle, next_angle])
        elif current_item in neg and next_item in neg:
            count_dico["neg_neg"] += 1
            list_b_to_b.append([current_angle, next_angle])

    return count_dico, (list_a_to_b, list_b_to_a, list_a_to_a, list_b_to_b)