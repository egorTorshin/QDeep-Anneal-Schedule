from neal import SimulatedAnnealingSampler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sampler = SimulatedAnnealingSampler()
print("Connected to sampler:", sampler.__class__.__name__)

max_points = sampler.properties.get("max_anneal_schedule_points", "N/A")
annealing_range = sampler.properties.get("annealing_time_range", None)
if annealing_range is not None:
    max_slope = 1.0 / annealing_range[0]
    print("Annealing time range:", annealing_range)
    print("Maximum slope:", max_slope)
else:
    print("Annealing time range property not available for SimulatedAnnealingSampler")

from helpers.draw import plot_schedule

schedule = [[0.0, 0.0], [50.0, 0.5], [250.0, 0.5], [300.0, 1.0]]
print("Schedule:", schedule)
plot_schedule(schedule, "Example Anneal Schedule with Pause")

schedule = [[0.0, 0.0], [12.0, 0.6], [12.8, 1.0]]
print("Schedule:", schedule)
plot_schedule(schedule, "Example Anneal Schedule with Quench")

schedule = [[0.0, 0.0], [40.0, 0.4], [90.0, 0.4], [91.2, 1.0]]
print("Schedule:", schedule)
plot_schedule(schedule, "Example Anneal Schedule with Pause and Quench")

h = {0: 1.0, 1: -1.0, 2: -1.0, 3: 1.0, 4: 1.0, 5: -1.0, 6: 0.0, 7: 1.0,
     8: 1.0, 9: -1.0, 10: -1.0, 11: 1.0, 12: 1.0, 13: 0.0, 14: -1.0, 15: 1.0}
J = {(9, 13): -1, (2, 6): -1, (8, 13): -1, (9, 14): -1, (9, 15): -1,
     (10, 13): -1, (5, 13): -1, (10, 12): -1, (1, 5): -1, (10, 14): -1,
     (0, 5): -1, (1, 6): -1, (3, 6): -1, (1, 7): -1, (11, 14): -1,
     (2, 5): -1, (2, 4): -1, (6, 14): -1}

sampler_embedded = sampler

runs = 1000
results = sampler_embedded.sample_ising(h, J, num_reads=runs, num_sweeps=100)
print("Simulated annealer run complete.")

plt.hist(results.record.energy, rwidth=1, align='left', bins=[-21, -20, -19, -18, -17, -16, -15])
plt.xlabel("Energy")
plt.ylabel("Frequency")
plt.title("Histogram of Energies")
plt.show()

unique_energies, counts = np.unique(results.record.energy, return_counts=True)
if np.any(unique_energies == -20.0):
    ground_state_count = counts[unique_energies == -20.0][0]
else:
    ground_state_count = 0
print("Ground state probability:", ground_state_count / runs)

with open("files/saved_pause_results.json", "r") as read_file:
    saved_pause_success_prob = pd.read_json(read_file)

from helpers.draw import plot_success_fraction
pause_plot = plot_success_fraction(saved_pause_success_prob,
                                   "Success Fraction Using Pause for a Range of Anneal-Schedule Parameters",
                                   "pause_duration")

anneal_time = 20.0
pause_duration = 20.0     
pause_start = 0.3         

num_sweeps = int(anneal_time + pause_duration)
runs = 1000
results = sampler_embedded.sample_ising(h, J, num_reads=runs, num_sweeps=num_sweeps)
success = np.count_nonzero(results.record.energy == -20.0) / runs
print("Success probability (pause experiment):", success)

pause_plot["axis"].scatter([pause_start], [success], color="red", s=100)
pause_plot["figure"]

with open("files/saved_quench_results.json", "r") as read_file:
    saved_quench_success_prob = pd.read_json(read_file).replace(0, np.nan)

quench_plot = plot_success_fraction(saved_quench_success_prob,
                                    "Success Fraction Using Quench for a Range of Anneal-Schedule Parameters",
                                    "quench_slope")

anneal_time = 50.0
quench_slope = 1.0      
quench_start = 0.45     

num_sweeps = int((1 - quench_start + quench_slope * quench_start * anneal_time) / quench_slope)
runs = 1000
results = sampler_embedded.sample_ising(h, J, num_reads=runs, num_sweeps=num_sweeps)
success = np.count_nonzero(results.record.energy == -20.0) / runs
print("Success probability (quench experiment):", success)

quench_plot["axis"].scatter([quench_start], [success], color="red", s=100)
quench_plot["figure"]

anneal_times = [10.0, 100.0, 300.0]
pause_durations = [10.0, 100.0, 300.0]
num_points = 5
s_low = 0.2
s_high = 0.6
pause_starts = np.linspace(s_low, s_high, num=num_points)

success_prob = pd.DataFrame(index=range(len(anneal_times) * len(pause_durations) * len(pause_starts)),
                            columns=["anneal_time", "pause_duration", "s_feature", "success_frac"],
                            data=None)
counter = 0

print("Starting simulated annealer sweeps for pause experiment...")
for anneal in anneal_times:
    for pause in pause_durations:
        for start in pause_starts:
            num_sweeps = int(anneal + pause)
            runs = 1000
            results = sampler_embedded.sample_ising(h, J, num_reads=runs, num_sweeps=num_sweeps)
            success_frac = np.count_nonzero(results.record.energy == -20.0) / runs
            success_prob.iloc[counter] = {"anneal_time": anneal,
                                          "pause_duration": pause,
                                          "s_feature": start,
                                          "success_frac": success_frac}
            counter += 1
            print("Completed simulation for anneal_time={}, pause_duration={}, s_feature={}".format(anneal, pause, start))
print("Pause experiment sweeps complete.")
print(success_prob)

anneal_times = [10.0, 100.0, 300.0]
quench_slopes = [1.0, 0.5, 0.25]
num_points = 5
s_low = 0.2
s_high = 0.9
quench_starts = np.linspace(s_low, s_high, num=num_points)

success_prob_quench = pd.DataFrame(index=range(len(anneal_times) * len(quench_slopes) * len(quench_starts)),
                                   columns=["anneal_time", "quench_slope", "s_feature", "success_frac"],
                                   data=None)
counter = 0

print("Starting simulated annealer sweeps for quench experiment...")
for anneal in anneal_times:
    for quench in quench_slopes:
        for start in quench_starts:
            num_sweeps = int((1 - start + quench * start * anneal) / quench)
            runs = 1000
            results = sampler_embedded.sample_ising(h, J, num_reads=runs, num_sweeps=num_sweeps)
            success_frac = np.count_nonzero(results.record.energy == -20.0) / runs
            success_prob_quench.iloc[counter] = {"anneal_time": anneal,
                                                 "quench_slope": quench,
                                                 "s_feature": start,
                                                 "success_frac": success_frac}
            counter += 1
            print("Completed simulation for anneal_time={}, quench_slope={}, s_feature={}".format(anneal, quench, start))
print("Quench experiment sweeps complete.")
print(success_prob_quench)
