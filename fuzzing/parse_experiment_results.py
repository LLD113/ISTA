import os
import statistics as stat

results = {}

for path, subdirs, files in os.walk("experiments"):
    for f in files:
        if f.endswith(".txt"):
            file_path = os.path.join(path, f)

            file = open(file_path, mode="r")
            lines = file.readlines()
            if lines[-3] != "checking adversarial\n":
                percent_adversarial = None
                nb_adversarial = None
                cov_inc = float(lines[-1].split(": ")[-1])
                final_cov = float(lines[-2].split(": ")[-1])
                nb_new_inputs = int(lines[-3].split(": ")[-1])
                iterations = int(lines[-4].split(": ")[-1])
                time_passed_min = float(lines[-5].split(": ")[-1])
                initial_cov = float(lines[-6].split(": ")[-1])
            else:
                percent_adversarial = float(lines[-1].split(": ")[-1]) * 100
                nb_adversarial = float(lines[-2].split(": ")[-1])
                cov_inc = float(lines[-4].split(": ")[-1])
                final_cov = float(lines[-5].split(": ")[-1])
                nb_new_inputs = int(lines[-6].split(": ")[-1])
                iterations = int(lines[-7].split(": ")[-1])
                time_passed_min = float(lines[-8].split(": ")[-1])
                initial_cov = float(lines[-9].split(": ")[-1])

            exp_name = file_path[:-6]

            if nb_new_inputs > 0:
                cov_inc_per_input = cov_inc / nb_new_inputs
            else:
                cov_inc_per_input = 0

            if exp_name not in results:
                results[exp_name] = {
                    'Coverage Increase': [cov_inc],
                    'Inital Coverage': [initial_cov],
                    'Final Coverage': [final_cov],
                    'Number of New Inputs': [nb_new_inputs],
                    'Number of Iterations': [iterations],
                    'Time Passed (Minutes)': [time_passed_min],
                    'Cov. Inc. per 1000 Input': [cov_inc_per_input * 1000],
                    'Number of Adversarial': [nb_adversarial],
                    'Percent of Adversarial': [percent_adversarial]
                }
            else:
                results[exp_name]['Coverage Increase'].append(cov_inc)
                results[exp_name]['Inital Coverage'].append(initial_cov)
                results[exp_name]['Final Coverage'].append(final_cov)
                results[exp_name]['Number of New Inputs'].append(nb_new_inputs)
                results[exp_name]['Number of Iterations'].append(iterations)
                results[exp_name]['Time Passed (Minutes)'].append(time_passed_min)
                results[exp_name]['Cov. Inc. per 1000 Input'].append(cov_inc_per_input * 1000)
                results[exp_name]['Number of Adversarial'].append(nb_adversarial)
                results[exp_name]['Percent of Adversarial'].append(percent_adversarial)

result_str = []
for exp_name in results:
    experiment_res = "EXPERIMENT: %s \n" % exp_name
    for f in results[exp_name].keys():
        if results[exp_name][f][0] != None:
            m = stat.mean(results[exp_name][f])
            std = stat.stdev(results[exp_name][f])
            experiment_res += "%s: %0.2f ± %0.2f \n" % (f, m, std)
    result_str.append(experiment_res)

result_str.sort()
print("\n".join(result_str))
