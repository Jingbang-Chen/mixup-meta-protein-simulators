import run_md_aspirin
import tqdm
# result_prefix = "save_traj"

temp = [280,300,320]

for i in range(3):
    for j in range(10):
        run_md_aspirin.gen_data(results_prefix = f'data_Aspirin/save_traj_{temp[i]}_{j}', temperature = temp[i])

temp = [285,290,295,305,310,315,330,350]
for i in range(8):
    for j in range(3):
        run_md_aspirin.gen_data(results_prefix = f'data_Aspirin/save_traj_{temp[i]}_{j}', temperature = temp[i])
