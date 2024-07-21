import os

root_dir = r'F:\chb-mit-scalp-eeg-database-1.0.0'
output_path = r'F:\solver ep\record.txt'

with open(output_path, 'w') as f:
    for folder_name in os.listdir(root_dir):
        if folder_name.startswith('chb'):
            patient_id = folder_name.split('chb')[1]
            summary_file_path = os.path.join(root_dir, folder_name, folder_name + '-summary.txt')
            with open(summary_file_path) as summary_file:
                lines = summary_file.readlines()
                segments = {}
                for i, line in enumerate(lines):
                    if line.startswith('File Name'):
                        file_name = line.split(': ')[1].strip()
                    elif line.startswith('Number of Seizures in File'):
                        num_seizures = int(line.split(': ')[1].strip())
                        if num_seizures > 0:
                            for j in range(num_seizures):
                                start_line_idx = i + j * 2 + 1
                                end_line_idx = i + j * 2 + 2
                                start_time = int(lines[start_line_idx].split(': ')[1].strip()[:-7])
                                end_time = int(lines[end_line_idx].split(': ')[1].strip()[:-7])
                                key = f'{patient_id}_{file_name}'
                                if key in segments:
                                    segments[key].append((start_time, end_time))
                                else:
                                    segments[key] = [(start_time, end_time)]
                if segments:
                    f.write(f"No. {patient_id}\n")
                    for file_name, time_segments in segments.items():
                        for start_time, end_time in time_segments:
                            f.write(f"'{file_name}': ({start_time}, {end_time}),\n")
