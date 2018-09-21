from shutil import copyfile
import copy, os, yaml, sys, random
from time import time

# a random semi-pronoucable uuid
def rand_id(num_syllables = 2, num_parts = 3):
    part1 = [ 's', 't', 'r', 'ch', 'b', 'c', 'w', 'z', 'h', 'k', 'p', 'ph', 'sh', 'f', 'fr' ]
    part2 = [ 'a', 'oo', 'ee', 'e', 'u', 'er', ]
    seps = [ '_', ] # [ '-', '_', '.', ]
    result = ""
    for i in range(num_parts):
        if i > 0:
            result += seps[random.randrange(len(seps))]
        indices1 = [random.randrange(len(part1)) for i in range(num_syllables)]
        indices2 = [random.randrange(len(part2)) for i in range(num_syllables)]
        for i1, i2 in zip(indices1, indices2):
            result += part1[i1] + part2[i2]
    return result


class FileLogger:
    ''' Helper that logs to a directory, original written by Russ Webb (github.com/rwebb)
        source_list: a list of files to save to the results directory
        log_def: definition of the logging { 'log_name1': [ keys ], 'log_name2': [ keys ], ...} '''
    def __init__(self, log_def, source_list=[],
                 path="results/", include_wall_time=True,
                 file_prefix=''):
        self.uuid = rand_id()
        self.dest_path = path
        if not os.path.isdir(path):
            os.makedirs(path)

        self.num_rows, self.start_time = 0, None
        self.log_def = copy.deepcopy(log_def)
        self.values, self.fp, self.info = {}, {}, {}
        self.file_prefix = file_prefix
        if include_wall_time:
            self.start_time = time()

        for log_name, items in log_def.items():
            csv_path = os.path.join(self.dest_path, file_prefix
                                    + self.uuid + "-"
                                    + log_name + '.csv')
            print('FILE LOG = ' + csv_path)
            self.fp[log_name] = open(csv_path, 'w')
            self.values[log_name] = {}
            for item in items:
                self.values[log_name][item] = 0.0

        self.source_list = source_list
        self.sources_copied = False

    def copy_sources(self):
        if self.sources_copied:
            return

        self.sources_copied = True
        for path in self.source_list:
            print("copying {}".format(path))
            _, filename = os.path.split(path)
            copyfile(path, os.path.join(self.dest_path, self.uuid + "-" + filename))

    def set_info(self, key, value): # rewrites yaml file each call
        self.info[key] = value
        with open(os.path.join(self.dest_path, self.uuid + '-info.yml'), 'w') as outfile:
            yaml.dump(self.info, outfile, default_flow_style = False)

    def record(self, log_name, key, value):
        self.values[log_name][key] = value

    def new_row(self, log_name):
        row = str(self.num_rows)
        self.num_rows += 1
        if self.start_time is not None:
            row += ',' + str(time() - self.start_time)

        for key in self.log_def[log_name]:
            row += ',' + str(self.values[log_name][key])

        self.fp[log_name].write(row + "\n")
        self.fp[log_name].flush()
