from tqdm import tqdm
import datetime
import sys
DEFAULT_PATH_ANALYSES = 'analyses/'
DEFAULT_PATH_LOG = 'logs/'


class Analyse:

    def __init__(self, original_file_swarm, corrected_file_swarm, failed_file_swarm, analyse_file, analyse_file_mode,
                 dense_layers, threshold, pif, dataset, seed, rna, mode):

        self.original_file_swarm = []
        self.corrected_file_swarm = []
        self.failed_file_swarm = []

        self.original_file_dict = {}
        self.corrected_file_dict = {}
        self.failed_file_dict = {}

        self.trace_found_in_original_and_corrected = 0
        self.trace_found_in_original_and_failed = 0
        self.trace_found_in_original_and_failed_and_corrected = 0
        self.original_file_swarm_location = original_file_swarm
        self.corrected_file_swarm_location = corrected_file_swarm
        self.failed_file_swarm_location = failed_file_swarm
        self.size_list_corrected = 0
        self.size_list_original = 0
        self.size_list_failed = 0

        self.analyse_file = analyse_file
        self.analyse_file_mode = analyse_file_mode
        self.dense_layers = dense_layers
        self.threshold = threshold
        self.pif = pif
        self.dataset = dataset
        self.seed = seed
        self.rna = rna
        self.mode = mode

    def load_corrected_swarm(self):

        file_swarm_original = open(self.corrected_file_swarm_location, 'r')

        for i, j in enumerate(file_swarm_original):
            if j[0] == "#":
                print("header: {}".format(j))
            else:
                #snapshot #peer #modified?
                keys = j.split(' ')
                #snapshot #peer
                self.corrected_file_swarm.append([int(keys[0]), int(keys[1])])

        self.size_list_corrected = len(self.corrected_file_swarm)
        self.corrected_file_swarm = sorted(self.corrected_file_swarm, key=lambda x: x[0])

    def load_original_swarm(self):

        file_swarm_original = open(self.original_file_swarm_location, 'r')

        for i, j in enumerate(file_swarm_original):

            if i != 0:

                keys = j.split(' ')
                self.original_file_swarm.append([int(keys[0]), int(keys[3])])

        self.size_list_original = len(self.original_file_swarm)

    def load_failed_swarm(self):

        file_swarm_original = open(self.failed_file_swarm_location, 'r')

        for i, j in enumerate(file_swarm_original):

            if i != 0:

                keys = j.split(' ')

                self.failed_file_swarm.append([int(keys[0]), int(keys[3])])

        self.size_list_failed = len(self.failed_file_swarm)


    def load_file_swarm_to_dict_old(self, file_name, peer_key=0, snapshot_key=3):
        result = {}
        count = 0
        with open(file_name, 'r') as file:
            for line in file:
                if line[0] == "#":
                    print("header: {}".format(line))
                else:
                    count += 1
                    #peer #snapshot
                    keys = line.split(' ')
                    peer = int(keys[peer_key])
                    snapshot = int(keys[snapshot_key])

                    #print("peer: {} snapshot: {}".format(peer, snapshot))
                    if peer not in result.keys():
                        result[peer] = []
                    result[peer].append(snapshot)

        return result, count

    def run_analise(self):
        print("\toriginal_file_swarm_location : {}".format(self.original_file_swarm_location))
        print("\tfailed_file_swarm_location   : {}".format(self.failed_file_swarm_location))
        print("\tcorrected_file_swarm_location: {}".format(self.corrected_file_swarm_location))

        self.failed_file_dict, self.size_list_failed = self.load_file_swarm_to_dict_old(self.failed_file_swarm_location,
                                                                                        peer_key=2, snapshot_key=0)

        print("self.failed_file_dict: {} self.size_list_failed: {} ".format(len(self.failed_file_dict.keys()),
                                                                                self.size_list_failed))


        self.corrected_file_dict, self.size_list_corrected = self.load_file_swarm_to_dict_old(
            self.corrected_file_swarm_location,
            peer_key=1, snapshot_key=0)

        print("corrected_file_dict: {} self.size_list_corrected: {} ".format(len(self.corrected_file_dict.keys()),
                                                                                self.size_list_corrected))

        with open(self.original_file_swarm_location, 'r') as original_file:
            self.size_list_original = 0
            #tamanho é apenas uma aproximação do valor real (ref. arquivo original) para reduzir custo computacional
            with tqdm(total=self.size_list_corrected) as pbar:
                for line in original_file:
                    if line[0] == "#":
                        print("header: {}".format(line))
                    else:
                        pbar.update(1)
                        self.size_list_original += 1
                        # peer #snapshot
                        keys = line.split(' ')
                        peer = int(keys[3])
                        snapshot = int(keys[0])

                        #print("\tsnapshot: {}".format(snapshot))
                        peer_snapshot_failed = self.search_in_dict(self.failed_file_dict, peer, snapshot)
                        #print("swarm_failed   : {} ".format(swarm_failed))
                        peer_snapshot_corrected = self.search_in_dict(self.corrected_file_dict, peer, snapshot)
                        #print("swarm_corrected: {} ".format(swarm_corrected))

                        if peer_snapshot_failed:
                            self.trace_found_in_original_and_failed += 1

                        if peer_snapshot_corrected:
                            self.trace_found_in_original_and_corrected += 1

                        if peer_snapshot_failed and peer_snapshot_corrected:
                            self.trace_found_in_original_and_failed_and_corrected += 1

    def run_analise_OLD(self):

        self.load_corrected_swarm()
        self.load_failed_swarm()
        self.load_original_swarm()

        for i in tqdm(range(len(self.original_file_swarm)), desc='Analyzing'):

            key_1, key_2 = self.original_file_swarm[i]

            swarm_failed = self.search_failed(key_1, key_2)
            swarm_corrected = self.search_corrected(key_1, key_2)

            if swarm_failed:

                self.trace_found_in_original_and_failed += 1

            if swarm_corrected:

                self.trace_found_in_original_and_corrected += 1

            if swarm_corrected and swarm_failed:

                self.trace_found_in_original_and_failed_and_corrected += 1

    def search_corrected(self, key_1, key_2):

        for i, j in enumerate(self.corrected_file_swarm):
            #snapshot
            if j[0] == key_1:
                #peer
                if j[1] == key_2:
                    del self.corrected_file_swarm[i]
                    return True

        return False

    def search_failed(self, key_1, key_2):

        for i, j in enumerate(self.failed_file_swarm):
            # snapshot
            if j[0] == key_1:
                # peer
                if j[1] == key_2:
                    del self.failed_file_swarm[i]
                    return True

        return False

    def load_file_swarm_to_dict(self, file_name, inversed=False):
        result = {}
        count = 0
        with open(file_name, 'r') as file:
            for line in file:
                if line[0] == "#":
                    print("header: {}".format(line))
                else:
                    count += 1
                    #peer #snapshot
                    keys = line.split(' ')
                    peer = int(keys[0])
                    snapshot = int(keys[1])

                    if inversed:
                        peer = int(keys[1])
                        snapshot = int(keys[0])

                    #print("peer: {} snapshot: {}".format(peer, snapshot))
                    if peer not in result.keys():
                        result[peer] = []
                    result[peer].append(snapshot)

        return result, count

    def load_corrected_swarm_mif_probabilistic(self):

        file_swarm_original = open(self.corrected_file_swarm_location, 'r')

        for i, j in enumerate(file_swarm_original):
            if j[0] == "#":
                print("header: {}".format(j))
            else:
                #snapshot #peer #modified?
                keys = j.split(' ')
                #snapshot #peer
                self.corrected_file_swarm.append([int(keys[0]), int(keys[1])])

        self.size_list_corrected = len(self.corrected_file_dict.keys())
        self.corrected_file_swarm = sorted(self.corrected_file_swarm, key=lambda x: x[0])

    def load_corrected_swarm_mif(self):

        file_swarm_original = open(self.corrected_file_swarm_location, 'r')

        for i, j in enumerate(file_swarm_original):
            if j[0] == "#":
                print("header: {}".format(j))
            else:
                #snapshot #peer #modified?
                keys = j.split(' ')
                #snapshot #peer
                self.corrected_file_swarm.append([int(keys[0]), int(keys[1])])

        self.size_list_corrected = len(self.corrected_file_dict.keys())
        self.corrected_file_swarm = sorted(self.corrected_file_swarm, key=lambda x: x[0])

    def load_original_swarm_mif(self):

        file_swarm_original = open(self.original_file_swarm_location, 'r')

        for i, j in enumerate(file_swarm_original):
            if j[0] == "#":
                print("header: {}".format(j))
            else:
                #peer #snapshot
                keys = j.split(' ')
                #snapshot #peer
                self.original_file_swarm.append([int(keys[1]), int(keys[0])])

        self.size_list_original = len(self.original_file_swarm)

    def load_failed_swarm_mif(self):

        with open(self.failed_file_swarm_location, 'r') as file:

            for line in file:
                if line[0] == "#":
                    print("header: {}".format(j))
                else:
                    #peer #snapshot
                    keys = line.split(' ')
                    peer = int(keys[0])
                    snapshot = int(keys[1])

                    #snapshot #peer
                    self.failed_file_swarm.append([snapshot, peer])

                    if peer not in self.failed_file_dict.keys():
                        self.failed_file_dict[peer] = []
                    self.failed_file_dict[peer].append(snapshot)

        self.size_list_failed = len(self.failed_file_swarm)

    def search_in_dict(self, dict, peer, snapshot):

        if peer in dict.keys():
            if snapshot in dict[peer]:
                return True

        return False

    def run_analise_mif_probabilistic(self):
        print("Original : {}".format(self.original_file_swarm_location))
        print("Failed   : {}".format(self.failed_file_swarm_location))
        print("Corrected: {}".format(self.corrected_file_swarm_location))

        self.corrected_file_dict, self.size_list_corrected = self.load_file_swarm_to_dict(
            self.corrected_file_swarm_location)
        print("corrected_file_dict: {} self.size_list_corrected: {} ".format(len(self.corrected_file_dict.keys()),
                                                                             self.size_list_corrected))

        self.failed_file_dict, self.size_list_failed = self.load_file_swarm_to_dict(self.failed_file_swarm_location)
        print("failed_file_dict   : {} self.size_list_failed   : {} ".format(len(self.failed_file_dict.keys()),
                                                                                self.size_list_failed))


        with open(self.original_file_swarm_location, 'r') as original_file:
            self.size_list_original = 0
            #tamanho é apenas uma aproximação do valor real (ref. arquivo original) para reduzir custo computacional
            with tqdm(total=self.size_list_corrected) as pbar:
                for line in original_file:
                    if line[0] == "#":
                        print("header: {}".format(line))
                    else:
                        pbar.update(1)
                        self.size_list_original += 1
                        # peer #snapshot
                        keys = line.split(' ')
                        peer = int(keys[0])
                        snapshot = int(keys[1])

                        #print("\tpeer: {} snapshot: {}".format(peer, snapshot))
                        peer_snapshot_failed = self.search_in_dict(self.failed_file_dict, peer, snapshot)
                        #print("swarm_failed   : {} ".format(peer_snapshot_failed))
                        peer_snapshot_corrected = self.search_in_dict(self.corrected_file_dict, peer, snapshot)
                        #print("swarm_corrected: {} ".format(peer_snapshot_corrected))

                        if peer_snapshot_failed:
                            self.trace_found_in_original_and_failed += 1

                        if peer_snapshot_corrected:
                            self.trace_found_in_original_and_corrected += 1

                        if peer_snapshot_failed and peer_snapshot_corrected:
                            self.trace_found_in_original_and_failed_and_corrected += 1

    def run_analise_mif(self):

        # self.original_file_dict, self.size_list_original = self.load_file_swarm_to_dict(
        #     self.original_file_swarm_location)
        # print("self.original_file_dict: {} self.size_list_original: {} ".format(len(self.original_file_dict), self.size_list_original))

        self.failed_file_dict, self.size_list_failed = self.load_file_swarm_to_dict(self.failed_file_swarm_location)
        print("self.failed_file_dict: {} self.size_list_failed: {} ".format(len(self.failed_file_dict.keys()),
                                                                                self.size_list_failed))
        self.corrected_file_dict, self.size_list_corrected = self.load_file_swarm_to_dict(
            self.corrected_file_swarm_location, True)
        print("corrected_file_dict: {} self.size_list_corrected: {} ".format(len(self.corrected_file_dict.keys()),
                                                                                self.size_list_corrected))

        with open(self.original_file_swarm_location, 'r') as original_file:
            self.size_list_original = 0
            #tamanho é apenas uma aproximação do valor real (ref. arquivo original) para reduzir custo computacional
            with tqdm(total=self.size_list_corrected) as pbar:
                for line in original_file:
                    if line[0] == "#":
                        print("header: {}".format(line))
                    else:
                        pbar.update(1)
                        self.size_list_original += 1
                        # peer #snapshot
                        keys = line.split(' ')
                        peer = int(keys[0])
                        snapshot = int(keys[1])

                        #print("\tsnapshot: {}".format(snapshot))
                        peer_snapshot_failed = self.search_in_dict(self.failed_file_dict, peer, snapshot)
                        #print("swarm_failed   : {} ".format(swarm_failed))
                        peer_snapshot_corrected = self.search_in_dict(self.corrected_file_dict, peer, snapshot)
                        #print("swarm_corrected: {} ".format(swarm_corrected))

                        if peer_snapshot_failed:
                            self.trace_found_in_original_and_failed += 1

                        if peer_snapshot_corrected:
                            self.trace_found_in_original_and_corrected += 1

                        if peer_snapshot_failed and peer_snapshot_corrected:
                            self.trace_found_in_original_and_failed_and_corrected += 1
                        #print("trace_found_in_original_and_failed   : {}".format(self.trace_found_in_original_and_failed))
                        #print("trace_found_in_original_and_corrected: {}".format(self.trace_found_in_original_and_corrected))
                        #print("trace_found_in_original_and_failed_and_corrected: {}".format(self.trace_found_in_original_and_failed_and_corrected))
                        #sys.exit()

        # for i in tqdm(range(len(self.original_file_swarm)), desc='Analyzing '):
        #     snapshot, peer = self.original_file_swarm[i]
        #     print("original       : peer: {} snapshot:{}".format(peer, snapshot))
        #     swarm_failed = self.search_failed_mif(snapshot, peer)
        #     print("swarm_failed   : {} ".format(swarm_failed))
        #     swarm_corrected = self.search_corrected_mif(snapshot, peer)
        #     print("swarm_corrected: {} ".format(swarm_corrected))
        #
        #     if swarm_failed:
        #         self.trace_found_in_original_and_failed += 1
        #
        #     if swarm_corrected:
        #         self.trace_found_in_original_and_corrected += 1
        #
        #     if swarm_corrected and swarm_failed:
        #         self.trace_found_in_original_and_failed_and_corrected += 1
        #     print("trace_found_in_original_and_failed: {} trace_found_in_original_and_corrected: {} trace_found_in_original_and_failed_and_corrected: {}".format(
        #         self.trace_found_in_original_and_failed,self.trace_found_in_original_and_corrected, self.trace_found_in_original_and_failed_and_corrected))
        #



    def write_results_analyse(self, start_time, size_window_left, size_window_right):

        analyse_results = open(self.analyse_file, self.analyse_file_mode)
        topology = "["

        for i in range(self.dense_layers):
            topology += "20, "

        topology += "1]"

        analyse_results.write('\nBEGIN ############################################\n\n')
        analyse_results.write(' RESULTS \n')
        end_time = datetime.datetime.now()
        duration = end_time - start_time
        analyse_results.write("  Start time: {}\n".format(start_time))
        analyse_results.write("  End time  : {}\n".format(end_time))
        analyse_results.write("  Duration  : {}\n".format(duration))
        analyse_results.write("  RNA       : {}\n".format(self.rna))
        analyse_results.write("  MODE      : {}\n".format(self.mode))
        analyse_results.write("  Topology  : {}\n".format(topology))
        analyse_results.write("  Threshold : {}\n".format(self.threshold))
        analyse_results.write("  PIF       : {}%\n".format(int(self.pif * 100)))
        analyse_results.write("  Dataset   : {}\n".format(self.dataset))
        analyse_results.write("  Seed      : {}\n\n".format(self.seed))
        analyse_results.write('  Size files:           \n')
        analyse_results.write('-----------------------------\n')
        analyse_results.write('  Total Traces original file         : {}\n'.format(self.size_list_original))
        analyse_results.write('  Total Traces failed file           : {}\n'.format(self.size_list_failed))
        analyse_results.write('  Total Traces corrected file        : {}\n'.format(self.size_list_corrected))

        faults = self.size_list_original - self.size_list_failed
        analyse_results.write('  Fails (Original-failed)            : {}\n'.format(faults))

        modification = self.size_list_corrected - self.size_list_failed
        analyse_results.write('  Modifications (Original-corrected) : {}\n'.format(modification))

        analyse_results.write('------------------------------\n')
        analyse_results.write('            Analyse:          \n')
        analyse_results.write('------------------------------\n')
        analyse_results.write('  Found in [Original, Corrected, Failed]: {}\n'.format(self.trace_found_in_original_and_failed_and_corrected))
        analyse_results.write('  Found in [Original, Corrected]        : {}\n'.format(self.trace_found_in_original_and_corrected))
        analyse_results.write('  Found in [Original, Failed]           : {}\n'.format(self.trace_found_in_original_and_failed))
        analyse_results.write('------------------------------\n')
        analyse_results.write('            Scores:           \n')
        analyse_results.write('------------------------------\n')
        tp = self.trace_found_in_original_and_corrected-self.trace_found_in_original_and_failed
        analyse_results.write('  True positive  (TP): {}\n'.format(tp))
        fp = self.size_list_corrected-self.trace_found_in_original_and_corrected
        analyse_results.write('  False positive (FP): {}\n'.format(fp))
        fn = self.size_list_original-self.trace_found_in_original_and_corrected
        analyse_results.write('  False negative (FN): {}\n'.format(fn))

        tn = self.size_list_original -tp -(fp+fn)
        analyse_results.write('  True negative  (TN): {}\n'.format(tn))

        # line = "#SUMMARY-OLD#"
        # line += ";{}".format(topology)
        # line += ";{}".format(self.size_list_original)
        # line += ";{}".format(faults)
        # line += ";{}".format(self.threshold)
        # line += ";{}%".format(int(self.pif*100))
        # line += ";{}".format(self.dataset)
        # line += ";{}".format(self.threshold)
        # line += ";{}".format(self.seed)
        # line += ";{}".format(tp)
        # line += ";{}".format(fp)
        # line += ";{}".format(fn)
        # line += ";{}".format(tn)
        # line += "\n"
        # print(line)

        # RNA	# topology	# threshold	# pif	# dataset	# seed	# snapshots	# fails	# modifications
        line ="#SUMMARY#"
        line += ";{}".format(self.mode)
        line += ";{}".format(topology)
        line += ";{}".format(size_window_left+size_window_right+1)
        line += ";{}".format(self.threshold)
        line += ";{}%".format(int(self.pif * 100))
        line += ";{}".format(self.dataset)
        line += ";{}".format(self.seed)

        line += ";{}".format(duration)
        line += ";{}".format(self.size_list_original)
        line += ";{}".format(faults)
        line += ";{}".format(modification)

        line += ";{}".format(tp)
        line += ";{}".format(fp)
        line += ";{}".format(fn)
        line += ";{}".format(tn)
        line += "\n"
        line = line.replace(";", "\t")
        print(line)

        analyse_results.write(line)
        analyse_results.write('\nEND ############################################\n\n')
        analyse_results.write('\n\n\n')
        analyse_results.close()
