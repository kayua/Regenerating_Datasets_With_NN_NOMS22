DEFAULT_NUM = 0.98


class Dataset:

    def __init__(self, size_window_left=None, size_window_right=None):

        self.output_file = None
        self.size_window_left = size_window_left
        self.size_window_right = size_window_right
        self.file_swarm = None
        self.file_swarm_lines = None
        self.current_file_swarm_line = None

    def load_samples(self, swarm_file):

        try:
            self.file_swarm = open(swarm_file, 'r')
            self.file_swarm_lines = self.file_swarm.readline()
            self.current_file_swarm_line = self.file_swarm_lines

        except FileNotFoundError:
            print("File swarm not found!")
            exit()

    def load_next_peer(self):

        list_peer_snapshots = []

        while True:

            if self.file_swarm_lines[0] == "#":
                self.file_swarm_lines = self.file_swarm.readline()
                continue #skipt #commented line

            line_loaded = self.file_swarm_lines.split(' ')

            line_file_read = [int(line_loaded[2]), int(line_loaded[0]), 1]
            list_peer_snapshots.append(line_file_read)
            self.file_swarm_lines = self.file_swarm.readline()

            try:
                if self.file_swarm_lines == "":
                    return list_peer_snapshots, 0

                if int(self.file_swarm_lines.split(' ')[2]) != line_file_read[0]:
                    return list_peer_snapshots, 1

            except IndexError:
                return list_peer_snapshots, 1

    def load_next_peer_mif(self):
        POS_PEER = 0
        POS_SNAPSHOT = 1

        list_peer_snapshots = []

        while True:

            if self.current_file_swarm_line[0] == "#":
                self.current_file_swarm_line = self.file_swarm.readline()
                continue #skip it #commented line

            # peer snapshot
            line_split = self.current_file_swarm_line.split(' ')
            peer = int(line_split[POS_PEER])       #
            snapshot = int(line_split[POS_SNAPSHOT])   #
            line_file_read = [peer, snapshot, 1] #TODO: check
            list_peer_snapshots.append(line_file_read)
            self.current_file_swarm_line = self.file_swarm.readline()

            try:
                if self.current_file_swarm_line == "":
                    return list_peer_snapshots, 0

                #if the next line is about another peer, get out
                if int(self.current_file_swarm_line.split(' ')[POS_PEER]) != peer:
                    return list_peer_snapshots, 1

            except IndexError:
                return list_peer_snapshots, 1

    def load_next_peer2(self):

        list_peer_snapshots = []

        while True:

            self.file_swarm_lines = self.file_swarm.readline()

            # linha comentada
            if self.file_swarm_lines[0] == "#":
                self.file_swarm_lines = self.file_swarm.readline()
                continue #skipt #commented line

            #linha vazia/ bugada
            try:
                if self.file_swarm_lines == "":
                    return list_peer_snapshots, 0

                if int(self.file_swarm_lines.split(' ')[2]) != line_file_read[0]:
                    return list_peer_snapshots, 1

            except IndexError:
                return list_peer_snapshots, 1

            # linha ok
            line_loaded = self.file_swarm_lines.split(' ')

            line_file_read = [int(line_loaded[2]), int(line_loaded[0]), 1]
            list_peer_snapshots.append(line_file_read)




    @staticmethod
    def fill_gaps_per_peer(list_snapshots):
        #print('snapshots input : {}'.format(list_snapshots))
        list_snapshots.sort(key=lambda x: x[1])

        if not len(list_snapshots[-1]) != 0:

            return 0

        iterator, iterator_list, temporary_list = list_snapshots[0][1], 0, []

        while iterator < int(list_snapshots[-1][1]):

            if list_snapshots[iterator_list][1] != iterator:

                temporary_list.append([int(list_snapshots[1][0]), int(iterator), 0])

            else:

                temporary_list.append(list_snapshots[iterator_list])
                iterator_list += 1

            iterator += 1

        temporary_list.append(list_snapshots[-1])
        #print('snapshots output: {}'.format(temporary_list))
        #sys.exit()
        return temporary_list

    def filling_borders(self, list_snapshots):

        window_left = [[list_snapshots[0][0], -1, 0]] * self.size_window_left
        window_right = [[list_snapshots[0][0], -1, 0]] * self.size_window_left

        return window_left + list_snapshots + window_right

    def create_windows(self, list_snapshots):

        list_x, list_y, list_support = [], [], []

        try:

            for i in range(self.size_window_left, len(list_snapshots) - self.size_window_right):
                list_x.append(list_snapshots[(i - self.size_window_left):(i + self.size_window_right + 1)])
                list_y.append(list_snapshots[i][2])
                list_support.append(list_snapshots[i])

            return list_x, list_y, list_support

        except IndexError:

            return [], []

    @staticmethod
    def get_samples_vectorized(sample, out):

        sample_vectorized = []

        for i in range(len(sample)):

            sample_vectorized.append(float(sample[i][2]))

        return sample_vectorized, float(out)

    def create_sample_training(self, list_snapshots):

        sample_x, sample_y = [], []

        list_x, list_y, list_support = self.create_windows(list_snapshots)

        for i in range(len(list_x)):

            x, y = self.get_samples_vectorized(list_x[i], list_y[i])
            sample_x.append(x)
            sample_y.append(float(y))

        return sample_x, sample_y, list_support

    def get_training_samples(self, list_snapshots):

        x, y, support_samples = self.create_sample_training(list_snapshots)

        for i in range(len(x)):

            true_position_left = False

            for j in range(2):

                if x[i][self.size_window_left-j-1] > DEFAULT_NUM:

                    true_position_left = True

            true_position_right = False

            for j in range(2):

                if x[i][self.size_window_left + j+1] > DEFAULT_NUM:

                    true_position_right = True

            if (not true_position_left) or (not true_position_right):

                y[i] = float(0)

            if x[i][self.size_window_left-1] > DEFAULT_NUM and x[i][self.size_window_left+1] > DEFAULT_NUM:

                y[i] = float(1)

            x[i][self.size_window_left] = float(0)
            x[i] = [float(i) for i in x[i]]

        return x, y, support_samples

    def get_predict_samples(self, list_snapshots):

        x, y, support_samples = self.create_sample_training(list_snapshots)

        for i in range(len(x)):

            x[i][self.size_window_left] = float(0)
            x[i] = [float(i) for i in x[i]]

        return x, y, support_samples

    def create_file_results(self, file_results):

        self.output_file = open(file_results, 'w')

    def write_swarm(self, values):

        for i in values:
            output_result_format = str(i[1]) + ' ' + str(i[0]) + ' ' + str(i[2]) + '\n'
            self.output_file.write(str(output_result_format))

