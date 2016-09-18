# DataSet interface
class DataSet(object):
    def __init__(self):
        pass

    def get_next_frame(self):
        pass

    def tear_down(self):
        pass

    def get_num_of_frames_in_dataset(self):
        pass
