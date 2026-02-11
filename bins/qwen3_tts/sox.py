class Transformer:
    def norm(self, db_level=-6):
        return self

    def build_array(self, input_array=None, sample_rate_in=None):
        raise RuntimeError("sox is not available in this binary")
