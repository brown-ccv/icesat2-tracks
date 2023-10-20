

class icesat2_confi:
    _instance = None
    def __new__(cls):
        if cls._instance is None:
            STARTUP_2021_IceSAT2="config/2021_IceSAT2_startup.py"
            exec(open(STARTUP_2021_IceSAT2).read())
            cls._instance = super(icesat2_confi, cls).__new__(cls)
        return cls._instance