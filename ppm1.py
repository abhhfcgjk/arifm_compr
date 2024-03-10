import numpy as np

class BitStream:
    def __init__(self, ifile:str, ofile:str):
        self.ifile = np.fromfile(ifile, dtype=np.uint8)
        self.file_size = self.ifile.size
        self.ibyte_idx = 0
        self.ofp = open(ofile, "wb")

    def read_byte(self) -> np.uint8:
        if self.ibyte_idx >= self.file_size:
            return np.array([], dtype=np.uint8)
        byte = self.ifile[self.ibyte_idx]
        self.ibyte_idx += 1
        return byte
    
    def write_byte(self, byte:np.uint8):
        self.ofp.write(byte)

    def close(self):
        self.ofp.close()

class FrequencyTable:
    def __init__(self):
        self.table = np.ones(256)
        self.b = np.zeros(257)
        self.size: np.uint = 256
        self.delitel: np.uint = 100
        self.high: np.uint = 0x100000000 - 1
        self.bits_high: np.uint8 = 32
        self.aggressiveness: np.uint = 5000
        # self.generate_probability()
        self.lim: np.uint = self.aggressiveness*50

    def generate_probability(self):
        multiplier = self.delitel / self.size
        self.b[1:] = np.cumsum(self.table * multiplier) + self.b[0]


    def update(self, byte : np.uint8):
        self.size += self.aggressiveness
        self.table[byte] += self.aggressiveness
        if self.table[byte] >= self.lim:
            self.rescale()
        # self.generate_probability()
    
    def symbol_freq(self, byte:np.uint8):
        return self.table[byte]

    def prev_freq(self, byte:np.uint8):
        return np.sum(self.table[:(byte-1)])
    
    def total_freq(self):
        return self.size

    # def cut_table(self):
    #     for i in range(self.table.size):
    #         if self.table[i] > 1:
    #             self.size -= self.table[i]//2
    #             self.table[i] //= 2

    
    def rescale(self):
        self.table = self.table - (self.table >> 1)
        self.size = np.sum(self.table)
    
    
    def getProbability(self, index: np.uint):
        return self.b[index+1]

    def getElem(self, index: np.uint8) -> np.uint8:
        return index.tobytes()


class ArithmeticCompressor:
    def __init__(self, bitstream : BitStream, frequency_table : FrequencyTable):
        self.bitstream = bitstream
        self.frequency_table = frequency_table
        # self.high = self.frequency_table.high
        # self.bits_high = self.frequency_table.bits_high
        # self.first_qtr:np.uint = np.uint((self.high + 1)/4)
        # self.half:np.uint = 2*self.first_qtr
        # self.third_qtr:np.uint = 3*self.first_qtr
        # self.bits_to_follow: np.uint8 = np.uint8(0)
        # self.delitel = self.frequency_table.delitel

        self.blk: np.uint8      = np.uint8(32)
        self.codebits: np.uint8 = np.uint8(24)
        self.top:np.int32       = np.int32(-1 >> (self.blk - self.codebits))
        self.bottom:np.int32    = self.top>>8
        self.bigbyte: np.int8   = np.int8(0xFF <<(self.codebits-8))
        self.range:np.int32     = self.top

        # self.value = np.float64(0)
        # self.decoding: bool = False

    def encode_normalize(self):
        while(self.range < self.bottom):
            if (self.low & self.bigbyte == self.bigbyte)and(self.range+(self.low & self.bottom-1) >= self.bottom):
                self.range = self.bottom - (self.low & self.bottom - 1)
            self.bitstream.write_byte(self.low >> 24)
            self.range <<= 8
            self.low <<= 8
    
    def decode_normalize(self):
        while(self.range < self.bottom):
            if (self.low & self.bigbyte == self.bigbyte)and(self.range+(self.low & self.bottom-1) >= self.bottom):
                self.range = self.bottom - (self.low & self.bottom - 1)
        self.low = (self.low<<8) | self.bitstream.read_byte()
        self.range <<= 8        

    def init_encoding(self):
        self.h = np.zeros(np.uint(self.bitstream.size+1))
        self.h[0] = self.high
        self.l = np.zeros(np.uint(self.bitstream.size+1))
        self.i: int = 0

    def encode(self, symbol_freq, prev_freq, total_freq):
        r = self.range // total_freq
        self.low += r*prev_freq
        self.range += r*symbol_freq
        self.encode_normalize()
        # self.frequency_table.update(0)


    def init_decoding(self, factor: np.uint8):
        self.bitstream.read_size()
        self.h = np.zeros(np.uint(self.bitstream.size+1))
        self.h[0] = np.uint(self.high)
        self.l = np.zeros(np.uint(self.bitstream.size+1))
        self.i = 0
        factor //= 8
        self.value = 0
        for _ in range(factor):
            self.value *= 0x100
            self.value += self.bitstream.read_byte()
    def decode(self, symbol_freq, prev_freq, total_freq) -> np.uint8:
        j = 0
        return np.uint8(j)


"""___________________________________________________________________________________________________"""


class ContextModel:
    def __init__(self):
        self.esc:np.uint = 0
        self.TotFreq:np.uint = 0
        self.count = np.zeros(256)
        self.table = np.ones(256)
        self.size = 256
        self.aggressiveness: np.uint = 5000
        self.lim: np.uint = self.aggressiveness*50

    def update(self, byte : np.uint8):
        self.size += self.aggressiveness
        self.table[byte] += self.aggressiveness
        if self.table[byte] >= self.lim:
            self.rescale()
    
    def symbol_freq(self, byte:np.uint8):
        return self.table[byte]

    def prev_freq(self, byte:np.uint8):
        return np.sum(self.table[:(byte-1)])
    
    def total_freq(self):
        return self.size

class PPMCompressor:
    def __init__(self, AC:ArithmeticCompressor):
        self.context_models = np.array(257, dtype=ContextModel)
        self.context_models[256].count = np.ones(256)
        self.context_models[256].TotFreq = 256
        self.context_models[256].esc = 0
        self.SP = 0
        self.context = [0]
        self.lim = 0x3FFF
        self.stack: list[ContextModel]
        self.AC = AC

    def update(self, c:np.uint8):
        while(self.SP):
            self.SP -= 1
            CM = self.stack[self.SP]
            if CM.TotFreq + CM.esc >= self.lim:
                CM.rescale()
            if not CM.count[c]:
                CM.esc += 1
            CM.count[c] += 1
            CM.TotFreq += 1

    def encode(self, CM:ContextModel, c:np.uint8) -> bool:
        self.stack[self.SP] = CM
        self.SP += 1
        if CM.count[c]:
            accumFreq: np.uint = np.sum(CM.count[:(c-1)])
            self.AC.encode(accumFreq, CM.count[c], CM.TotFreq+CM.esc)
            return True
        else:
            if CM.esc:
                self.AC.encode(CM.TotFreq, CM.esc, CM.TotFreq+CM.esc)
            return False


def compress_ppm(ifile : str, ofile : str):
    c: np.uint8 = 0
    success: bool = False
    
    AC = ArithmeticCompressor(BitStream(ifile, ofile), FrequencyTable())
    CM = PPMCompressor(AC)
    cm = CM.context_models
    AC.init_encoding()
    while (c := AC.bitstream.read_byte()).size:
        success = CM.encode(cm[CM.context[0]], c)
        if not success:
            CM.encode(cm[256], c)
        CM.update(c)
        CM.context[0] = c
        AC.frequency_table.update(c)
    AC.encode(start=cm[CM.context[0]].TotFreq, inter=cm[CM.context[0]].esc, size=cm[CM.context[0]].TotFreq+cm[CM.context[0]].esc)
    AC.encode(cm[CM.context[256]].TotFreq, cm[CM.context[256]].esc, cm[CM.context[256]].TotFreq+cm[CM.context[256]].esc)



def decompress_ppm(ifile : str, ofile : str):
    with open(ifile, 'rb') as ifp, open(ofile, 'wb') as ofp:
        ### PUT YOUR CODE HERE
        ### implement ppm algorithm for decompression

        ### This is an implementation of simple copying

        text = ifp.read()
        ofp.write(text)
