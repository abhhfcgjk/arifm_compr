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
        self.symbol_freq = np.ones(256)
        # self.b = np.zeros(257)
        self.prev_freq = np.cumsum(self.symbol_freq)
        self.total_freq: np.uint = 256
        # self.delitel: np.uint = 100
        # self.high: np.uint = 0x100000000 - 1
        # self.bits_high: np.uint8 = 32
        self.aggressiveness: np.uint = 5000
        # self.generate_probability()
        self.lim: np.uint = self.aggressiveness*50

    # def generate_probability(self):
    #     multiplier = self.delitel / self.size
    #     self.b[1:] = np.cumsum(self.table * multiplier) + self.b[0]


    def update(self, byte : np.uint8):
        self.total_freq += self.aggressiveness
        self.symbol_freq[byte] += self.aggressiveness
        self.prev_freq[byte:] = np.cumsum(self.symbol_freq[byte:])
        if self.symbol_freq[byte] >= self.lim:
            self.rescale()
        # self.generate_probability()

    # def cut_table(self):
    #     for i in range(self.table.size):
    #         if self.table[i] > 1:
    #             self.size -= self.table[i]//2
    #             self.table[i] //= 2

    
    def rescale(self):
        self.symbol_freq = self.symbol_freq - (self.symbol_freq >> 1)
        self.total_freq = np.sum(self.symbol_freq)
    
    
    # def getProbability(self, index: np.uint):
    #     return self.b[index+1]

    # def getElem(self, index: np.uint8) -> np.uint8:
    #     return index.tobytes()


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

        self.blk: np.int32      = np.int32(32)
        self.codebits: np.int32 = np.int32(24)
        self.top:np.int32       = np.int32(-1 >> (self.blk - self.codebits))
        self.bottom:np.int32    = np.int32(self.top>>8)
        self.bigbyte: np.int32  = np.int32(0xFF <<(self.codebits-8))
        self.range:int     = int(self.top)
        self.low:int       =  0

        # self.value = np.float64(0)
        # self.decoding: bool = False

    # def encode_normalize(self):
    #     while(self.range < self.bottom):
    #         if ((self.low & self.bigbyte) == self.bigbyte)and((self.range+(self.low & (self.bottom-1))) >= self.bottom):
    #             self.range = self.bottom - (self.low & self.bottom - 1)
    #         self.bitstream.write_byte(self.low >> 24)
    #         self.range <<= 8
    #         self.low <<= 8

    def encode_normalize(self):
        while self.range < self.bottom:
            if ((int(self.low) & int(self.bigbyte)) == int(self.bigbyte)) and ((int(self.range) + (int(self.low) & (int(self.bottom) - 1))) >= int(self.bottom)):
                self.range = int(self.bottom) - ((int(self.low) & int(self.bottom)) - 1)
            print(hex(np.int32(self.low)),hex(np.int32(self.range)))
            self.bitstream.write_byte(np.uint8(int(self.low) >> 24))

            self.range = int(self.range)
            self.range <<= 8
            self.low = int(self.low)
            self.low <<= 8
    
    def decode_normalize(self):
        while(self.range < self.bottom):
            if ((int(self.low) & int(self.bigbyte)) == int(self.bigbyte)) and ((int(self.range) + (int(self.low) & (int(self.bottom) - 1))) >= int(self.bottom)):
                self.range = int(self.bottom) - (int(self.low) & int(self.bottom) - 1)
        self.low = (int(self.low)<<8) | int(self.bitstream.read_byte())
        self.range <<= 8        

    # def init_encoding(self):
    #     self.h = np.zeros(np.uint(self.bitstream.size+1))
    #     self.h[0] = self.high
    #     self.l = np.zeros(np.uint(self.bitstream.size+1))
    #     self.i: int = 0

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


def compress_ari(ifile:str, ofile:str):
    bts = BitStream(ifile, ofile)

    AC = ArithmeticCompressor(bts, FrequencyTable())

    while (c := AC.bitstream.read_byte()).size:
        AC.encode(AC.frequency_table.symbol_freq[c], AC.frequency_table.prev_freq[c], AC.frequency_table.total_freq)
        AC.frequency_table.update(c)
    AC.bitstream.close()

def decompress_ari(ifile:str, ofile:str):
    pass

