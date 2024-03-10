import numpy as np


class BitStream:
    def __init__(self, ifile : str, ofile : str):
        """Create a BitStream, open ifile and ofile, initialize buffers, ...

        Parameters
        ----------
        ifile: str
            Name of input file
        ofile: str
            Name of output file
        """
        # self.ifp = open(ifile, "rb")
        self.ofp = open(ofile, "wb")
        self.ifp = np.fromfile(ifile, dtype=np.uint8)
        self.input_size:np.uint32 = np.uint32(self.ifp.size)
        self.__readed_byte: np.uint8 = np.uint8(0)
        self.__eof: bool = False
        self.__count_unreaded_bits_from_byte: np.uint8 = np.uint8(0)
        self.__count_writed_bits_to_byte: np.uint8 = np.uint8(0)
        self.__byte_to_write: np.uint8 = np.uint8(0)
        self.size:np.uint32 = self.input_size
        self.output_size: np.uint32 = np.uint32(0)
        self.cur: np.uint32 = np.uint32(0)
    
    
    
    def read_size(self)->np.uint:
        self.cur = 4 #
        self.size = self.ifp[0] + self.ifp[1]*16*16 + self.ifp[2]*16*16*16*16 + self.ifp[3]*16*16*16*16*16*16 #
        # print(self.size)
        return self.size #

    def write_size(self):
        # print(self.input_size.tobytes(), self.input_size)
        self.ofp.write(self.input_size.tobytes())

    def read_bit(self) -> np.uint8:
        """Read 1 bit from ifile

        Returns
        -------
        bit: np.uint8
            Next bit from ifile
        """
        if(self.__count_unreaded_bits_from_byte == 0):# !!!!!!!!!!!!!!!!! <= 0 -> ==0
            if not self.__eof:
                self.__readed_byte = self.read_byte()
                self.__count_unreaded_bits_from_byte = np.uint8(8)
            else:
                return 0
        self.__count_unreaded_bits_from_byte -= np.uint8(1)
        bit = (self.__readed_byte & 0x80) >> 7
        self.__readed_byte <<= 1
        return np.uint8(bit)
        


    def write_bit(self, bit : np.uint8):
        """Write 1 bit to ofile

        Parameters
        ----------
        bit: np.uint8
            1 bit to write to ofile
        """
        if(self.__count_writed_bits_to_byte < 8):
            self.__byte_to_write <<= np.uint8(1)
            self.__byte_to_write |= bit
            self.__count_writed_bits_to_byte += np.uint8(1)
            if(self.__count_writed_bits_to_byte == 8):
                self.write_byte(np.uint8(self.__byte_to_write))
                self.__byte_to_write = np.uint8(0)
                self.__count_writed_bits_to_byte = np.uint8(0)


    def read_byte(self) -> np.uint8:
        """Read 1 byte (symbol) from ifile

        Returns
        -------
        byte: np.uint8
            Next byte (symbol) from ifile
        """
        
        if self.cur < self.input_size:
            self.__readed_byte = self.ifp[self.cur]
            self.cur += 1

            # print(self.cur)

            return self.__readed_byte
        self.__eof = True
        return 0


    def write_byte(self, byte : np.uint8):
        """Write 1 byte (symbol) to ofile

        Parameters
        ----------
        byte : np.uint8
            1 byte (symbol) to write to ofile
        """
        self.output_size += 1
        self.ofp.write(byte.tobytes())


    def close(self):
        """Write last bits from buffer to ofile 
           and close ifile, ofile
        """
        if(self.__count_writed_bits_to_byte != 0):
            self.__byte_to_write <<= np.uint8(8 - self.__count_writed_bits_to_byte)
            self.ofp.write(np.uint8(self.__byte_to_write).tobytes())
        self.ofp.close()

    def EOF(self) ->bool:
        return self.__eof

    def BitsPlusFollow(self, bit: np.uint8, bits_to_follow: np.uint):
        self.write_bit(bit)
        while(bits_to_follow > 0):
            self.write_bit(not bit)
            bits_to_follow -= 1



class FrequencyTable:
    def __init__(self):
        """Create frequency table
        """
        self.table = np.ones(256)
        self.b = np.zeros(257)
        self.size: np.uint = 256
        self.delitel: np.uint = 100
        self.high: np.uint = 0x100000000 - 1
        self.bits_high: np.uint8 = 32
        self.aggressiveness: np.uint = 5000
        self.generate_probability()
        self.lim: np.uint = self.aggressiveness*50
        ### test_3: agr = 1000, lim = agr*1000
        ### test_7: arg = 3000, lim = 100000
        ### 3695610 - 15000, 15000*500/ 10000, 10000*500

    def generate_probability(self):
        multiplier = self.delitel / self.size
        self.b[1:] = np.cumsum(self.table * multiplier) + self.b[0]


    def update(self, byte : np.uint8):
        """Use 1 byte (symbol) to update frequency table

        Parameters
        ----------
        byte : np.uint8
            1 byte (symbol)
        """
        self.size += self.aggressiveness
        self.table[byte] += self.aggressiveness
        # np.sort(self.table)
        if self.table[byte] >= self.lim:
            self.cut_table()
        self.generate_probability()

    def cut_table(self):
        for i in range(self.table.size):
            if self.table[i] > 1:
                self.size -= self.table[i]//2
                self.table[i] //= 2
    
    
    def getProbability(self, index: np.uint):
        return self.b[index+1]

    def getElem(self, index: np.uint8) -> np.uint8:
        return index.tobytes()


class ArithmeticCompressor:
    def __init__(self, bitstream : BitStream, frequency_table : FrequencyTable):
        """Create arithmetic compressor, initialize all parameters

        Parameters
        ----------
        bitstream : BitStream
            bitstream to read/write bits/bytes
        frequency_table : FrequencyTable
            Frequency table for arithmetic compressor
        """
        self.bitstream = bitstream
        self.frequency_table = frequency_table
        self.high = self.frequency_table.high
        self.bits_high = self.frequency_table.bits_high
        self.first_qtr:np.uint = np.uint((self.high + 1)/4)
        self.half:np.uint = 2*self.first_qtr
        self.third_qtr:np.uint = 3*self.first_qtr
        self.bits_to_follow: np.uint8 = np.uint8(0)
        self.delitel = self.frequency_table.delitel

        self.value = np.float64(0)
        self.decoding: bool = False

    def init_encoding(self):
        self.h = np.zeros(np.uint(self.bitstream.size+1))
        self.h[0] = self.high
        self.l = np.zeros(np.uint(self.bitstream.size+1))
        self.i: int = 0

    def encode_byte(self, byte : np.uint8):
        j = byte
        self.i += 1

        
        l_prev = self.l[self.i-1]
        h_prev = self.h[self.i-1]
        diff_prev = h_prev - l_prev + 1


        self.l[self.i] = l_prev + self.frequency_table.getProbability(j-1) * diff_prev // self.delitel
        self.h[self.i] = l_prev + self.frequency_table.getProbability(j) * diff_prev // self.delitel - 1

        li = self.l[self.i]
        hi = self.h[self.i]

        while True:
            if hi < self.half:
                self.bitstream.BitsPlusFollow(0, self.bits_to_follow)
                self.bits_to_follow = 0
            elif li >= self.half:
                self.bitstream.BitsPlusFollow(1, self.bits_to_follow)
                self.bits_to_follow = 0
                li -= self.half
                hi -= self.half
            elif li >= self.first_qtr and hi < self.third_qtr:
                self.bits_to_follow += 1
                li -= self.first_qtr
                hi -= self.first_qtr
            else:
                break

            li += li
            hi += hi + 1
        self.l[self.i] = li
        self.h[self.i] = hi


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
    def decode_byte(self) -> np.uint8:
        self.i += 1
        
        
        l_prev = self.l[self.i-1]
        h_prev = self.h[self.i-1]
        diff_prev = h_prev - l_prev + 1

        freq = ((self.value - l_prev + 1) * self.delitel + 1) / diff_prev
        j = 0
        while self.frequency_table.b[j+1] <= freq:
            j += 1

        
        self.l[self.i] = l_prev + self.frequency_table.getProbability(j-1) * diff_prev // self.delitel
        self.h[self.i] = l_prev + self.frequency_table.getProbability(j) * diff_prev // self.delitel - 1

        li = self.l[self.i]
        hi = self.h[self.i]

        while True:
            if hi < self.half:
                pass
            elif li >= self.half:
                delta = self.half
                li -= delta
                hi -= delta
                self.value -= delta
            elif li >= self.first_qtr and hi < self.third_qtr:
                delta = self.first_qtr
                li -= delta
                hi -= delta
                self.value -= delta
            else:
                break

            li += li
            hi += hi + 1
            self.value += self.value + self.bitstream.read_bit()

        self.l[self.i] = li
        self.h[self.i] = hi

        return np.uint8(j)
    

def compress_ari(ifile : str, ofile : str):
    """PUT YOUR CODE HERE
       implement an arithmetic encoding algorithm for compression
    Parameters
    ----------
    ifile: str
        Name of input file
    ofile: str
        Name of output file
    """
    FreqTable = FrequencyTable()

    bitStream = BitStream(ifile, ofile)
    bitStream.write_size()

    AC = ArithmeticCompressor(bitStream, FreqTable)
    AC.init_encoding()
    while(not AC.bitstream.EOF()):
        c = AC.bitstream.read_byte()
        if AC.bitstream.EOF():
            AC.bitstream.BitsPlusFollow(1, AC.bits_to_follow)
            break
        AC.encode_byte(c)
        AC.frequency_table.update(c)

    AC.bitstream.close()
    


def decompress_ari(ifile : str, ofile : str):
    """PUT YOUR CODE HERE
       implement an arithmetic decoding algorithm for decompression
    Parameters
    ----------
    ifile: str
        Name of input file
    ofile: str
        Name of output file
    """
    FreqTable = FrequencyTable()
    bitStream = BitStream(ifile, ofile)

    AC = ArithmeticCompressor(bitStream, FreqTable)

    AC.init_decoding(AC.bits_high)
    while(AC.bitstream.size > AC.bitstream.output_size):
        c = AC.decode_byte()
        AC.frequency_table.update(c)
        AC.bitstream.write_byte(c)
    AC.bitstream.close()

