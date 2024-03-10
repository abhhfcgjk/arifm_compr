import numpy as np
import sys
# np.set_printoptions(threshold=sys.maxsize)

class BitStream:
    def __init__(self, ifile : str, ofile : str):
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
        self.write_buffer_size = 256*256
        self.write_buffer: np.array = np.empty(self.write_buffer_size, dtype=np.uint8)
        # self.write_buffer_size = 0
        self.SP = 0
    
    def read_size(self)->np.uint:
        self.cur = 4 #
        self.size = self.ifp[0] + self.ifp[1]*16*16 + self.ifp[2]*16*16*16*16 + self.ifp[3]*16*16*16*16*16*16 #
        # print(self.size)
        return self.size #

    def write_size(self):
        # print(self.input_size.tobytes(), self.input_size)
        self.ofp.write(self.input_size.tobytes())

    def read_bit(self) -> np.uint8:
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
        if(self.__count_writed_bits_to_byte < 8):
            self.__byte_to_write <<= np.uint8(1)
            self.__byte_to_write |= bit
            self.__count_writed_bits_to_byte += np.uint8(1)
            if(self.__count_writed_bits_to_byte == 8):
                self.write_byte(np.uint8(self.__byte_to_write))
                self.__byte_to_write = np.uint8(0)
                self.__count_writed_bits_to_byte = np.uint8(0)


    def read_byte(self) -> np.uint8:
        
        if self.cur < self.input_size:
            self.__readed_byte = self.ifp[self.cur]
            self.cur += 1

            # print(self.cur)

            return self.__readed_byte
        self.__eof = True
        return 0


    def write_byte(self, byte : np.uint8):
        self.output_size += 1
        self.ofp.write(byte.tobytes())


    def close(self):
        if(self.__count_writed_bits_to_byte != 0):
            self.__byte_to_write <<= np.uint8(8 - self.__count_writed_bits_to_byte)
            self.ofp.write(np.uint8(self.__byte_to_write).tobytes())
        self.ofp.close()

    def EOF(self) ->bool:
        return self.__eof

    def buffer_write_bit(self, bit: np.uint8):
        self.write_buffer[self.SP] = bit
        self.SP += 1
    def BitsPlusFollow(self, bit: np.uint8, bits_to_follow: np.uint):
        self.buffer_write_bit(bit)
        while(bits_to_follow > 0):
            self.buffer_write_bit(not bit)
            bits_to_follow -= 1
    def drop_buffer(self):
        for i in range(self.SP):
            self.write_bit(self.write_buffer[i])
        # self.write_buffer = np.array([],dtype=bytes)
        self.SP = 0
    def write_buf_size(self):
        self.ofp.write(np.uint16(self.SP).tobytes())
    def read_buf_size(self):
        self.cur += 2 #
        return self.ifp[self.cur-2] + self.ifp[self.cur-1]*16*16
        # print(self.size)

class FrequencyTable:
    def __init__(self, size):
        self.size: np.uint = size+1
        self.table = np.ones(self.size, dtype=np.uint64)
        self.b = np.zeros(self.size+1)
        
        self.delitel: np.uint = 1000
        self.high: np.uint32 = 0x10000000000000000 - 1
        self.bits_high: np.uint8 = 64
        self.aggressiveness: np.uint = 5000
        self.generate_probability()
        self.lim: np.uint = self.aggressiveness*50

        ### test_3: agr = 1000, lim = agr*1000
        ### test_7: arg = 3000, lim = 100000
        ### 3695610 - 15000, 15000*500/ 10000, 10000*500

    def generate_probability(self):
        multiplier = self.delitel / self.size
        self.b[1:] = np.cumsum(self.table * multiplier)

    def update(self, byte : np.uint):
        self.size += self.aggressiveness
        self.table[byte] += self.aggressiveness
        if self.table[byte] >= self.lim:
            self.rescale()
        self.generate_probability()

    def rescale(self):
        print(self.table)
        self.table = self.table - (self.table >> 1)
        self.size = np.sum(self.table)

    def getProbability(self, index: np.uint):
        return self.b[index+1]

    def getElem(self, index: np.uint) -> np.uint8:
        return ...


class ArithmeticCompressor:
    def __init__(self, bitstream : BitStream, frequency_table : FrequencyTable):
        self.bitstream = bitstream
        self.frequency_table = frequency_table
        self.high = self.frequency_table.high
        self.bits_high = self.frequency_table.bits_high
        self.first_qtr:np.uint = np.uint((self.high + 1)>>2)
        self.half:np.uint = 2*self.first_qtr
        self.third_qtr:np.uint = 3*self.first_qtr
        self.bits_to_follow: np.uint8 = np.uint8(0)
        self.delitel = self.frequency_table.delitel

        self.value = np.float64(0)
        self.decoding: bool = False

    def init_arithmetic(self):
        self.h = np.zeros(np.uint(self.bitstream.size+1))
        self.h[0] = np.uint(self.high)
        self.l = np.zeros(np.uint(self.bitstream.size+1))
        self.i = 0
    def init_encoding(self):
        self.init_arithmetic()
        print("B: ", self.frequency_table.b.size, self.frequency_table.b, "TABLE: ", self.frequency_table.table.size, self.frequency_table.table)

    def encode_byte(self, byte : np.uint):
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
                print(li, hi)
                li -= self.half
                hi -= self.half
                print(li, hi)
            elif li >= self.first_qtr and hi < self.third_qtr:
                self.bits_to_follow += 1
                li -= self.first_qtr
                hi -= self.first_qtr
            else:
                break
            # print(self.high>=hi,'HALF, HIGH ', self.half, self.high, li, hi)
            if self.high < hi:
                print("BOOOOOOOOBS")
                print(self.frequency_table.table, '\n', self.frequency_table.b, '\n', self.frequency_table.table.size)
            li += li
            hi += hi + 1
        
        self.l[self.i] = li
        self.h[self.i] = hi


    def init_decoding(self):
        self.init_arithmetic()
        # self.set_value(factor)
        # self.value = 0

    def set_value(self, factor:np.uint8, high:np.uint8):
        # factor //= 8
        self.value:np.int64 = np.int64(0)
        for _ in range(factor):
            self.value += self.value + np.int64(self.bitstream.read_bit())
        print(type(self.value), type(high-factor))
        self.value <<= high-factor
    def decode_byte(self) -> np.uint32:
        self.i += 1
        
        
        l_prev = self.l[self.i-1]
        h_prev = self.h[self.i-1]
        diff_prev = h_prev - l_prev + 1

        freq = ((self.value - l_prev + 1) * self.delitel + 1) / diff_prev
        j = 0
        # if(freq >= self.frequency_table.b[-2]):
        #     j = self.frequency_table.size - 1
        # else:
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

        return np.uint32(j)
    
class ContextModel:
    def __init__(self, degree: int, bitstream: BitStream):
        self.table_size = 256**(degree+1)
        self.esc = self.table_size
        self.AC = ArithmeticCompressor(bitstream, FrequencyTable(self.table_size))
        self.context = np.empty(degree)
        self.degree = degree
        # self.AC.init_encoding()

    def encode(self, context: np.array, symbol: np.uint8) -> bool:
        print(f"CM({self.degree})")
        index: np.int64 = 0
        for c in context[(len(context)-self.degree):]:
            index += index*(256-1) + c
            print("Context: ",context[(len(context)-self.degree):],(len(context)-self.degree))
        index += index*(256-1) + symbol
        # print(self.AC.frequency_table.table.size, index, len_context, self.degree)
        if (self.degree == 0)or(self.AC.frequency_table.table[index] > 1):
            self.AC.encode_byte(index)
            self.AC.frequency_table.update(index)
            return True

        self.AC.encode_byte(self.esc)
        self.AC.frequency_table.update(self.esc)

        return False

    def decode(self):
        c = self.AC.decode_byte()
        
        if c==self.esc:
            self.AC.frequency_table.update(c)
            return False
        
        i = self.degree
        m = 256**(i+1) - 1
        while(m):
            print((r :=np.uint8((c & m)>> 8*i)).tobytes(), hex(m), i*8)
            self.AC.bitstream.write_byte(r)
            i -= 1
            m >>= 8
        self.AC.frequency_table.update(c)
        return True
        
                


def compress_ppm(ifile : str, ofile : str):

    bitStream = BitStream(ifile, ofile)
    bitStream.write_size()

    cm = np.array([ContextModel(0,bitStream), ContextModel(1,bitStream)])

    context = np.zeros(cm.size-1, dtype=np.uint8)
    len_model = len(cm)-1

    for i in range(len_model+1):
        cm[i].AC.init_encoding()

    model = 0
    CM:ContextModel = cm[model]
    AC = CM.AC
    # l:np.uint64 = 0
    # success = False
    while(not AC.bitstream.EOF()):
        c = AC.bitstream.read_byte()
        if AC.bitstream.EOF():
            AC.bitstream.BitsPlusFollow(1, AC.bits_to_follow)
            break
        # success = CM.encode(AC, context, len_context)
        # if not success:
        #     model -= 1
        #     CM = cm[model]
        #     success = CM.encode(AC, context, len_context)
        #     if not success:
        #         model -= 1
        #         CM = cm[model]
        #         success = CM.encode(AC, context, len_context)
        print("C: ", c)
        while (not CM.encode(context, c)):
            AC.bitstream.write_buf_size()
            AC.bitstream.drop_buffer()
            model -= 1
            CM = cm[model]
            AC = CM.AC
        print(c, context, end='')
        context = np.roll(context, -1)
        context[-1] = c
        print(context)
        
        # l += 1
        model = len_model
        CM = cm[model]

    AC.bitstream.close()
    


def decompress_ppm(ifile : str, ofile : str):

    bitStream = BitStream(ifile, ofile)
    fsize = bitStream.read_size()

    cm = np.array([ContextModel(0,bitStream), ContextModel(1,bitStream)])
    
    context = np.zeros(cm.size-1, dtype=np.uint8)
    len_model = len(cm)-1

    for i in range(len_model+1):
        cm[i].AC.init_decoding()

    model = 0
    CM:ContextModel = cm[model]
    _high = CM.AC.frequency_table.bits_high
    s = CM.AC.bitstream.read_buf_size()
    CM.AC.set_value(s, _high)
    AC = CM.AC
    l:np.uint64 = 0

    while(fsize > bitStream.output_size):
        while (not CM.decode()):
            model -= 1
            CM = cm[model]
            AC = CM.AC
            CM.AC.set_value(CM.AC.bitstream.read_buf_size(), _high)
        
        model = len_model
        CM = cm[model]
    AC.bitstream.close()

