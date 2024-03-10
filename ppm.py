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

    def BitsPlusFollow(self, bit: np.uint8, bits_to_follow: np.uint):
        self.write_bit(bit)
        while(bits_to_follow > 0):
            self.write_bit(not bit)
            bits_to_follow -= 1



class FrequencyTable:
    def __init__(self, size:np.uint):
        self.size: np.uint = size
        self.table = np.ones(self.size, dtype=np.uint64)
        self.b = np.zeros(self.size+1)
        
        self.delitel: np.uint = 1000
        self.high: np.uint32 = 0x100000000 - 1
        self.bits_high: np.uint8 = 32
        self.aggressiveness: np.uint = 5000
        self.generate_probability()
        self.lim: np.uint = self.aggressiveness*50000

        ### test_3: agr = 1000, lim = agr*1000
        ### test_7: arg = 3000, lim = 100000
        ### 3695610 - 15000, 15000*500/ 10000, 10000*500

    def generate_probability(self):
        multiplier = self.delitel / self.size
        self.b[1:] = np.cumsum(self.table * multiplier)

    def update(self, byte : np.uint64):
        self.size += self.aggressiveness
        self.table[byte] += self.aggressiveness
        if self.table[byte] >= self.lim:
            self.rescale()
        self.generate_probability()

    def rescale(self):
        # self.table = self.table - (self.table >> 1)
        
        for i in range(self.table.size):
            if self.table[i] > 2:
                self.size -= self.table[i]//2
                self.table[i] //= 2
        # self.size = np.sum(self.table)

    def getProbability(self, index: np.uint64):
        return self.b[np.uint64(index+1)]



class ArithmeticCompressor:
    def __init__(self, bitstream : BitStream, frequency_table : FrequencyTable):
        self.bitstream = bitstream
        self.frequency_table = frequency_table
        self.high = self.frequency_table.high
        self.bits_high = self.frequency_table.bits_high
        self.first_qtr:np.uint = np.uint(np.uint(self.high + 1)>>np.uint(2))
        self.half:np.uint = 2*self.first_qtr
        self.third_qtr:np.uint = 3*self.first_qtr
        self.bits_to_follow: np.uint8 = np.uint8(0)
        self.delitel = self.frequency_table.delitel

        self.value = np.float64(0)

    def init_arithmetic(self):
        self.h = np.zeros(np.uint(2*self.bitstream.size+1))
        self.h[0] = np.uint(self.high)
        self.l = np.zeros(np.uint(2*self.bitstream.size+1))
        self.i = 0
    def init_encoding(self):
        self.init_arithmetic()
        # print("B: ", self.frequency_table.b.size, self.frequency_table.b, "TABLE: ", self.frequency_table.table.size, self.frequency_table.table)

    def encode_byte(self, byte : np.uint):
        j = np.uint64(byte)
        self.i += 1

        l_prev = self.l[self.i-1]
        h_prev = self.h[self.i-1]
        diff_prev = h_prev - l_prev + 1
        # print(diff_prev, l_prev, h_prev)

        
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
                # print(li, hi)
                li -= self.half
                hi -= self.half
                # print(li, hi)
            elif li >= self.first_qtr and hi < self.third_qtr:
                self.bits_to_follow += 1
                li -= self.first_qtr
                hi -= self.first_qtr
            else:
                break
            # print(li, hi)

            li += li
            hi += hi + 1
        
        self.l[self.i] = li
        self.h[self.i] = hi
        # print(self.frequency_table.table)
        # print(li, hi)
        # print(0.0,self.first_qtr, self.half, self.third_qtr, self.high)


    def init_decoding(self):
        self.init_arithmetic()
        # self.set_value(factor)
        # self.value = 0

    def set_value(self, factor:np.uint8):
        # factor //= 8
        self.value:np.uint64 = np.uint64(0)
        for _ in range(factor):
            self.value += self.value + self.bitstream.read_bit() 
    def decode_byte(self) -> np.uint16:
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
        # print(self.frequency_table.table)
        # print(0.0,self.first_qtr, self.half, self.third_qtr, self.high)
        if j ==256:
            return 256
        return np.uint8(j)
    
class ContextModel:
    def __init__(self, degree):
        self.table_size = 257
        self.esc = self.table_size-1
        self.FT = FrequencyTable(self.table_size)
        self.degree = degree
        
class Models:
    def __init__(self, AC:ArithmeticCompressor, k_lim):
        self.cm: np.array = np.empty(257, dtype=ContextModel)
        self.SP = 0
        self.stack = np.empty(2, dtype=ContextModel)
        self.AC:ArithmeticCompressor = AC
        self.k = k_lim
    def init_model(self):

        self.cm[256] = ContextModel(0)
        self.cm[256].FT.lim = self.cm[256].FT.aggressiveness*self.k
        # for i in range(257):
        #     self.cm[256].FT.table[i] += 1
        #     self.cm[256].FT.size += 1
        self.cm[256].FT.generate_probability()
        for i in range(256):
            self.cm[i] = ContextModel(1)
            self.cm[i].FT.lim = self.cm[i].FT.aggressiveness*self.k
        # print(self.cm)

    def encode(self, CM:ContextModel, c:np.uint16):
        self.AC.frequency_table = CM.FT
        self.stack[self.SP] = CM
        self.SP += 1
        # print(f"CM:{CM.degree}")
        if (CM.degree==0)or(CM.FT.table[c] > 1):
            # print("C_encode", np.uint8(c), np.uint8(c).tobytes())
            self.AC.encode_byte(c)
            return True
        else:
            if ((CM.FT.table[CM.esc]) > 1):
                # print("C_encode", CM.esc, np.uint8(CM.esc).tobytes())
                self.AC.encode_byte(CM.esc)
                self.update(CM, CM.esc)
            return False

    def decode(self, CM:ContextModel, c:np.array):
        self.AC.frequency_table = CM.FT
        self.stack[self.SP] = CM
        self.SP += 1
        # print(f"CM:{CM.degree}")
        if (CM.degree!=0)and(CM.FT.table[CM.esc] <= 1):
            return False
        
        d = self.AC.decode_byte()
        # print("C_decode: ", d, np.uint8(d).tobytes())
        if d==CM.esc:
            c[0] = 0
            self.update(CM, CM.esc)
            return False
        else:
            # self.update(CM,d)
            c[0] = d
            return True

    def update(self,CM:ContextModel, byte : np.uint64):
        CM.FT.size += CM.FT.aggressiveness
        CM.FT.table[byte] += CM.FT.aggressiveness
        if CM.FT.table[byte] >= CM.FT.lim:
            CM.FT.rescale()
        CM.FT.generate_probability()

    def update_model(self, c:np.uint16):
        # print("SP: ",self.SP)
        while(self.SP):
            self.SP -= 1
            
            CM: ContextModel = self.stack[self.SP]
            if CM.FT.table[c] >= CM.FT.lim:
                CM.FT.rescale()
            agr = CM.FT.aggressiveness
            if (CM.FT.table[c] <= 1):
                CM.FT.table[CM.esc] += agr
                CM.FT.size += agr
            CM.FT.size += agr
            CM.FT.table[c] += agr
            CM.FT.generate_probability()
    def __getitem__(self, item):
        return self.cm[item]



def compress_ppm(ifile : str, ofile : str):

    bitStream = BitStream(ifile, ofile)
    bitStream.write_size()

    k = 50*(bitStream.input_size//1000)
    AC:ArithmeticCompressor = ArithmeticCompressor(bitStream, ContextModel(0).FT)
    cm = Models(AC, k)
    context = np.array([0], dtype=np.uint8)

    AC.init_encoding()
    cm.init_model()

    success = False
    # l:np.uint64 = 0
    # success = False
    while(not AC.bitstream.EOF()):
        c = AC.bitstream.read_byte()
        if AC.bitstream.EOF():
            AC.bitstream.BitsPlusFollow(1, AC.bits_to_follow)
            break

        # print("C: ", c)

        success = cm.encode(cm[context[0]], c)
        C= cm[context[0]]
        if not success:
            # print(cm[256].FT.b)
            cm.encode(cm[256], c)
            C = cm[256]
        cm.update_model(c)
        # print(C.FT.table)
        context[0] = c
    # if success:
    #     # AC.bitstream.BitsPlusFollow(1, AC.bits_to_follow)
    #     pass
    # AC.frequency_table = cm[context[0]].FT
    # # AC.bitstream.BitsPlusFollow(1, AC.bits_to_follow)
    # cm.encode(cm[context[0]], cm[context[0]].esc)
    # if not success:
    #     # AC.bitstream.BitsPlusFollow(1, AC.bits_to_follow)
    #     pass
    # AC.frequency_table = cm[256].FT
    # # AC.bitstream.BitsPlusFollow(1, AC.bits_to_follow)
    # cm.encode(cm[256], cm[256].esc)
    # AC.bitstream.BitsPlusFollow(1, AC.bits_to_follow)
    # print(cm[256].FT.table)
    AC.bitstream.close()
    


def decompress_ppm(ifile : str, ofile : str):

    bitStream = BitStream(ifile, ofile)
    fsize = bitStream.read_size()
    
    k = 50*(fsize//1000)
    AC:ArithmeticCompressor = ArithmeticCompressor(bitStream, ContextModel(0).FT)
    cm = Models(AC,k)
    context = np.array([0], dtype=np.uint8)

    AC.init_decoding()
    AC.set_value(32)
    cm.init_model()
    # print(AC.value)
    c = np.array([0], dtype=np.uint8)
    success = False

    while(fsize > bitStream.output_size):
        success = cm.decode(cm[context[0]], c)
        C = cm[context[0]]
        if not success:
            # print(cm[256].FT.b)
            success = cm.decode(cm[256], c)
            C = cm[256]
            if not success:
                break

        cm.update_model(c[0])
        # print(C.FT.table)
        context[0] = c[0]
        AC.bitstream.write_byte(c[0])
    # print(fsize, bitStream.output_size)

    AC.bitstream.close()

