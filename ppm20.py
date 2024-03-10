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
    def __init__(self, size:int):
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

    def update(self, byte : np.uint64):
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
        self.h = np.zeros(np.uint(2*self.bitstream.size+1))
        self.h[0] = np.uint(self.high)
        self.l = np.zeros(np.uint(2*self.bitstream.size+1))
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

    def set_value(self, factor:np.uint8):
        # factor //= 8
        self.value:np.uint64 = np.uint64(0)
        for _ in range(factor):
            self.value += self.value + self.bitstream.read_bit() 
    def decode_byte(self) -> np.uint64:
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

        return np.uint64(j)
    
class ContextModel:
    def __init__(self, degree: int):
        self.table_size = 256**(degree+1)
        self.esc = self.table_size
        self.FT = FrequencyTable(self.table_size)
        self.context = np.empty(degree)
        self.degree = degree
        # self.AC.init_encoding()

    def encode(self,AC:ArithmeticCompressor, context: np.array, symbol: np.uint8) -> bool:
        print(f"CM({self.degree}), symbol: {symbol}")
        if self.degree==0:
            AC.encode_byte(symbol)
            self.FT.update(symbol)
            return True

        index: np.int64 = 0
        for c in context:
            index += index*(256-1) + c
            print("Context: ",context[(len(context)-self.degree):],(len(context)-self.degree))
        index += index*(256-1) + symbol
        
        print(index, context, self.degree, "b_len: ", len(self.FT.b), "len_table: ", len(self.FT.table), len(AC.frequency_table.table))
        if (self.FT.table[index] > 1):
            AC.encode_byte(index)
            self.FT.update(index)
            return True
        
        AC.encode_byte(self.esc)
        self.FT.update(self.esc)
        self.FT.update(index)
        return False

    def decode(self, AC:ArithmeticCompressor):
        c = AC.decode_byte()
        self.FT.update(c)
        if c==self.esc:
            return c
        return c

        
    def update_model(self, context: np.array, c:np.uint64):
        
        index = 0
        for i in range(self.degree):
            index += 255*index + context[-i]
            print("CONTUXT",context[-i])
        index += 255*index + c
        print('UPDATE: ', context, c, index, "CM", self.degree)
        self.FT.update(index)
        


def compress_ppm(ifile : str, ofile : str):

    bitStream = BitStream(ifile, ofile)
    bitStream.write_size()

    cm = np.array([ContextModel(0), ContextModel(1)])
    AC:ArithmeticCompressor = ArithmeticCompressor(bitStream, cm[0].FT)
    context = np.zeros(cm.size-1, dtype=np.uint8)
    len_model = len(cm)-1

    AC.init_encoding()

    model = len_model
    CM:ContextModel = cm[model]
    con_size = 0
    # l:np.uint64 = 0
    # success = False
    while(not AC.bitstream.EOF()):
        c = AC.bitstream.read_byte()
        if AC.bitstream.EOF():
            AC.bitstream.BitsPlusFollow(1, AC.bits_to_follow)
            break
        # for i in range(len(cm)-model):
        #     cm[i].update_model(context, c)
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

        while not (CM.encode(AC, context[-model:], c)):
            
            model -= 1
            CM = cm[model]
            AC.frequency_table = CM.FT
            for i in range(len(cm)-model):
                cm[i].update_model(context,CM.esc)
            print("b: ", len(CM.FT.b), "Table: ", len(CM.FT.table), len(AC.frequency_table.table))
            

        for i in range(len(cm)-model):
            cm[i].update_model(context, c)

        context = np.roll(context, -1)
        context[-1] = c
        
        # l += 1
        model = len_model
        CM = cm[model]
        AC.frequency_table = CM.FT

    AC.bitstream.close()
    


def decompress_ppm(ifile : str, ofile : str):

    bitStream = BitStream(ifile, ofile)
    fsize = bitStream.read_size()

    cm = np.array([ContextModel(0), ContextModel(1)])
    AC:ArithmeticCompressor = ArithmeticCompressor(bitStream, cm[0].FT)
    context = np.zeros(cm.size-1, dtype=np.uint8)
    len_model = len(cm)-1

    AC.init_decoding()
    AC.set_value(64)

    model = 0
    CM:ContextModel = cm[model]

    while(fsize > bitStream.output_size):
        # while not ((c:=CM.decode(context))==CM.esc):
        #     model -= 1
        #     CM = cm[model]
        #     AC = CM.AC
        #     for i in range(len_model+1):
        #         cm[i].AC.frequency_table.update(c)
        # for i in range(len_model+1):
        #     cm[i].AC.frequency_table.update(c)
            
        
        # model = len_model
        # CM = cm[model]
        
        # while not (CM.decode(AC, context[-model:], c)):
            
        #     model -= 1
        #     CM = cm[model]
        #     AC.frequency_table = CM.FT
        #     for i in range(len(cm)-model):
        #         cm[i].update_model(context,CM.esc)
        #     print("b: ", len(CM.FT.b), "Table: ", len(CM.FT.table), len(AC.frequency_table.table))
        while (c:=CM.decode(AC))==CM.esc:
            model -= 1
            CM = cm[model]
            AC.frequency_table = CM.FT
            for i in range(len(cm)-model):
                cm[i].update_model(context,CM.esc)
            print("b: ", len(CM.FT.b), "Table: ", len(CM.FT.table), len(AC.frequency_table.table))
        
        i = CM.degree
        m = np.uint64(256**(i+1) - 1)
        print("C: ", c, "DEG: ", CM.degree)
        while(m):
            context = np.roll(context, -1)
            print((r :=np.uint8((c & m)>> np.uint64(8*i))).tobytes(), hex(m), i*8)
            context[-1] = r
            AC.bitstream.write_byte(r)
            i -= 1
            m = m>>np.uint64(8)

        for i in range(len(cm)-model):
            cm[i].update_model(context, r)
        model = len_model
        CM = cm[model]
        AC.frequency_table = CM.FT
        print('loop')
    AC.bitstream.close()

