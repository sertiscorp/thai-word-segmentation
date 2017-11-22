from collections import OrderedDict

class ThaiWordSegmentLabeller:
    # Construct lookup table to convert characters to input labels
    _char_set = [chr(x) for x in [0] + [0x000A] + list(range(0x0020, 0x007F)) + \
            list(range(0x0E01, 0x0E3A)) + list(range(0x0E3F, 0x0E4D)) + list(range(0x0E50, 0x0E5A))]
    _dictionary = OrderedDict(map(tuple, map(reversed, enumerate(_char_set))))
    
    # Convert string to input labels
    def get_input_labels(string):
        return [ThaiWordSegmentLabeller._dictionary.get(char, 0) for char in string]

    # Convert string to output labels
    def get_output_labels(string):
        return [True] + (len(string) - 1) * [False]
    
    def get_input_vocabulary_size():
        return len(ThaiWordSegmentLabeller._dictionary)
    
    def get_output_vocabulary_size():
        return 2