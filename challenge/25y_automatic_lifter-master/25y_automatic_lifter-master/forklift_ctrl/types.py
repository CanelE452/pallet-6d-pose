from enum import Enum

class MessageFlag:
    STD = 0x0000  # 11-bit ID
    EXT = 0x0004  # 29-bit ID

class OperationMode(Enum):
    DRIVING = "driving"
    LIFT = "lift"
    FOLDING = "folding"
    REACH = "reach"
