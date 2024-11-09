# Built-in Types
const BoxIndex = UInt8     # Index of a box in the list of boxes
const GoalIndex = BoxIndex # Index of a goal in the list of boxes
const TileIndex = Int32    # The index of a tile in a board graph - ie a position on the board
const TileValue = UInt8    # The contents of a board tile, made up of flags
const ReachabilityCount = UInt32

const FLOOR = zero(TileValue) # An empty tile value

# Tile flags
const WALL                    = one(TileValue)<<0
const BOX                     = one(TileValue)<<1
const GOAL                    = one(TileValue)<<2
const PLAYER                  = one(TileValue)<<3
const BOARD_PIECES            = BOX + GOAL + PLAYER

# Move directions
const DIR_UP = 0x01
const DIR_LEFT = 0x02
const DIR_DOWN = 0x03
const DIR_RIGHT = 0x04
const Direction = UInt8
const DIRECTIONS = DIR_UP:DIR_RIGHT
const DIR_DXS = [0, -1, 0, 1]
const DIR_DYS = [-1, 0, 1, 0]
const N_DIRS = length(DIRECTIONS)
const NEXT_DIRECTION = [DIRECTIONS[mod1(dir+1,4)] for dir in DIRECTIONS]
const OPPOSITE_DIRECTION = [DIRECTIONS[mod1(dir+2,4)] for dir in DIRECTIONS]

# Board characters
const CH_BOX              = 'b'
const CH_BOX_ON_GOAL      = 'B'
const CH_GOAL             = '.'
const CH_FLOOR            = ' '
const CH_NON_BLANK_FLOOR  = '-'
const CH_PLAYER           = 'p'
const CH_PLAYER_ON_GOAL   = 'P'
const CH_SQUARE_SET       = '%'
const CH_WALL             = '#'

"""
A Board is a state represented as a list of tiles alongside the static game
information like floors / walls / goals.
A valid board state will have walls at the borders so we do not have to do special-case checking.
"""
const Board = Matrix{TileValue}