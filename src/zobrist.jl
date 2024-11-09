"""
This method computes the components used when calculating Zobrist hash-keys.
Zobrist hashing starts by randomly generating bitstrings for each possible element of a board game.
Now any board configuration can be broken up into independent piece/position components, which are mapped to the
random bitstrings. The final Zobrist hash is computed by combining those bitstrings using bitwise XOR.

Only box positions are hashed. The player position is saved separately.
Note that two states are only equal if both their zobrist hashes match
and their player positions (or reachable player positions) match.
As such, we make checking for equality easier by placeing the player in the reachable tile with the lowest tile index.

Rather than computing the hash for the entire board every time,
the Zobrist hash value of a board can be updated simply by XORing out the bitstring(s) for positions that have changed,
and XORing in the bitstrings for the new positions.
This makes Zobrist hashing very efficient for traversing a game tree.
"""
function construct_zobrist_hash_components(n_board_tiles::Integer; seed::UInt64=zero(UInt64))::Vector{UInt64}
    rng = MersenneTwister(seed)
    hash_components = zeros(UInt64, n_board_tiles)
    for i in 1:length(hash_components)
        while hash_components[i] == zero(UInt64)
            hash_components[i] = Random.rand(rng, UInt64)
            # values must be unique
            for j in 1:i-1
                if hash_components[j] == hash_components[i]
                    hash_components[i] = 0
                end
            end
        end
    end
    return hash_components
end

function calculate_zobrist_hash(□_boxes::Vector{TileIndex}, hash_components::Vector{UInt64})::UInt64
    # Only box positions are taken into account;
    # The player position isn't considered, it is saved separately in each state.
    result = zero(UInt64)
    for □ in □_boxes
        result ⊻= hash_components[□]
    end
    return result
end